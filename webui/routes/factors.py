"""
因子库相关路由（V3架构）
使用透明因子存储与统一引擎
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from flask import Blueprint, request, jsonify
import math
import threading
import time
from factor_miner.core.factor_evaluator import FactorEvaluator, CrossSectionalEvaluator
from factor_miner.core.factor_engine import get_global_engine
from factor_miner.core.evaluation_io import (
    save_evaluation_results as core_save_evaluation_results,
    load_evaluations as core_load_evaluations,
)

_cs_cancel_event = threading.Event()
# from factor_miner.core.factor_storage import get_global_storage

bp = Blueprint('factors', __name__)

# 因子库路径（V3 扁平化）
FACTOR_LIBRARY_DIR = Path(__file__).parent.parent.parent / "factorlib"
_FACTOR_LIST_CACHE_TTL_SEC = 30
_factor_list_cache = {"expires_at": 0.0, "payload": None}
_factor_list_cache_lock = threading.Lock()

# Alpha101因子公式映射
ALPHA101_FORMULAS = {
    'alpha001': '(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)',
    'alpha002': '(-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))',
    'alpha003': '(-1 * correlation(rank(open), rank(volume), 10))',
    'alpha004': '(-1 * Ts_Rank(rank(low), 9))',
    'alpha005': '(rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))',
    'alpha006': '(-1 * correlation(open, volume, 10))',
    'alpha007': '((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1))',
    'alpha008': '(-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))',
    'alpha009': '((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))',
    'alpha010': 'rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))',
    # 可以继续添加更多Alpha101因子公式...
}

# 页面路由 - 已迁移到main.py

# API路由
@bp.route('/list')
def list_factors():
    """获取因子列表（V4：从新的文件夹结构读取）
    
    因子分类说明：
    - basic_kline: 基础K线因子，仅需OHLCV数据
    - extra_data: 额外数据因子，需要资金费率、持仓量等
    - ml_pretrained: ML预训练因子，需要ML模型或预训练数据
    - event_factor: 事件因子（不建议用于截面评估）
        * 识别规则：category='pattern' 或因子名包含cross/gap/breakout等关键词
        * 特征：仅0/1取值，unique值=2
        * 截面评估缺陷：取值区分度低、数据稀疏、分组效果差
    """
    try:
        now_ts = time.time()
        with _factor_list_cache_lock:
            if _factor_list_cache["payload"] is not None and now_ts < _factor_list_cache["expires_at"]:
                return jsonify(_factor_list_cache["payload"])

        # 检查两个文件夹：technicals 和 minactors
        technicals_dir = FACTOR_LIBRARY_DIR / "technicals" / "definitions"
        minactors_dir = FACTOR_LIBRARY_DIR / "minactors" / "definitions"
        
        factors = []
        
        # 读取 technicals 文件夹
        if technicals_dir.exists():
            for file in technicals_dir.glob("*.json"):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    data['source'] = 'technicals'  # 标记来源
                    factors.append(data)
                except Exception as e:
                    print(f"❌ 读取因子文件失败 {file}: {e}")
                    continue
        
        # 读取 minactors 文件夹
        if minactors_dir.exists():
            for file in minactors_dir.glob("*.json"):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    data['source'] = 'minactors'  # 标记来源
                    factors.append(data)
                except Exception as e:
                    print(f"❌ 读取因子文件失败 {file}: {e}")
                    continue
        
        # 处理因子数据
        processed_factors = []
        for data in factors:
            try:
                comp = data.get('computation_data', {})
                formula_preview = None
                if data.get('computation_type') == 'formula':
                    formula_preview = comp.get('formula') or None
                elif data.get('computation_type') == 'function':
                    formula_preview = comp.get('function_code') or None
                
                # 聚合评估均值（核心IO）
                evaluated = False
                eval_count = 0
                avg_metrics = {}
                last_evaluated_at = None
                try:
                    eval_payload = core_load_evaluations(data.get('factor_id'))
                    evaluations = (eval_payload or {}).get('evaluations') or []
                    eval_count = len(evaluations)
                    if eval_count > 0:
                        evaluated = True
                        keys = ['ic_pearson', 'ic_spearman', 'icir', 'win_rate', 'sharpe_ratio', 'long_short_return']
                        sums = {k: 0.0 for k in keys}
                        counts = {k: 0 for k in keys}
                        for ev in evaluations:
                            res = (ev or {}).get('results') or {}
                            for k in keys:
                                v = res.get(k)
                                if isinstance(v, (int, float)):
                                    sums[k] += float(v)
                                    counts[k] += 1
                            last_evaluated_at = (ev or {}).get('evaluated_at') or last_evaluated_at
                        for k in keys:
                            avg_metrics[k] = (sums[k] / counts[k]) if counts[k] > 0 else None
                except Exception:
                    pass
                
                # 数值安全处理，避免NaN进入JSON
                def _safe_num(x):
                    return float(x) if isinstance(x, (int, float)) and math.isfinite(x) else None
                avg_metrics_clean = {}
                for k, v in (avg_metrics or {}).items():
                    avg_metrics_clean[k] = _safe_num(v)
                
                category = data.get('category', '')
                factor_id = data.get('factor_id', '').lower()
                factor_name = data.get('name', '').lower()
                
                is_event_factor = False
                if category == 'pattern':
                    is_event_factor = True
                else:
                    event_keywords = ['cross', 'gap', 'breakout', 'breakdown', 'signal', 'event', 'direction']
                    for keyword in event_keywords:
                        if keyword in factor_id or keyword in factor_name:
                            is_event_factor = True
                            break
                
                data_requirement = 'basic_kline'
                if is_event_factor:
                    data_requirement = 'event_factor'
                elif category == 'ml':
                    data_requirement = 'ml_pretrained'
                elif category == 'crypto':
                    data_requirement = 'extra_data'
                
                processed_factors.append({
                    'id': data.get('factor_id'),
                    'name': data.get('name'),
                    'description': data.get('description'),
                    'type': data.get('category'),
                    'source': data.get('source'),
                    'data_requirement': data_requirement,
                    'created_at': data.get('metadata', {}).get('created_at'),
                    'computation_type': data.get('computation_type'),
                    'formula': formula_preview,
                    'evaluated': evaluated,
                    'evaluations_count': eval_count,
                    'avg_metrics': avg_metrics_clean,
                    'last_evaluated_at': last_evaluated_at,
                    'ic': _safe_num((avg_metrics or {}).get('ic_pearson')),
                    'ir': _safe_num((avg_metrics or {}).get('icir')),
                    'sharpe': _safe_num((avg_metrics or {}).get('sharpe_ratio')),
                })
            except Exception as e:
                print(f"❌ 处理因子数据失败 {data.get('factor_id', 'unknown')}: {e}")
                continue
        
        response_payload = {'success': True, 'factors': processed_factors, 'total': len(processed_factors)}
        with _factor_list_cache_lock:
            _factor_list_cache["payload"] = response_payload
            _factor_list_cache["expires_at"] = time.time() + _FACTOR_LIST_CACHE_TTL_SEC
        return jsonify(response_payload)
    except Exception as e:
        return jsonify({'success': False, 'message': f'获取因子列表失败: {str(e)}'})


def _evaluate_factor_process_worker(
    factor_id,
    data_dict,
    n_groups,
    normalize_method,
    base_timeframe,
    factor_timeframe,
    factor_bar_mode,
    max_lookback,
    predict_step,
    sample_step,
    min_coverage,
    min_valid_count,
    min_group_size,
    treat_zero_as_invalid,
    enable_data_cleaning,
    remove_zero_volume,
    liquidity_filter_ratio,
    enable_outlier_treatment,
    outlier_method,
    outlier_group_minutes,
    outlier_mad_n,
    outlier_winsor_lower,
    outlier_winsor_upper,
    compute_fsc,
    compute_ic_decay_curve,
    ic_decay_max_lag,
):
    """多进程 worker：单因子截面评估。"""
    t0 = time.time()
    try:
        engine = get_global_engine()
        cs_evaluator = CrossSectionalEvaluator(
            n_groups=n_groups,
            normalize_method=normalize_method,
            predict_step=predict_step,
            sample_step=sample_step,
            base_timeframe=base_timeframe,
            factor_timeframe=factor_timeframe,
            factor_bar_mode=factor_bar_mode,
            max_lookback=max_lookback,
            min_coverage=min_coverage,
            min_valid_count=min_valid_count,
            min_group_size=min_group_size,
            treat_zero_as_invalid=treat_zero_as_invalid,
            enable_data_cleaning=enable_data_cleaning,
            remove_zero_volume=remove_zero_volume,
            liquidity_filter_ratio=liquidity_filter_ratio,
            enable_outlier_treatment=enable_outlier_treatment,
            outlier_method=outlier_method,
            outlier_group_minutes=outlier_group_minutes,
            outlier_mad_n=outlier_mad_n,
            outlier_winsor_lower=outlier_winsor_lower,
            outlier_winsor_upper=outlier_winsor_upper,
            compute_fsc=compute_fsc,
            compute_ic_decay_curve=compute_ic_decay_curve,
            ic_decay_max_lag=ic_decay_max_lag,
        )
        result = cs_evaluator.evaluate_cross_sectional(data_dict, factor_id, engine, timeframe=base_timeframe)
        elapsed = time.time() - t0
        if result.get('success'):
            return {
                'factor_id': factor_id,
                'success': True,
                'n_symbols': result.get('n_symbols'),
                'n_periods': result.get('n_periods'),
                'n_periods_total': result.get('n_periods_total'),
                'n_periods_ic': result.get('n_periods_ic'),
                'n_periods_returns': result.get('n_periods_returns'),
                'ic': result.get('ic'),
                'returns': result.get('returns'),
                'coverage': result.get('coverage'),
                'summary': result.get('summary'),
                'elapsed': round(elapsed, 2)
            }
        return {
            'factor_id': factor_id,
            'success': False,
            'message': result.get('message', '评估失败'),
            'elapsed': round(elapsed, 2)
        }
    except Exception as ex:
        elapsed = time.time() - t0
        return {
            'factor_id': factor_id,
            'success': False,
            'message': str(ex),
            'elapsed': round(elapsed, 2)
        }

def get_traditional_factor_formula(_):
    """V3不再从旧CSV推导传统指标公式，保留占位。"""
    return None

@bp.route('/evaluate', methods=['POST'])
def evaluate_factor():
    """评估因子"""
    try:
        data = request.get_json()
        
        factor_id = data.get('factor_id')
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        exchange = data.get('exchange', 'binance')
        trade_type = data.get('trade_type', 'futures')
        
        if not all([factor_id, symbol, timeframe, start_date, end_date]):
            return jsonify({
                'success': False,
                'message': '缺少必要参数'
            })
        
        # 使用V3引擎直接计算
        engine = get_global_engine()
        # 加载本地数据
        market_data = load_local_market_data(symbol, timeframe, start_date, end_date, exchange, trade_type)
        
        if market_data is None or market_data.empty:
            return jsonify({
                'success': False,
                'message': '无法加载市场数据'
            })
        
        # 计算因子值（V3）
        factor_values = engine.compute_single_factor(factor_id, market_data)
        
        if factor_values is None:
            return jsonify({
                'success': False,
                'message': '因子计算失败'
            })
        
        # 评估因子
        evaluator = FactorEvaluator()
        
        # 简化对齐策略：强制以 market_data.index 为准
        market_data = market_data.sort_index()
        market_data['returns'] = market_data['close'].pct_change()
        
        # 确保因子为Series
        if hasattr(factor_values, 'columns'):
            try:
                factor_values = factor_values.iloc[:, 0]
            except Exception:
                factor_values = factor_values.squeeze()

        # 直接按市场数据索引重建，窗口期产生的NaN后续一起过滤
        factor_values = factor_values.reindex(market_data.index)
        returns = market_data['returns']

        # 同步过滤非空并保证样本数量
        mask = factor_values.notna() & returns.notna()
        factor_values = factor_values[mask]
        returns = returns[mask]

        if len(factor_values) < 30:
            return jsonify({
                'success': False,
                'message': f'数据不足：样本数 {len(factor_values)} < 30，请扩大时间范围或选择更低频时间框架'
            })
        
        # 评估因子
        evaluation_results = evaluator.evaluate_single_factor(
            factor=factor_values,
            returns=returns,
            factor_name=factor_id
        )
        
        # 保存评估结果
        save_evaluation_results(factor_id, evaluation_results, {
            'symbol': symbol,
            'timeframe': timeframe,
            'start_date': start_date,
            'end_date': end_date
        })
        
        return jsonify({
            'success': True,
            'message': '因子评估完成',
            'results': evaluation_results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'因子评估失败: {str(e)}'
        })

@bp.route('/detail/<factor_id>')
def get_factor_detail(factor_id):
    """获取因子详情"""
    try:
        factor_info = parse_factor_id(factor_id)
        if not factor_info:
            return jsonify({
                'success': False,
                'message': '无效的因子ID'
            })
        
        # 获取因子详细信息
        factor_detail = get_factor_detail_info(factor_info)
        
        return jsonify({
            'success': True,
            'factor': factor_detail
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取因子详情失败: {str(e)}'
        })

@bp.route('/export/<factor_id>')
def export_factor(factor_id):
    """导出因子"""
    try:
        # V4导出：从新的文件夹结构导出
        export_dir = FACTOR_LIBRARY_DIR / "minactors"  # 导出到minactors目录
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # 先在minactors中查找
        definition_file = FACTOR_LIBRARY_DIR / "minactors" / "definitions" / f"{factor_id}.json"
        if not definition_file.exists():
            # 再在technicals中查找
            definition_file = FACTOR_LIBRARY_DIR / "technicals" / "definitions" / f"{factor_id}.json"
            if not definition_file.exists():
                return jsonify({'success': False, 'message': '因子定义不存在'})
        
        export_path = export_dir / f"{factor_id}_definition.json"
        with open(definition_file, 'r', encoding='utf-8') as src, open(export_path, 'w', encoding='utf-8') as dst:
            dst.write(src.read())
        
        return jsonify({'success': True, 'message': '因子导出成功', 'export_path': str(export_path)})
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'因子导出失败: {str(e)}'
        })

@bp.route('/evaluations/<factor_id>')
def get_evaluations(factor_id: str):
    """获取某因子的历史评估记录（多结果结构）"""
    try:
        payload = core_load_evaluations(factor_id)
        return jsonify({'success': True, 'factor_id': payload.get('factor_id', factor_id), 'evaluations': payload.get('evaluations', [])})
    except Exception as e:
        return jsonify({'success': False, 'message': f'获取评估历史失败: {str(e)}'})

@bp.route('/batch_evaluate', methods=['POST'])
def batch_evaluate():
    """批量评估（SSE流式+并发优化）：多个因子*多个交易对*多个时间框架"""
    import time
    import logging
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from flask import Response, stream_with_context

    logger = logging.getLogger(__name__)
    
    payload = request.get_json() or {}
    factor_ids = payload.get('factor_ids') or []
    symbols = payload.get('symbols') or []
    timeframes = payload.get('timeframes') or []
    start_date = payload.get('start_date')
    end_date = payload.get('end_date')
    exchange = payload.get('exchange', 'binance')
    trade_type = payload.get('trade_type', 'futures')

    if not factor_ids or not symbols or not timeframes or not start_date or not end_date:
        return jsonify({'success': False, 'message': '缺少必要参数（factor_ids/symbols/timeframes/start_date/end_date）'})

    total_tasks = len(factor_ids) * len(symbols) * len(timeframes)
    logger.info(f"批量评估请求: {len(factor_ids)} 因子 × {len(symbols)} 币种 × {len(timeframes)} 时间框架 = {total_tasks} 任务")

    def generate():
        _cs_cancel_event.clear()
        try:
            engine = get_global_engine()
            evaluator = FactorEvaluator()
            all_results = []
            completed_count = 0
            t0 = time.time()

            yield _sse_event('progress', {
                'phase': 'evaluating',
                'message': f'开始批量评估，共 {total_tasks} 个任务...',
                'completed': 0,
                'total': total_tasks
            })

            def evaluate_single_task(factor_id, symbol, timeframe):
                if _cs_cancel_event.is_set():
                    return {
                        'factor_id': factor_id, 'symbol': symbol, 'timeframe': timeframe,
                        'success': False, 'message': '评估已取消'
                    }
                try:
                    market_data = load_local_market_data(symbol, timeframe, start_date, end_date, exchange, trade_type)
                    if market_data is None or market_data.empty:
                        return {
                            'factor_id': factor_id, 'symbol': symbol, 'timeframe': timeframe,
                            'success': False, 'message': '无法加载市场数据'
                        }
                    factor_values = engine.compute_single_factor(factor_id, market_data)
                    if factor_values is None:
                        return {
                            'factor_id': factor_id, 'symbol': symbol, 'timeframe': timeframe,
                            'success': False, 'message': '因子计算失败'
                        }
                    market_data = market_data.sort_index()
                    market_data['returns'] = market_data['close'].pct_change()
                    if hasattr(factor_values, 'columns'):
                        try:
                            factor_values = factor_values.iloc[:, 0]
                        except Exception:
                            factor_values = factor_values.squeeze()
                    factor_values = factor_values.reindex(market_data.index)
                    returns = market_data['returns']
                    mask = factor_values.notna() & returns.notna()
                    factor_values = factor_values[mask]
                    returns = returns[mask]
                    if len(factor_values) < 30:
                        return {
                            'factor_id': factor_id, 'symbol': symbol, 'timeframe': timeframe,
                            'success': False, 'message': f'数据不足：样本数 {len(factor_values)} < 30'
                        }
                    eval_res = evaluator.evaluate_single_factor(factor=factor_values, returns=returns, factor_name=factor_id)
                    return {
                        'factor_id': factor_id, 'symbol': symbol, 'timeframe': timeframe,
                        'success': True, 'results': eval_res
                    }
                except Exception as ex:
                    return {
                        'factor_id': factor_id, 'symbol': symbol, 'timeframe': timeframe,
                        'success': False, 'message': str(ex)
                    }

            max_workers = min(total_tasks, os.cpu_count() or 4, 8)
            executor = ThreadPoolExecutor(max_workers=max_workers)
            try:
                futures = {}
                for factor_id in factor_ids:
                    for symbol in symbols:
                        for timeframe in timeframes:
                            future = executor.submit(evaluate_single_task, factor_id, symbol, timeframe)
                            futures[future] = (factor_id, symbol, timeframe)

                for future in as_completed(futures):
                    if _cs_cancel_event.is_set():
                        break
                    result = future.result()
                    all_results.append(result)
                    completed_count += 1
                    
                    result_clean = _sanitize_for_json(result)
                    yield _sse_event('task_result', {
                        'result': result_clean,
                        'completed': completed_count,
                        'total': total_tasks,
                        'message': f'已完成 {completed_count}/{total_tasks}'
                    })
            except GeneratorExit:
                _cs_cancel_event.set()
                for f in futures:
                    f.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            except KeyboardInterrupt:
                _cs_cancel_event.set()
                for f in futures:
                    f.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=True)

            elapsed = time.time() - t0
            success_count = sum(1 for r in all_results if r.get('success'))
            fail_count = total_tasks - success_count
            logger.info(f"批量评估完成: 成功 {success_count}/{total_tasks}, 失败 {fail_count}, 耗时 {elapsed:.1f}s")

            yield _sse_event('done', {
                'total': total_tasks,
                'success_count': success_count,
                'fail_count': fail_count,
                'elapsed': round(elapsed, 2)
            })
        except Exception as e:
            logger.error(f"批量评估异常: {e}", exc_info=True)
            yield _sse_event('error', {'message': f'批量评估失败: {str(e)}'})
            yield _sse_event('done', {'total': total_tasks, 'success_count': 0, 'fail_count': total_tasks})

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        }
    )


@bp.route('/cross_sectional_evaluate', methods=['POST'])
def cross_sectional_evaluate():
    """
    截面因子评估（SSE 流式 + 并行优化）
    - 市场数据仅加载一次（所有因子共享）
    - 多因子并行计算（ThreadPoolExecutor）
    - 结果逐条流式返回（SSE），前端实时更新进度
    """
    import logging
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, FIRST_COMPLETED, as_completed
    from flask import Response, stream_with_context

    logger = logging.getLogger(__name__)
    
    payload = request.get_json() or {}
    factor_ids = payload.get('factor_ids') or []
    symbols = payload.get('symbols') or []
    base_timeframe = payload.get('base_timeframe') or payload.get('timeframe', '1h')
    factor_timeframe = payload.get('factor_timeframe') or base_timeframe
    factor_bar_mode = str(payload.get('factor_bar_mode', 'completed')).lower()
    start_date = payload.get('start_date')
    end_date = payload.get('end_date')
    exchange = payload.get('exchange', 'binance')
    trade_type = payload.get('trade_type', 'futures')
    n_groups = payload.get('n_groups', 5)
    normalize_method = payload.get('normalize_method', 'rank')
    predict_step = payload.get('predict_step', 1)
    sample_step = payload.get('sample_step', 1)
    max_lookback = payload.get('max_lookback', 200)
    min_coverage = payload.get('min_coverage', 0.3)
    min_valid_count = payload.get('min_valid_count', 30)
    min_group_size = payload.get('min_group_size', 5)
    treat_zero_as_invalid = payload.get('treat_zero_as_invalid', True)
    enable_data_cleaning = payload.get('enable_data_cleaning', False)
    remove_zero_volume = payload.get('remove_zero_volume', True)
    liquidity_filter_ratio = payload.get('liquidity_filter_ratio', 0.5)
    enable_outlier_treatment = payload.get('enable_outlier_treatment', False)
    outlier_method = str(payload.get('outlier_method', 'mad')).lower()
    outlier_group_minutes = payload.get('outlier_group_minutes', 30)
    outlier_mad_n = payload.get('outlier_mad_n', 5.0)
    outlier_winsor_lower = payload.get('outlier_winsor_lower', 0.01)
    outlier_winsor_upper = payload.get('outlier_winsor_upper', 0.99)
    compute_fsc = payload.get('compute_fsc', False)
    compute_ic_decay_curve = payload.get('compute_ic_decay_curve', False)
    ic_decay_max_lag = payload.get('ic_decay_max_lag', 5)
    parallel_backend = str(payload.get('parallel_backend', 'thread')).lower()
    try:
        heartbeat_interval_sec = float(payload.get('heartbeat_interval_sec', 15))
    except (TypeError, ValueError):
        heartbeat_interval_sec = 15.0

    logger.info(f"截面评估请求: {len(factor_ids)} 个因子, {len(symbols)} 个币种, {start_date} ~ {end_date}")

    if not factor_ids or not symbols or len(symbols) < 2:
        return jsonify({
            'success': False,
            'message': '截面评估需要至少选择1个因子和2个币种'
        })

    if not start_date or not end_date:
        return jsonify({
            'success': False,
            'message': '请选择日期范围'
        })

    try:
        n_groups = int(n_groups)
    except (TypeError, ValueError):
        return jsonify({
            'success': False,
            'message': '参数 n_groups 必须为整数'
        }), 400

    if n_groups < 2:
        return jsonify({
            'success': False,
            'message': '参数 n_groups 不能小于 2'
        }), 400

    if n_groups > len(symbols):
        n_groups = len(symbols)

    try:
        predict_step = int(predict_step)
    except (TypeError, ValueError):
        return jsonify({
            'success': False,
            'message': '参数 predict_step 必须为整数'
        }), 400

    if predict_step < 1:
        return jsonify({
            'success': False,
            'message': '参数 predict_step 不能小于 1'
        }), 400

    try:
        sample_step = int(sample_step)
    except (TypeError, ValueError):
        return jsonify({
            'success': False,
            'message': '参数 sample_step 必须为整数'
        }), 400

    if sample_step < 1:
        return jsonify({
            'success': False,
            'message': '参数 sample_step 不能小于 1'
        }), 400

    try:
        max_lookback = int(max_lookback)
    except (TypeError, ValueError):
        return jsonify({
            'success': False,
            'message': '参数 max_lookback 必须为整数'
        }), 400

    if max_lookback < 1:
        return jsonify({
            'success': False,
            'message': '参数 max_lookback 不能小于 1'
        }), 400

    try:
        min_coverage = float(min_coverage)
    except (TypeError, ValueError):
        return jsonify({
            'success': False,
            'message': '参数 min_coverage 必须为数值'
        }), 400
    if not (0 <= min_coverage <= 1):
        return jsonify({
            'success': False,
            'message': '参数 min_coverage 必须在 [0, 1] 区间'
        }), 400

    try:
        min_valid_count = int(min_valid_count)
    except (TypeError, ValueError):
        return jsonify({
            'success': False,
            'message': '参数 min_valid_count 必须为整数'
        }), 400
    if min_valid_count < 1:
        return jsonify({
            'success': False,
            'message': '参数 min_valid_count 不能小于 1'
        }), 400

    try:
        min_group_size = int(min_group_size)
    except (TypeError, ValueError):
        return jsonify({
            'success': False,
            'message': '参数 min_group_size 必须为整数'
        }), 400
    if min_group_size < 1:
        return jsonify({
            'success': False,
            'message': '参数 min_group_size 不能小于 1'
        }), 400

    if isinstance(treat_zero_as_invalid, str):
        treat_zero_as_invalid = treat_zero_as_invalid.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
    else:
        treat_zero_as_invalid = bool(treat_zero_as_invalid)
    if isinstance(enable_data_cleaning, str):
        enable_data_cleaning = enable_data_cleaning.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
    else:
        enable_data_cleaning = bool(enable_data_cleaning)
    if isinstance(remove_zero_volume, str):
        remove_zero_volume = remove_zero_volume.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
    else:
        remove_zero_volume = bool(remove_zero_volume)
    if isinstance(enable_outlier_treatment, str):
        enable_outlier_treatment = enable_outlier_treatment.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
    else:
        enable_outlier_treatment = bool(enable_outlier_treatment)
    if isinstance(compute_fsc, str):
        compute_fsc = compute_fsc.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
    else:
        compute_fsc = bool(compute_fsc)
    if isinstance(compute_ic_decay_curve, str):
        compute_ic_decay_curve = compute_ic_decay_curve.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
    else:
        compute_ic_decay_curve = bool(compute_ic_decay_curve)

    try:
        liquidity_filter_ratio = float(liquidity_filter_ratio)
    except (TypeError, ValueError):
        return jsonify({'success': False, 'message': '参数 liquidity_filter_ratio 必须为数值'}), 400
    if not (0 <= liquidity_filter_ratio <= 1):
        return jsonify({'success': False, 'message': '参数 liquidity_filter_ratio 必须在 [0, 1] 区间'}), 400
    if outlier_method not in ('mad', 'winsor'):
        return jsonify({'success': False, 'message': '参数 outlier_method 仅支持 mad / winsor'}), 400
    try:
        outlier_group_minutes = int(outlier_group_minutes)
    except (TypeError, ValueError):
        return jsonify({'success': False, 'message': '参数 outlier_group_minutes 必须为整数'}), 400
    if outlier_group_minutes < 1:
        return jsonify({'success': False, 'message': '参数 outlier_group_minutes 不能小于 1'}), 400
    try:
        outlier_mad_n = float(outlier_mad_n)
        outlier_winsor_lower = float(outlier_winsor_lower)
        outlier_winsor_upper = float(outlier_winsor_upper)
    except (TypeError, ValueError):
        return jsonify({'success': False, 'message': '异常值参数必须为数值'}), 400
    if not (0 <= outlier_winsor_lower < outlier_winsor_upper <= 1):
        return jsonify({'success': False, 'message': 'Winsor 分位参数需满足 0<=lower<upper<=1'}), 400
    try:
        ic_decay_max_lag = int(ic_decay_max_lag)
    except (TypeError, ValueError):
        return jsonify({'success': False, 'message': '参数 ic_decay_max_lag 必须为整数'}), 400
    if ic_decay_max_lag < 1:
        return jsonify({'success': False, 'message': '参数 ic_decay_max_lag 不能小于 1'}), 400

    base_timeframe = str(base_timeframe or '1h').lower()
    factor_timeframe = str(factor_timeframe or base_timeframe).lower()
    if factor_bar_mode not in ('completed', 'intrabar', 'intrabar_strict'):
        return jsonify({
            'success': False,
            'message': '参数 factor_bar_mode 仅支持 completed / intrabar / intrabar_strict'
        }), 400

    def generate():
        _cs_cancel_event.clear()
        try:
            import os
            engine = get_global_engine()
            cs_evaluator = CrossSectionalEvaluator(
                n_groups=n_groups,
                normalize_method=normalize_method,
                predict_step=predict_step,
                sample_step=sample_step,
                base_timeframe=base_timeframe,
                factor_timeframe=factor_timeframe,
                factor_bar_mode=factor_bar_mode,
                max_lookback=max_lookback,
                min_coverage=min_coverage,
                min_valid_count=min_valid_count,
                min_group_size=min_group_size,
                treat_zero_as_invalid=treat_zero_as_invalid,
                enable_data_cleaning=enable_data_cleaning,
                remove_zero_volume=remove_zero_volume,
                liquidity_filter_ratio=liquidity_filter_ratio,
                enable_outlier_treatment=enable_outlier_treatment,
                outlier_method=outlier_method,
                outlier_group_minutes=outlier_group_minutes,
                outlier_mad_n=outlier_mad_n,
                outlier_winsor_lower=outlier_winsor_lower,
                outlier_winsor_upper=outlier_winsor_upper,
                compute_fsc=compute_fsc,
                compute_ic_decay_curve=compute_ic_decay_curve,
                ic_decay_max_lag=ic_decay_max_lag,
            )
            total = len(factor_ids)
            logger.info(f"开始截面评估，共 {total} 个因子")

            yield _sse_event('progress', {
                'phase': 'loading',
                'message': f'正在加载 {len(symbols)} 个币种的市场数据...',
                'completed': 0,
                'total': total
            })

            data_dict = {}
            load_t0 = time.time()
            total_symbols = len(symbols)
            
            def load_single_symbol(symbol):
                if _cs_cancel_event.is_set():
                    return (symbol, None)
                try:
                    md = load_local_market_data(
                        symbol, base_timeframe, start_date, end_date, exchange, trade_type
                    )
                    if md is not None and not md.empty:
                        return (symbol, md)
                except Exception as e:
                    logger.warning(f"加载 {symbol} 失败: {e}")
                return (symbol, None)
            
            load_workers = min(total_symbols, os.cpu_count() or 4, 8)
            loader = ThreadPoolExecutor(max_workers=load_workers)
            try:
                load_futures = {loader.submit(load_single_symbol, s): s for s in symbols}
                loaded_count = 0
                for future in as_completed(load_futures):
                    if _cs_cancel_event.is_set():
                        break
                    loaded_count += 1
                    symbol, md = future.result()
                    if md is not None:
                        data_dict[symbol] = md
                    if loaded_count % 5 == 0 or loaded_count == total_symbols:
                        yield _sse_event('progress', {
                            'phase': 'loading',
                            'message': f'数据加载中 {loaded_count}/{total_symbols}，有效币种 {len(data_dict)} 个',
                            'completed': 0,
                            'total': total
                        })
            except GeneratorExit:
                _cs_cancel_event.set()
                for f in load_futures:
                    f.cancel()
                loader.shutdown(wait=False, cancel_futures=True)
                raise
            except KeyboardInterrupt:
                _cs_cancel_event.set()
                for f in load_futures:
                    f.cancel()
                loader.shutdown(wait=False, cancel_futures=True)
                raise
            loader.shutdown(wait=True)

            load_elapsed = time.time() - load_t0
            logger.info(f"市场数据加载完成: {len(data_dict)}/{len(symbols)} 个币种, 耗时 {load_elapsed:.1f}s")

            if len(data_dict) < 2:
                logger.error(f"有效币种数量不足: {len(data_dict)} < 2")
                yield _sse_event('error', {
                    'message': f'有效币种数量不足（{len(data_dict)} < 2）'
                })
                yield _sse_event('done', {'results': []})
                return

            yield _sse_event('progress', {
                'phase': 'evaluating',
                'message': f'数据加载完成 ({len(data_dict)} 个币种, {load_elapsed:.1f}s)，开始评估 {total} 个因子...',
                'completed': 0,
                'total': total
            })

            def evaluate_one_factor(factor_id):
                """单个因子的截面评估（线程安全）"""
                if _cs_cancel_event.is_set():
                    return {
                        'factor_id': factor_id,
                        'success': False,
                        'message': '评估已取消',
                        'elapsed': 0
                    }
                t0 = time.time()
                try:
                    result = cs_evaluator.evaluate_cross_sectional(
                        data_dict, factor_id, engine, timeframe=base_timeframe
                    )
                    elapsed = time.time() - t0
                    if result.get('success'):
                        logger.debug(f"因子 {factor_id} 评估成功，耗时 {elapsed:.2f}s")
                        return {
                            'factor_id': factor_id,
                            'success': True,
                            'n_symbols': result.get('n_symbols'),
                            'n_periods': result.get('n_periods'),
                            'n_periods_total': result.get('n_periods_total'),
                            'n_periods_ic': result.get('n_periods_ic'),
                            'n_periods_returns': result.get('n_periods_returns'),
                            'ic': result.get('ic'),
                            'returns': result.get('returns'),
                            'coverage': result.get('coverage'),
                            'summary': result.get('summary'),
                            'elapsed': round(elapsed, 2)
                        }
                    else:
                        logger.warning(f"因子 {factor_id} 评估失败: {result.get('message')}")
                        return {
                            'factor_id': factor_id,
                            'success': False,
                            'message': result.get('message', '评估失败'),
                            'elapsed': round(elapsed, 2)
                        }
                except Exception as ex:
                    elapsed = time.time() - t0
                    logger.error(f"因子 {factor_id} 评估异常: {ex}", exc_info=True)
                    return {
                        'factor_id': factor_id,
                        'success': False,
                        'message': str(ex),
                        'elapsed': round(elapsed, 2)
                    }

            cpu_count = os.cpu_count() or 4
            if parallel_backend == 'process':
                max_workers = min(len(factor_ids), max(cpu_count - 1, 1), 4)
                executor_cls = ProcessPoolExecutor
            else:
                max_workers = min(len(factor_ids), cpu_count, 8)
                executor_cls = ThreadPoolExecutor
            completed_count = 0
            all_results = []
            eval_t0 = time.time()
            logger.info(f"开始并行评估，后端={parallel_backend}, workers={max_workers}")

            executor = executor_cls(max_workers=max_workers)
            pending = set()
            try:
                if executor_cls is ProcessPoolExecutor:
                    future_to_factor = {
                        executor.submit(
                            _evaluate_factor_process_worker,
                            fid,
                            data_dict,
                            n_groups,
                            normalize_method,
                            base_timeframe,
                            factor_timeframe,
                            factor_bar_mode,
                            max_lookback,
                            predict_step,
                            sample_step,
                            min_coverage,
                            min_valid_count,
                            min_group_size,
                            treat_zero_as_invalid,
                            enable_data_cleaning,
                            remove_zero_volume,
                            liquidity_filter_ratio,
                            enable_outlier_treatment,
                            outlier_method,
                            outlier_group_minutes,
                            outlier_mad_n,
                            outlier_winsor_lower,
                            outlier_winsor_upper,
                            compute_fsc,
                            compute_ic_decay_curve,
                            ic_decay_max_lag,
                        ): fid
                        for fid in factor_ids
                    }
                else:
                    future_to_factor = {
                        executor.submit(evaluate_one_factor, fid): fid
                        for fid in factor_ids
                    }
                pending = set(future_to_factor.keys())
                heartbeat_ts = time.time()

                while pending:
                    done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)

                    if _cs_cancel_event.is_set():
                        logger.info("检测到取消信号，停止评估循环...")
                        break

                    if not done:
                        now = time.time()
                        if now - heartbeat_ts >= max(heartbeat_interval_sec, 1.0):
                            elapsed = now - eval_t0
                            avg_sec = (elapsed / completed_count) if completed_count > 0 else 0
                            eta_sec = (avg_sec * (total - completed_count)) if completed_count > 0 else None
                            eta_text = f"{int(eta_sec)}s" if eta_sec is not None else "估算中"
                            yield _sse_event('progress', {
                                'phase': 'evaluating',
                                'completed': completed_count,
                                'total': total,
                                'message': f'评估运行中... 已完成 {completed_count}/{total}，待处理 {len(pending)}，预计剩余 {eta_text}'
                            })
                            heartbeat_ts = now
                        continue

                    for future in done:
                        factor_id = future_to_factor[future]
                        completed_count += 1
                        try:
                            result = future.result()
                        except Exception as ex:
                            logger.error(f"获取因子 {factor_id} 结果失败: {ex}")
                            result = {
                                'factor_id': factor_id,
                                'success': False,
                                'message': str(ex)
                            }

                        all_results.append(result)

                        result_clean = _sanitize_for_json(result)

                        yield _sse_event('factor_result', {
                            'result': result_clean,
                            'completed': completed_count,
                            'total': total,
                            'message': f'已完成 {completed_count}/{total}: {factor_id}'
                                + (f' ({result.get("elapsed", "?")}s)' if result.get('elapsed') else '')
                        })
            except GeneratorExit:
                logger.info("截面评估被中断（GeneratorExit），正在取消所有待处理任务...")
                _cs_cancel_event.set()
                for f in pending:
                    f.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            except KeyboardInterrupt:
                logger.info("截面评估被中断（KeyboardInterrupt），正在取消所有待处理任务...")
                _cs_cancel_event.set()
                for f in pending:
                    f.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=True)

            eval_elapsed = time.time() - eval_t0
            total_elapsed = time.time() - load_t0

            success_count = sum(1 for r in all_results if r.get('success'))
            fail_count = sum(1 for r in all_results if not r.get('success'))
            logger.info(f"截面评估完成: 成功 {success_count}/{total}, 失败 {fail_count}, 总耗时 {total_elapsed:.1f}s")

            yield _sse_event('done', {
                'total': total,
                'success_count': success_count,
                'fail_count': fail_count,
                'eval_elapsed': round(eval_elapsed, 2),
                'total_elapsed': round(total_elapsed, 2),
                'workers': max_workers,
                'parallel_backend': parallel_backend
            })
        except Exception as e:
            logger.error(f"截面评估过程发生异常: {e}", exc_info=True)
            yield _sse_event('error', {
                'message': f'评估过程发生错误: {str(e)}'
            })
            yield _sse_event('done', {
                'total': len(factor_ids),
                'success_count': len([r for r in all_results if r.get('success')]) if 'all_results' in locals() else 0,
                'fail_count': len([r for r in all_results if not r.get('success')]) if 'all_results' in locals() else len(factor_ids),
                'error': str(e)
            })

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',  # 禁用 nginx 缓冲
            'Connection': 'keep-alive',
        }
    )


@bp.route('/cancel_evaluation', methods=['POST'])
def cancel_evaluation():
    """取消正在进行的截面评估"""
    _cs_cancel_event.set()
    return jsonify({'success': True, 'message': '取消信号已发送'})


def _sse_event(event_type: str, data: dict) -> str:
    """构造 SSE 事件字符串"""
    payload = json.dumps(_sanitize_for_json({'type': event_type, **data}), ensure_ascii=False)
    return f"event: message\ndata: {payload}\n\n"


def _sanitize_for_json(obj):
    """递归清理 dict/list 中的 NaN/Inf 值（JSON 不支持）"""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj



def parse_alpha101_filename(filename):
    """解析Alpha101文件名"""
    # 格式: alpha101_results_SYMBOL_TIMEFRAME.pkl
    parts = filename.replace('.pkl', '').split('_')
    if len(parts) >= 4:
        symbol = parts[2]
        timeframe = parts[3]
        return symbol, timeframe
    return 'Unknown', 'Unknown'

def clean_factor_name(factor_name, factor_type=''):
    """清理因子名称，移除不合理的币种后缀"""
    # 需要移除的币种后缀列表
    crypto_symbols = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'DOGE', 'LINK', 'LPT', 'MOVR', 'PEOPLE', 'SUI', 'FIL']
    
    # 移除因子名称末尾的币种后缀
    for symbol in crypto_symbols:
        # 移除 "_SYMBOL" 格式的后缀
        if factor_name.endswith(f'_{symbol}'):
            factor_name = factor_name[:-len(symbol)-1]
        # 移除 "_SYMBOL_USDT" 格式的后缀  
        if factor_name.endswith(f'_{symbol}_USDT'):
            factor_name = factor_name[:-len(symbol)-6]
        # 移除 "_SYMBOL_timeframe" 格式的后缀
        for tf in ['1h', '4h', '1m', '5m', '15m', '1d']:
            if factor_name.endswith(f'_{symbol}_{tf}'):
                factor_name = factor_name[:-len(symbol)-len(tf)-2]
    
    return factor_name


def parse_factor_id(factor_id):
    """解析因子ID"""
    parts = factor_id.split('_', 2)
    if len(parts) >= 2:
        return {
            'type': parts[0],
            'subtype': parts[1] if len(parts) > 2 else None,
            'identifier': parts[2] if len(parts) > 2 else parts[1]
        }
    return None

def calculate_factor_values(factor_info, market_data):
    """计算因子值"""
    # V3已不再使用该函数，保留占位
    return None

def save_evaluation_results(factor_id, results, metadata):
    """转调核心层的评估结果保存"""
    try:
        core_save_evaluation_results(factor_id, results, metadata)
    except Exception as e:
        print(f"保存评估结果失败: {e}")

def get_factor_detail_info(factor_info):
    """获取因子详细信息"""
    # 这里实现获取因子详细信息的逻辑
    return {
        'id': factor_info.get('identifier'),
        'type': factor_info.get('type'),
        'description': '因子详细信息',
        'formula': '因子计算公式',
        'parameters': {},
        'evaluation_history': []
    }

def export_factor_data(factor_info):
    """导出因子数据"""
    export_dir = FACTOR_LIBRARY_DIR / "minactors"  # 导出到minactors目录
    export_dir.mkdir(parents=True, exist_ok=True)
    
    identifier = factor_info.get('identifier') or factor_info.get('id') or 'factor'
    export_file = export_dir / f"{identifier}_export.csv"
    
    # 创建示例导出文件
    pd.DataFrame({
        'factor_name': [factor_info['identifier']],
        'type': [factor_info['type']],
        'exported_at': [datetime.now().isoformat()]
    }).to_csv(export_file, index=False)
    
    return str(export_file)

def load_local_market_data(symbol, timeframe, start_date, end_date, exchange='binance', trade_type='futures'):
    """从本地加载市场数据"""
    try:
        # 构建文件路径
        data_dir = Path(__file__).parent.parent.parent / "data" / exchange / trade_type
        
        # 解析交易对格式
        # API返回的交易对格式是 BTC_USDT，但实际文件名需要 BTC_USDT_USDT
        if '_' in symbol:
            parts = symbol.split('_')
            if len(parts) >= 2:
                # 如果输入是 BTC_USDT，我们需要构建 BTC_USDT_USDT
                base_symbol = parts[0]  # BTC
                filename = f"{base_symbol}_USDT_USDT-{timeframe}-{trade_type}.feather"
            else:
                base_symbol = symbol
                filename = f"{base_symbol}_USDT_USDT-{timeframe}-{trade_type}.feather"
        else:
            base_symbol = symbol
            filename = f"{base_symbol}_USDT_USDT-{timeframe}-{trade_type}.feather"
            
        file_path = data_dir / filename
        
        if not file_path.exists():
            print(f"数据文件不存在: {file_path}")
            return None
        
        # 读取feather文件
        data = pd.read_feather(file_path)
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            print(f"数据文件缺少必要的列: {required_columns}")
            return None
        
        # 统一的时间列解析（自动识别秒/毫秒/字符串）
        def parse_time_series(series):
            try:
                if pd.api.types.is_datetime64_any_dtype(series):
                    return series
                if pd.api.types.is_numeric_dtype(series):
                    s = series.dropna()
                    unit = 's'
                    if len(s) > 0:
                        sample = s.iloc[0]
                        unit = 'ms' if sample > 10_000_000_000 else 's'
                    return pd.to_datetime(series, errors='coerce', unit=unit, utc=True)
                return pd.to_datetime(series, errors='coerce', utc=True)
            except Exception:
                return pd.to_datetime(series, errors='coerce', utc=True)
        
        # 如果有时间列，设置为索引
        time_col = None
        for cand in ['timestamp', 'datetime', 'time', 'date']:
            if cand in data.columns:
                time_col = cand
                break
        
        if time_col is not None:
            data[time_col] = parse_time_series(data[time_col])
            data.set_index(time_col, inplace=True)
        else:
            if not isinstance(data.index, pd.DatetimeIndex):
                if len(data) > 1000:
                    return None
        
        if start_date and end_date:
            try:
                start_dt = pd.to_datetime(start_date, utc=True)
                end_dt = pd.to_datetime(end_date, utc=True)
                
                if not isinstance(data.index, pd.DatetimeIndex):
                    data.index = pd.to_datetime(data.index, utc=True)
                
                try:
                    if data.index.tz is None:
                        data.index = data.index.tz_localize('UTC')
                    else:
                        data.index = data.index.tz_convert('UTC')
                except Exception:
                    data.index = data.index.tz_localize(None)
                
                original_count = len(data)
                data = data[(data.index >= start_dt) & (data.index <= end_dt)]
                
                if len(data) == 0:
                    return None
                    
            except Exception:
                return None
        
        # 确保数据按时间排序
        data.sort_index(inplace=True)
        
        if len(data) == 0:
            return None
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            return None
        
        if not isinstance(data.index, pd.DatetimeIndex):
            return None
        
        if (data.index.max() - data.index.min()).total_seconds() < 60:
            return None
        
        print(f"✅ {symbol} [{timeframe}]: {len(data)} 条 ({data.index.min().date()} ~ {data.index.max().date()})")
        return data
        
    except Exception as e:
        print(f"加载本地数据失败: {e}")
        return None 


@bp.route('/ensemble_backtest', methods=['POST'])
def ensemble_backtest():
    """
    因子组合回测API
    - 支持从截面评估结果中选择因子
    - 支持导入CSV文件
    - 自动处理反向因子（IC为负时取负）
    - 支持多种组合方法
    """
    try:
        import numpy as np
        from factor_miner.core.factor_optimizer import FactorOptimizer
        
        payload = request.get_json() or {}
        
        factor_ids = payload.get('factor_ids', [])
        factor_ic_dict = payload.get('factor_ic_dict', {})
        auto_reverse = payload.get('auto_reverse', True)
        ic_source = payload.get('ic_source', 'evaluation')
        ensemble_method = payload.get('ensemble_method', 'ic_weight')
        symbols = payload.get('symbols', [])
        timeframe = payload.get('timeframe', '1h')
        start_date = payload.get('start_date')
        end_date = payload.get('end_date')
        exchange = payload.get('exchange', 'binance')
        trade_type = payload.get('trade_type', 'futures')
        n_groups = payload.get('n_groups', 5)
        transaction_cost = payload.get('transaction_cost', 0.001)
        predict_step = payload.get('predict_step', 1)
        sample_step = payload.get('sample_step', 1)
        factor_timeframe = payload.get('factor_timeframe', timeframe)
        factor_bar_mode = payload.get('factor_bar_mode', 'completed')
        max_lookback = payload.get('max_lookback', 200)
        min_coverage = payload.get('min_coverage', 0.3)
        min_valid_count = payload.get('min_valid_count', 30)
        min_group_size = payload.get('min_group_size', 5)
        treat_zero_as_invalid = payload.get('treat_zero_as_invalid', True)

        try:
            n_groups = int(n_groups)
        except (TypeError, ValueError):
            return jsonify({
                'success': False,
                'message': '参数 n_groups 必须为整数'
            }), 400
        if n_groups < 2:
            return jsonify({
                'success': False,
                'message': '参数 n_groups 不能小于 2'
            }), 400

        try:
            transaction_cost = float(transaction_cost)
        except (TypeError, ValueError):
            return jsonify({
                'success': False,
                'message': '参数 transaction_cost 必须为数字'
            }), 400
        if transaction_cost < 0:
            return jsonify({
                'success': False,
                'message': '参数 transaction_cost 不能小于 0'
            }), 400

        try:
            predict_step = int(predict_step)
        except (TypeError, ValueError):
            return jsonify({
                'success': False,
                'message': '参数 predict_step 必须为整数'
            }), 400
        if predict_step < 1:
            return jsonify({
                'success': False,
                'message': '参数 predict_step 不能小于 1'
            }), 400

        try:
            sample_step = int(sample_step)
        except (TypeError, ValueError):
            return jsonify({
                'success': False,
                'message': '参数 sample_step 必须为整数'
            }), 400
        if sample_step < 1:
            return jsonify({
                'success': False,
                'message': '参数 sample_step 不能小于 1'
            }), 400
        
        if not factor_ids or not symbols:
            return jsonify({
                'success': False,
                'message': '请选择因子和币种'
            })
        if len(symbols) < 2:
            return jsonify({
                'success': False,
                'message': '截面组合回测至少需要 2 个币种'
            }), 400
        
        if not start_date or not end_date:
            return jsonify({
                'success': False,
                'message': '请选择时间范围'
            })
        
        engine = get_global_engine()
        data_dict = {}
        
        for symbol in symbols:
            try:
                md = load_local_market_data(symbol, timeframe, start_date, end_date, exchange, trade_type)
                if md is not None and not md.empty:
                    data_dict[symbol] = md
            except Exception as e:
                print(f"加载 {symbol} 失败: {e}")
        
        if len(data_dict) < 2:
            return jsonify({
                'success': False,
                'message': f'加载到有效市场数据的币种仅 {len(data_dict)} 个，截面组合回测至少需要 2 个'
            })
        
        prep_evaluator = CrossSectionalEvaluator(
            n_groups=n_groups,
            normalize_method='rank',
            predict_step=predict_step,
            sample_step=1,
            base_timeframe=timeframe,
            factor_timeframe=factor_timeframe,
            factor_bar_mode=factor_bar_mode,
            max_lookback=max_lookback,
            min_coverage=min_coverage,
            min_valid_count=min_valid_count,
            min_group_size=min_group_size,
            treat_zero_as_invalid=treat_zero_as_invalid,
        )

        all_factors = {}
        for factor_id in factor_ids:
            try:
                cs_df = prep_evaluator.prepare_cross_sectional_data(data_dict, factor_id, engine)
                if cs_df is None or cs_df.empty:
                    continue
                for symbol in cs_df['symbol'].unique():
                    sym_mask = cs_df['symbol'] == symbol
                    sym_data = cs_df.loc[sym_mask].sort_values('date')
                    fv = sym_data.set_index('date')['factor_value']
                    if symbol not in all_factors:
                        all_factors[symbol] = {}
                    all_factors[symbol][factor_id] = fv
            except Exception as e:
                print(f"计算因子 {factor_id} 截面数据失败: {e}")

        all_factors_dfs = {}
        for symbol, factor_dict in all_factors.items():
            if factor_dict:
                all_factors_dfs[symbol] = pd.DataFrame(factor_dict)

        if not all_factors_dfs:
            return jsonify({
                'success': False,
                'message': '无法计算因子值'
            })

        for symbol, factors_df in all_factors_dfs.items():
            for fid in factors_df.columns:
                if auto_reverse:
                    ic_value = factor_ic_dict.get(fid, 0)
                    if ic_value < 0:
                        factors_df[fid] = -factors_df[fid]
        
        ensemble_factors = {}
        reversed_factors = []

        if ic_source == 'backtest' and ensemble_method == 'ic_weight':
            recalc_ic_dict = {}
            for fid in factor_ids:
                all_fv = []
                all_rt = []
                for sym, fdf in all_factors_dfs.items():
                    if fid in fdf.columns:
                        md = data_dict[sym].copy().sort_index()
                        md['future_returns'] = md['close'].pct_change(periods=predict_step).shift(-predict_step)
                        common_idx = fdf[fid].index.intersection(md.index)
                        if len(common_idx) > 0:
                            fv = fdf.loc[common_idx, fid].shift(1)
                            rt = md.loc[common_idx, 'future_returns']
                            mask = fv.notna() & rt.notna()
                            if mask.sum() > 10:
                                all_fv.extend(fv[mask].values.tolist())
                                all_rt.extend(rt[mask].values.tolist())
                if len(all_fv) >= 20:
                    try:
                        ic_val = float(pd.Series(all_fv).corr(pd.Series(all_rt)))
                        if np.isfinite(ic_val):
                            recalc_ic_dict[fid] = ic_val
                        else:
                            recalc_ic_dict[fid] = 0.0
                    except Exception:
                        recalc_ic_dict[fid] = 0.0
                else:
                    recalc_ic_dict[fid] = 0.0
            factor_ic_dict = recalc_ic_dict

        ic_weight_map = {}
        abs_ic_sum = 0.0
        for fid in factor_ids:
            try:
                ic_val = float(factor_ic_dict.get(fid, 0.0))
            except (TypeError, ValueError):
                ic_val = 0.0
            if auto_reverse:
                weight_ic = abs(ic_val)
            else:
                weight_ic = ic_val
            ic_weight_map[fid] = weight_ic
            abs_ic_sum += abs(weight_ic)
        if abs_ic_sum > 0:
            ic_weight_map = {k: v / abs_ic_sum for k, v in ic_weight_map.items()}
        else:
            eq_w = 1.0 / max(len(factor_ids), 1)
            ic_weight_map = {fid: eq_w for fid in factor_ids}

        for symbol, factors_df in all_factors_dfs.items():
            try:
                market_data = data_dict[symbol]
                if ensemble_method == 'equal_weight':
                    ensemble_factor = factors_df.mean(axis=1)
                elif ensemble_method == 'ic_weight':
                    active_cols = [c for c in factors_df.columns if c in ic_weight_map]
                    if not active_cols:
                        ensemble_factor = factors_df.mean(axis=1)
                    else:
                        weights = np.array([ic_weight_map[c] for c in active_cols], dtype=float)
                        weight_sum = float(weights.sum())
                        if weight_sum <= 0:
                            ensemble_factor = factors_df[active_cols].mean(axis=1)
                        else:
                            weights = weights / weight_sum
                            ensemble_factor = factors_df[active_cols].mul(weights, axis=1).sum(axis=1)
                elif ensemble_method == 'ml_weight':
                    md = market_data.copy().sort_index()
                    ml_returns = md['close'].pct_change(periods=predict_step).shift(-predict_step)
                    optimizer = FactorOptimizer()
                    optimizer.set_data(None, ml_returns)
                    ensemble_factor = optimizer._create_ml_weighted_factor_walk_forward(factors_df)
                elif ensemble_method in ('max_icir_weight', 'max_icir'):
                    md = market_data.copy().sort_index()
                    icir_returns = md['close'].pct_change(periods=predict_step).shift(-predict_step)
                    optimizer = FactorOptimizer()
                    optimizer.set_data(None, icir_returns)
                    ensemble_factor = optimizer.create_ensemble_factor(
                        factors_df,
                        method='max_icir_weight'
                    )
                else:
                    ensemble_factor = factors_df.mean(axis=1)
                
                ensemble_factors[symbol] = ensemble_factor
            except Exception as e:
                print(f"创建组合因子 {symbol} 失败: {e}")
        
        if not ensemble_factors:
            return jsonify({
                'success': False,
                'message': '无法创建组合因子'
            })

        effective_n_groups = min(max(n_groups, 2), len(ensemble_factors))

        cs_rows = []
        for symbol, ensemble_factor in ensemble_factors.items():
            market_data = data_dict.get(symbol)
            if market_data is None or market_data.empty:
                continue
            try:
                md = market_data.copy().sort_index()
                md['future_returns'] = (
                    md['close'].pct_change(periods=predict_step).shift(-predict_step)
                )
                common_idx = ensemble_factor.index.intersection(md.index)
                if len(common_idx) == 0:
                    continue

                fv = ensemble_factor.loc[common_idx]
                rt = md.loc[common_idx, 'future_returns']
                mask = fv.notna() & rt.notna()
                fv = fv[mask]
                rt = rt[mask]
                if len(fv) == 0:
                    continue

                sym_df = pd.DataFrame({
                    'date': fv.index,
                    'symbol': symbol,
                    'factor_value': fv.values,
                    'returns': rt.values
                })
                cs_rows.append(sym_df)
            except Exception as e:
                print(f"构建截面回测数据失败 {symbol}: {e}")

        if not cs_rows:
            return jsonify({
                'success': False,
                'message': '截面回测数据构建失败（请检查时间范围和币种数据完整性）'
            }), 400

        cs_data = pd.concat(cs_rows, ignore_index=True)
        cs_data['date'] = pd.to_datetime(cs_data['date'])
        if sample_step > 1 and not cs_data.empty:
            sampled_dates = pd.DatetimeIndex(sorted(cs_data['date'].drop_duplicates())).to_series().iloc[::sample_step]
            cs_data = cs_data[cs_data['date'].isin(sampled_dates.values)]

        cs_evaluator = CrossSectionalEvaluator(
            n_groups=effective_n_groups,
            normalize_method='rank',
            predict_step=predict_step,
            sample_step=sample_step,
            min_coverage=min_coverage,
            min_valid_count=min_valid_count,
            min_group_size=min_group_size,
            treat_zero_as_invalid=treat_zero_as_invalid,
        )
        ic_results = cs_evaluator.calculate_cross_sectional_ic(cs_data)
        returns_results = cs_evaluator.calculate_cross_sectional_returns(cs_data, timeframe=timeframe, transaction_cost=transaction_cost)

        performance_summary = {
            'CROSS_SECTIONAL_LS': {
                'total_return': returns_results.get('total_return'),
                'total_return_after_cost': returns_results.get('total_return_after_cost'),
                'sharpe_ratio': returns_results.get('sharpe_ratio'),
                'sharpe_ratio_after_cost': returns_results.get('sharpe_ratio_after_cost'),
                'max_drawdown': returns_results.get('max_drawdown'),
                'max_drawdown_after_cost': returns_results.get('max_drawdown_after_cost'),
                'win_rate': returns_results.get('win_rate'),
                'win_rate_after_cost': returns_results.get('win_rate_after_cost'),
            }
        }

        if auto_reverse:
            reversed_factors = [fid for fid in factor_ids if float(factor_ic_dict.get(fid, 0) or 0) < 0]
        
        def _clean_nan(obj):
            import math
            if isinstance(obj, dict):
                return {k: _clean_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_clean_nan(v) for v in obj]
            elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            else:
                return obj
        
        return jsonify({
            'success': True,
            'ensemble_method': ensemble_method,
            'ic_source': ic_source,
            'n_factors': len(factor_ids),
            'n_symbols': len(data_dict),
            'performance_summary': _clean_nan(performance_summary),
            'cross_sectional_ic': _clean_nan(ic_results),
            'cross_sectional_performance': _clean_nan(returns_results),
            'factor_ids': factor_ids,
            'reversed_factors': reversed_factors,
            'predict_step': predict_step,
            'sample_step': sample_step,
            'n_groups_effective': effective_n_groups
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'组合回测失败: {str(e)}'
        })

@bp.route('/factor_correlation', methods=['POST'])
def factor_correlation():
    """
    计算选中因子的截面相关性矩阵
    - 对每个币种分别计算因子值，然后合并为截面数据
    - 计算因子间的 Pearson 和 Spearman 相关系数
    - 返回相关性矩阵和高相关因子对
    """
    try:
        import numpy as np

        payload = request.get_json() or {}
        factor_ids = payload.get('factor_ids', [])
        factor_ic_dict = payload.get('factor_ic_dict', {})
        auto_reverse = payload.get('auto_reverse', True)
        symbols = payload.get('symbols', [])
        timeframe = payload.get('timeframe', '1h')
        start_date = payload.get('start_date')
        end_date = payload.get('end_date')
        exchange = payload.get('exchange', 'binance')
        trade_type = payload.get('trade_type', 'futures')
        corr_threshold = payload.get('corr_threshold', 0.7)

        if not factor_ids or len(factor_ids) < 2:
            return jsonify({
                'success': False,
                'message': '至少需要选择2个因子才能计算相关性'
            }), 400

        if not symbols or len(symbols) < 2:
            return jsonify({
                'success': False,
                'message': '至少需要2个币种来计算截面相关性'
            }), 400

        if not start_date or not end_date:
            return jsonify({
                'success': False,
                'message': '请选择时间范围'
            })

        engine = get_global_engine()
        data_dict = {}
        for symbol in symbols:
            try:
                md = load_local_market_data(symbol, timeframe, start_date, end_date, exchange, trade_type)
                if md is not None and not md.empty:
                    data_dict[symbol] = md
            except Exception:
                continue

        if len(data_dict) < 2:
            return jsonify({
                'success': False,
                'message': '成功加载市场数据的币种不足2个'
            })

        all_factor_data = {}
        for symbol, market_data in data_dict.items():
            symbol_factors = {}
            for factor_id in factor_ids:
                try:
                    factor_values = engine.compute_single_factor(factor_id, market_data)
                    if factor_values is not None:
                        if auto_reverse:
                            ic_value = factor_ic_dict.get(factor_id, 0)
                            if ic_value < 0:
                                factor_values = -factor_values
                        symbol_factors[factor_id] = factor_values
                except Exception:
                    continue
            if symbol_factors:
                all_factor_data[symbol] = symbol_factors

        if len(all_factor_data) < 2:
            return jsonify({
                'success': False,
                'message': '成功计算因子值的币种不足2个'
            })

        combined_rows = []
        for symbol, factors in all_factor_data.items():
            try:
                max_len = max(len(v) for v in factors.values())
                ref_key = max(factors.keys(), key=lambda k: len(factors[k]))
                ref_index = factors[ref_key].index
                row_data = {'symbol': symbol}
                for fid, fv in factors.items():
                    aligned = fv.reindex(ref_index)
                    row_data[fid] = aligned
                df = pd.DataFrame(row_data, index=ref_index)
                combined_rows.append(df)
            except Exception:
                continue

        if not combined_rows:
            return jsonify({
                'success': False,
                'message': '无法构建因子数据'
            })

        combined = pd.concat(combined_rows, axis=0)
        factor_cols = [fid for fid in factor_ids if fid in combined.columns]
        if len(factor_cols) < 2:
            return jsonify({
                'success': False,
                'message': '有效因子不足2个'
            })

        factor_matrix = combined[factor_cols].dropna()

        if len(factor_matrix) < 10:
            return jsonify({
                'success': False,
                'message': f'有效数据点不足（{len(factor_matrix)} < 10）'
            })

        pearson_corr = factor_matrix.corr(method='pearson')
        spearman_corr = factor_matrix.corr(method='spearman')

        pearson_matrix = []
        spearman_matrix = []
        for i, fid_i in enumerate(factor_cols):
            pearson_row = []
            spearman_row = []
            for j, fid_j in enumerate(factor_cols):
                p_val = pearson_corr.iloc[i, j]
                s_val = spearman_corr.iloc[i, j]
                pearson_row.append(float(p_val) if np.isfinite(p_val) else None)
                spearman_row.append(float(s_val) if np.isfinite(s_val) else None)
            pearson_matrix.append(pearson_row)
            spearman_matrix.append(spearman_row)

        high_corr_pairs = []
        for i in range(len(factor_cols)):
            for j in range(i + 1, len(factor_cols)):
                p_val = pearson_matrix[i][j]
                s_val = spearman_matrix[i][j]
                if p_val is not None and abs(p_val) >= corr_threshold:
                    high_corr_pairs.append({
                        'factor_1': factor_cols[i],
                        'factor_2': factor_cols[j],
                        'pearson': p_val,
                        'spearman': s_val,
                    })

        high_corr_pairs.sort(key=lambda x: abs(x['pearson']), reverse=True)

        def _clean_nan(obj):
            import math
            if isinstance(obj, dict):
                return {k: _clean_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_clean_nan(v) for v in obj]
            elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            else:
                return obj

        return jsonify(_clean_nan({
            'success': True,
            'factor_ids': factor_cols,
            'n_symbols': len(all_factor_data),
            'n_data_points': len(factor_matrix),
            'pearson_matrix': pearson_matrix,
            'spearman_matrix': spearman_matrix,
            'high_corr_pairs': high_corr_pairs,
            'corr_threshold': corr_threshold,
        }))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'相关性计算失败: {str(e)}'
        })


# 新增：因子计算API
@bp.route('/calculate', methods=['POST'])
def calculate_factor():
    """计算因子值"""
    try:
        data = request.get_json()
        factor_id = data.get('factor_id')
        market_data = data.get('data', {})
        parameters = data.get('parameters', {})
        
        print(f"🔍 收到因子计算请求: {factor_id}")
        print(f"🔍 市场数据长度: {len(market_data.get('close', []))}")
        print(f"🔍 参数: {parameters}")
        
        if not factor_id:
            return jsonify({'success': False, 'error': '缺少factor_id参数'})
        
        # 查找因子定义（先在minactors中查找，再在technicals中查找）
        factor_file = FACTOR_LIBRARY_DIR / "minactors" / "definitions" / f"{factor_id}.json"
        if not factor_file.exists():
            factor_file = FACTOR_LIBRARY_DIR / "technicals" / "definitions" / f"{factor_id}.json"
            if not factor_file.exists():
                print(f"❌ 因子定义文件不存在: {factor_id}")
                return jsonify({'success': False, 'error': f'因子 {factor_id} 不存在'})
        
        with open(factor_file, 'r', encoding='utf-8') as f:
            factor_info = json.load(f)
        
        print(f"🔍 因子信息: {factor_info.get('name', factor_id)}")
        
        # 检查因子类型
        computation_type = factor_info.get('computation_type')
        
        if computation_type == 'formula':
            # 公式因子
            factor_values = calculate_formula_factor(factor_info, market_data, parameters)
        elif computation_type == 'ml':
            # ML因子
            factor_values = calculate_ml_factor(factor_info, market_data, parameters)
        else:
            # 默认使用函数计算
            factor_values = calculate_function_factor(factor_info, market_data, parameters)
        
        if factor_values is not None:
            print(f"✅ 因子计算成功，返回 {len(factor_values)} 个值")
            return jsonify({
                'success': True,
                'factor_values': factor_values,
                'factor_name': factor_info.get('name', factor_id)
            })
        else:
            print(f"❌ 因子计算失败")
            return jsonify({'success': False, 'error': '因子计算失败'})
            
    except Exception as e:
        print(f"❌ 因子计算API异常: {e}")
        return jsonify({'success': False, 'error': str(e)})

def calculate_formula_factor(factor_info, market_data, parameters):
    """计算公式因子"""
    try:
        print(f"🔍 计算公式因子: {factor_info.get('name')}")
        # 这里可以实现公式解析和计算
        # 暂时返回简单的移动平均线作为示例
        close_prices = market_data.get('close', [])
        if not close_prices:
            print("❌ 没有收盘价数据")
            return None
        
        period = parameters.get('period', 20)
        if len(close_prices) < period:
            print(f"❌ 数据长度 {len(close_prices)} 小于周期 {period}")
            return None
        
        # 计算简单移动平均线
        factor_values = []
        for i in range(len(close_prices)):
            if i < period - 1:
                factor_values.append(None)
            else:
                window = close_prices[i-period+1:i+1]
                avg = sum(window) / len(window)
                factor_values.append(avg)
        
        print(f"✅ 公式因子计算完成，返回 {len(factor_values)} 个值")
        return factor_values
    except Exception as e:
        print(f"❌ 公式因子计算失败: {e}")
        return None

def calculate_ml_factor(factor_info, market_data, parameters):
    """计算ML因子"""
    try:
        print(f"🔍 计算ML因子: {factor_info.get('name')}")
        # 这里可以实现ML模型预测
        # 暂时返回随机值作为示例
        close_prices = market_data.get('close', [])
        if not close_prices:
            print("❌ 没有收盘价数据")
            return None
        
        import random
        factor_values = [random.uniform(-1, 1) for _ in range(len(close_prices))]
        print(f"✅ ML因子计算完成，返回 {len(factor_values)} 个值")
        return factor_values
    except Exception as e:
        print(f"❌ ML因子计算失败: {e}")
        return None

def calculate_function_factor(factor_info, market_data, parameters):
    """计算函数因子"""
    try:
        print(f"🔍 计算函数因子: {factor_info.get('name')}")
        # 尝试导入并调用因子函数
        factor_name = factor_info.get('factor_id', '')
        if not factor_name:
            print("❌ 因子ID为空")
            return None
        
        # 构建函数文件路径（先在technicals中查找，再在minactors中查找）
        function_file = FACTOR_LIBRARY_DIR / "technicals" / "functions" / f"{factor_name}.py"
        if not function_file.exists():
            function_file = FACTOR_LIBRARY_DIR / "minactors" / "functions" / f"{factor_name}.py"
            if not function_file.exists():
                print(f"❌ 因子函数文件不存在: {factor_name}")
                return None
        
        print(f"🔍 找到因子函数文件: {function_file}")
        
        # 动态导入因子函数
        import importlib.util
        spec = importlib.util.spec_from_file_location(factor_name, function_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 准备数据
        df_data = pd.DataFrame({
            'open': market_data.get('open', []),
            'high': market_data.get('high', []),
            'low': market_data.get('low', []),
            'close': market_data.get('close', []),
            'volume': market_data.get('volume', [])
        })
        
        print(f"🔍 准备数据DataFrame: {df_data.shape}")
        
        # 调用calculate函数
        if hasattr(module, 'calculate'):
            print(f"🔍 调用因子函数: {factor_name}.calculate()")
            factor_values = module.calculate(df_data, **parameters)
            
            if isinstance(factor_values, pd.Series):
                result = factor_values.tolist()
            elif isinstance(factor_values, (list, tuple)):
                result = list(factor_values)
            else:
                print(f"❌ 因子函数返回类型不支持: {type(factor_values)}")
                return None
            
            print(f"✅ 函数因子计算完成，返回 {len(result)} 个值")
            return result
        else:
            print(f"❌ 因子函数 {factor_name} 没有calculate函数")
            return None
            
    except Exception as e:
        print(f"❌ 函数因子计算失败: {e}")
        import traceback
        traceback.print_exc()
        return None 
