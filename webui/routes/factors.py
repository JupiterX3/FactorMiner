"""
因子库相关路由（V3架构）
使用透明因子存储与统一引擎
"""

import json
import re
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from flask import Blueprint, request, jsonify
import math
import threading
import time
import uuid
from factor_miner.core.factor_evaluator import FactorEvaluator, FactorStatistics, CrossSectionalEvaluator
from factor_miner.core.factor_catalog import FactorCatalogService
from factor_miner.core.factor_repository import FactorRepository
from factor_miner.core.factor_engine import get_global_engine
from factor_miner.core.data_loader import DataLoader
from factor_miner.core.evaluation_io import (
    save_evaluation_results as core_save_evaluation_results,
)

# 每个评估/组合/相关性请求使用独立的 threading.Event，
# 通过 request_id 精确取消，避免多用户互相影响。
# 若 request 未提供 request_id，回退到 _LEGACY_CANCEL_ID 的全局事件（向后兼容）。
_LEGACY_CANCEL_ID = '__legacy_global__'
_cs_cancel_events: dict = {}
_cs_cancel_lock = threading.Lock()
_cs_active_executors: dict = {}
_cs_executor_lock = threading.Lock()
_shutdown_event = threading.Event()


def _get_cancel_event(request_id: str) -> threading.Event:
    """获取（或创建）指定 request_id 对应的取消事件。"""
    rid = str(request_id) if request_id else _LEGACY_CANCEL_ID
    with _cs_cancel_lock:
        ev = _cs_cancel_events.get(rid)
        if ev is None:
            ev = threading.Event()
            _cs_cancel_events[rid] = ev
        return ev


def _discard_cancel_event(request_id: str) -> None:
    """请求完成后清理 Event，避免内存累积。"""
    rid = str(request_id) if request_id else _LEGACY_CANCEL_ID
    with _cs_cancel_lock:
        _cs_cancel_events.pop(rid, None)


def _set_cancel(request_id: str) -> int:
    """设置指定 request_id（或全部）为取消，返回被触发的 event 数量。"""
    if request_id:
        ev = _get_cancel_event(request_id)
        ev.set()
        return 1
    with _cs_cancel_lock:
        for ev in _cs_cancel_events.values():
            ev.set()
        return len(_cs_cancel_events)


def _register_active_executor(request_id: str, executor) -> None:
    rid = str(request_id) if request_id else _LEGACY_CANCEL_ID
    with _cs_executor_lock:
        _cs_active_executors[rid] = executor


def _pop_active_executor(request_id: str):
    rid = str(request_id) if request_id else _LEGACY_CANCEL_ID
    with _cs_executor_lock:
        return _cs_active_executors.pop(rid, None)


def _pop_all_active_executors() -> list:
    with _cs_executor_lock:
        items = list(_cs_active_executors.items())
        _cs_active_executors.clear()
        return items


def _force_stop_executor(executor) -> None:
    if executor is None:
        return
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        try:
            executor.shutdown(wait=False)
        except Exception:
            pass
    try:
        procs = getattr(executor, '_processes', None) or {}
        for p in procs.values():
            try:
                if p is not None and p.is_alive():
                    p.terminate()
            except Exception:
                continue
        for p in procs.values():
            try:
                if p is not None:
                    p.join(timeout=1)
            except Exception:
                continue
    except Exception:
        pass


def trigger_global_shutdown():
    """外部信号（如 SIGINT）调用：取消所有评估并停止所有执行器"""
    _shutdown_event.set()
    _set_cancel('')
    for _, ex in _pop_all_active_executors():
        _force_stop_executor(ex)


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
    """获取因子列表（v4：按一级分类目录动态扫描）

    一级分类（category / data_requirement）对应物理目录：
    - basic_kline  : 仅需 OHLCV
    - derivatives  : 需要 OI / LSR / taker_buy / basis 等衍生品微观结构
    - funding      : 需要资金费率历史

    二级分类（subcategory，写进 JSON）：
    - technical / mined / event / open_interest / long_short_ratio /
      taker_volume / basis / funding_carry / funding_arbitrage 等

    事件因子（event）仅 0/1 取值，不建议用于截面评估；前端会据此打标。
    """
    try:
        now_ts = time.time()
        with _factor_list_cache_lock:
            if _factor_list_cache["payload"] is not None and now_ts < _factor_list_cache["expires_at"]:
                return jsonify(_factor_list_cache["payload"])

        # 第 2 阶段接入点：因子发现与评估聚合改由 Catalog 统一提供。
        # 这里仍保留列表页现有响应结构，避免一次性改动前端。
        catalog = FactorCatalogService()
        known_groups = catalog.repository.list_source_groups()
        processed_factors = []
        for summary in catalog.list_factors():
            try:
                factor_def = catalog.get_factor(summary.factor_id)
                if factor_def is None:
                    continue

                comp_artifacts = factor_def.artifacts
                computation_data = {
                    'function_file': comp_artifacts.function_file,
                    'formula_file': comp_artifacts.formula_file,
                    'formula': comp_artifacts.formula_inline,
                    'algorithm_name': comp_artifacts.algorithm_name,
                    'proxy_key': comp_artifacts.proxy_key,
                    'factor_name': comp_artifacts.factor_name,
                    'entry_point': comp_artifacts.entry_point,
                    **(comp_artifacts.extra or {}),
                }

                formula_preview = None
                if factor_def.computation_type == 'formula':
                    formula_preview = comp_artifacts.formula_inline or None
                elif factor_def.computation_type == 'function':
                    formula_preview = computation_data.get('function_code') or None

                agg = summary.evaluation_aggregation
                avg_metrics = agg.avg_metrics or {}

                category = factor_def.source_group
                subcategory = (factor_def.factor_kind or '').lower()
                traits = factor_def.traits.to_dict()
                factor_id_lower = factor_def.factor_id.lower()
                factor_name_lower = factor_def.name.lower()

                is_event_factor = bool(traits.get('is_event')) or subcategory == 'event'
                if not is_event_factor and category == 'pattern':
                    is_event_factor = True
                if not is_event_factor:
                    event_keywords = ['cross', 'gap', 'breakout',
                                      'breakdown', 'signal', 'event', 'direction']
                    for keyword in event_keywords:
                        if keyword in factor_id_lower or keyword in factor_name_lower:
                            is_event_factor = True
                            break

                data_requirement = category if category in known_groups else 'basic_kline'
                if is_event_factor:
                    data_requirement = 'event_factor'
                if subcategory == 'mined':
                    data_requirement = 'mined_factor'

                processed_factors.append({
                    'id': factor_def.factor_id,
                    'name': factor_def.name,
                    'description': factor_def.description,
                    'type': category,
                    'subcategory': factor_def.factor_kind,
                    'source': factor_def.source_group,
                    'data_requirement': data_requirement,
                    'created_at': factor_def.metadata.get('created_at'),
                    'is_window': traits.get('is_window', factor_def.metadata.get('is_window')),
                    'min_warmup_bars': traits.get('min_warmup_bars', factor_def.metadata.get('min_warmup_bars')),
                    'source_family': factor_def.metadata.get('source_family'),
                    'computation_type': factor_def.computation_type,
                    'formula': formula_preview,
                    'evaluated': agg.evaluated,
                    'evaluations_count': agg.eval_count,
                    'avg_metrics': avg_metrics,
                    'last_evaluated_at': agg.last_evaluated_at,
                    'ic': avg_metrics.get('ic_pearson'),
                    'rank_ic': avg_metrics.get('ic_spearman'),
                    'icir': avg_metrics.get('icir'),
                    'pos_ic_ratio': avg_metrics.get('ic_positive_ratio'),
                    'long_short_return': avg_metrics.get('long_short_return'),
                    'win_rate': avg_metrics.get('win_rate'),
                    'ir': avg_metrics.get('icir'),
                    'sharpe': avg_metrics.get('sharpe_ratio'),
                })
            except Exception as e:
                print(f"❌ Catalog 读取因子失败 {summary.factor_id}: {e}")
                continue

        response_payload = {
            'success': True, 'factors': processed_factors, 'total': len(processed_factors)}
        with _factor_list_cache_lock:
            _factor_list_cache["payload"] = response_payload
            _factor_list_cache["expires_at"] = time.time() + \
                _FACTOR_LIST_CACHE_TTL_SEC
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
    n_ic_segments,
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
            n_ic_segments=n_ic_segments,
        )
        result = cs_evaluator.evaluate_cross_sectional(
            data_dict, factor_id, engine, timeframe=base_timeframe)
        elapsed = time.time() - t0
        if result.get('success'):
            ic_payload = result.get('ic') or {}
            # rank_ic_series 数据量较大（每个因子数千条），评估流里不再回传给前端，
            # 如需 IC 时序相关性，由独立端点按需重算。
            if isinstance(ic_payload, dict) and 'rank_ic_series' in ic_payload:
                ic_payload = {k: v for k, v in ic_payload.items()
                              if k != 'rank_ic_series'}
            return {
                'factor_id': factor_id,
                'success': True,
                'n_symbols': result.get('n_symbols'),
                'n_periods': result.get('n_periods'),
                'n_periods_total': result.get('n_periods_total'),
                'n_periods_ic': result.get('n_periods_ic'),
                'n_periods_returns': result.get('n_periods_returns'),
                'ic': ic_payload,
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
        market_data = load_local_market_data(
            symbol, timeframe, start_date, end_date, exchange, trade_type)

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
    """导出因子（v4：按一级分类目录动态查找定义文件）。"""
    try:
        export_dir = FACTOR_LIBRARY_DIR / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)

        definition_file = _find_factor_definition_file(factor_id)
        if definition_file is None:
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


def _find_factor_definition_file(factor_id: str):
    from factor_miner.core.factor_repository import FactorRepository
    repo = FactorRepository()
    return repo.find_definition_file(factor_id)


def _find_factor_function_file(factor_id: str):
    from factor_miner.core.factor_repository import FactorRepository
    from factor_miner.core.factor_executor import FactorExecutor
    repo = FactorRepository()
    factor_def = repo.load_definition(factor_id)
    if factor_def is None:
        return None
    executor = FactorExecutor(repo)
    if factor_def.artifacts.function_file:
        resolved = executor._resolve_artifact_path(factor_def.artifacts.function_file)
        if resolved is not None and resolved.exists():
            return resolved
    return None


_repo_instance = None


def _get_repo():
    global _repo_instance
    if _repo_instance is None:
        _repo_instance = FactorRepository()
    return _repo_instance


@bp.route('/batch_delete', methods=['POST'])
def batch_delete_factors():
    """批量删除因子（从因子库中移除定义文件）"""
    try:
        payload = request.get_json() or {}
        factor_ids = payload.get('factor_ids') or []
        if not factor_ids:
            return jsonify({'success': False, 'message': '未选择要删除的因子'})

        from factor_miner.core.factor_lifecycle import FactorLifecycleService
        lifecycle = FactorLifecycleService()

        deleted = []
        failed = []
        for fid in factor_ids:
            try:
                result = lifecycle.delete_factor(fid, cascade=True)
                if result.success:
                    deleted.append(fid)
                else:
                    failed.append(fid)
            except Exception:
                failed.append(fid)

        with _factor_list_cache_lock:
            _factor_list_cache["payload"] = None

        return jsonify({
            'success': True,
            'deleted': deleted,
            'failed': failed,
            'deleted_count': len(deleted),
            'failed_count': len(failed),
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'批量删除失败: {str(e)}'}), 500


@bp.route('/evaluations/<factor_id>')
def get_evaluations(factor_id: str):
    """获取某因子的历史评估记录（多结果结构）"""
    try:
        catalog = FactorCatalogService()
        payload = catalog.repository.load_evaluations(factor_id)
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
    request_id = str(payload.get('request_id')
                     or '').strip() or _LEGACY_CANCEL_ID
    n_groups = payload.get('n_groups', 5)
    predict_step = payload.get('predict_step', 1)
    min_coverage = payload.get('min_coverage', 0.3)
    min_sample = payload.get('min_sample', 30)
    transaction_cost = payload.get('transaction_cost', 0.0)
    ic_decay_max_lag = payload.get('ic_decay_max_lag', 5)

    if not factor_ids or not symbols or not timeframes or not start_date or not end_date:
        return jsonify({'success': False, 'message': '缺少必要参数（factor_ids/symbols/timeframes/start_date/end_date）'})

    total_tasks = len(factor_ids) * len(symbols) * len(timeframes)
    logger.info(
        f"批量评估请求: {len(factor_ids)} 因子 × {len(symbols)} 币种 × {len(timeframes)} 时间框架 = {total_tasks} 任务 (request_id={request_id})")

    cancel_event = _get_cancel_event(request_id)
    cancel_event.clear()

    def generate():
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
                if cancel_event.is_set():
                    return {
                        'factor_id': factor_id, 'symbol': symbol, 'timeframe': timeframe,
                        'success': False, 'message': '评估已取消'
                    }
                try:
                    market_data = load_local_market_data(
                        symbol, timeframe, start_date, end_date, exchange, trade_type)
                    if market_data is None or market_data.empty:
                        return {
                            'factor_id': factor_id, 'symbol': symbol, 'timeframe': timeframe,
                            'success': False, 'message': '无法加载市场数据'
                        }
                    factor_values = engine.compute_single_factor(
                        factor_id, market_data)
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
                    coverage = len(factor_values) / len(market_data) if len(market_data) > 0 else 0
                    if coverage < min_coverage:
                        return {
                            'factor_id': factor_id, 'symbol': symbol, 'timeframe': timeframe,
                            'success': False, 'message': f'覆盖率不足：{coverage:.1%} < {min_coverage:.1%}'
                        }
                    if len(factor_values) < min_sample:
                        return {
                            'factor_id': factor_id, 'symbol': symbol, 'timeframe': timeframe,
                            'success': False, 'message': f'数据不足：样本数 {len(factor_values)} < {min_sample}'
                        }
                    eval_res = evaluator.stats.comprehensive_factor_analysis(
                        factor=factor_values, returns=returns, factor_name=factor_id,
                        ic_decay_max_lag=ic_decay_max_lag, n_groups=n_groups)
                    if transaction_cost > 0 and 'long_short_return' in eval_res:
                        eval_res['long_short_return_net'] = eval_res['long_short_return'] - transaction_cost
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
            _register_active_executor(request_id, executor)
            try:
                futures = {}
                for factor_id in factor_ids:
                    for symbol in symbols:
                        for timeframe in timeframes:
                            future = executor.submit(
                                evaluate_single_task, factor_id, symbol, timeframe)
                            futures[future] = (factor_id, symbol, timeframe)

                for future in as_completed(futures):
                    if cancel_event.is_set():
                        break
                    if _shutdown_event.is_set():
                        for f in futures:
                            f.cancel()
                        _force_stop_executor(executor)
                        raise GeneratorExit("服务关闭")
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
                cancel_event.set()
                for f in futures:
                    f.cancel()
                _pop_active_executor(request_id)
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            except KeyboardInterrupt:
                cancel_event.set()
                for f in futures:
                    f.cancel()
                _pop_active_executor(request_id)
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                _pop_active_executor(request_id)
                executor.shutdown(wait=True)

            elapsed = time.time() - t0
            success_count = sum(1 for r in all_results if r.get('success'))
            fail_count = total_tasks - success_count
            logger.info(
                f"批量评估完成: 成功 {success_count}/{total_tasks}, 失败 {fail_count}, 耗时 {elapsed:.1f}s")

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
        finally:
            _discard_cancel_event(request_id)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        }
    )


@bp.route('/rolling-ic', methods=['POST'])
def rolling_ic():
    """计算单个因子的滚动 IC 时序"""
    try:
        data = request.get_json()
        factor_id = data.get('factor_id')
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        exchange = data.get('exchange', 'binance')
        trade_type = data.get('trade_type', 'futures')
        window = data.get('window', 60)

        if not all([factor_id, symbol, timeframe, start_date, end_date]):
            return jsonify({'success': False, 'message': '缺少必要参数'})

        engine = get_global_engine()
        market_data = load_local_market_data(
            symbol, timeframe, start_date, end_date, exchange, trade_type)

        if market_data is None or market_data.empty:
            return jsonify({'success': False, 'message': '无法加载市场数据'})

        factor_values = engine.compute_single_factor(factor_id, market_data)
        if factor_values is None:
            return jsonify({'success': False, 'message': '因子计算失败'})

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

        if len(factor_values) < window + 10:
            return jsonify({'success': False, 'message': f'数据不足：需要至少 {window+10} 个样本'})

        lagged_factor = factor_values.shift(1)
        valid_mask = lagged_factor.notna() & returns.notna()
        lagged_factor = lagged_factor[valid_mask]
        returns_aligned = returns[valid_mask]

        if len(lagged_factor) < window + 10:
            return jsonify({'success': False, 'message': f'数据不足：需要至少 {window+10} 个有效样本'})

        stats_obj = FactorStatistics()
        rolling_ic_series = stats_obj.calculate_rolling_ic(
            lagged_factor, returns_aligned, window=window)

        valid_mask = rolling_ic_series.notna()
        dates = rolling_ic_series.index[valid_mask].strftime('%Y-%m-%d %H:%M:%S').tolist()
        ic_values = rolling_ic_series[valid_mask].tolist()

        return jsonify({
            'success': True,
            'dates': dates,
            'rolling_ic': ic_values,
            'window': window
        })

    except Exception as e:
        return jsonify({'success': False, 'message': f'滚动IC计算失败: {str(e)}'})


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
    base_timeframe = payload.get(
        'base_timeframe') or payload.get('timeframe', '1h')
    factor_timeframe = payload.get('factor_timeframe') or base_timeframe
    factor_bar_mode = str(payload.get('factor_bar_mode', 'completed')).lower()
    start_date = payload.get('start_date')
    end_date = payload.get('end_date')
    # 可选：OOS 起始日期。提供时，同一批因子会在 IS=[start, oos) 与 OOS=[oos, end] 两段
    # 分别跑一次截面评估，SSE 返回同时包含 is / oos 两套指标；不填则走原逻辑（整段一次）。
    oos_start_date = payload.get('oos_start_date')
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
    n_ic_segments = payload.get('n_ic_segments', 4)
    parallel_backend = str(payload.get('parallel_backend', 'auto')).lower()
    request_id = str(payload.get('request_id')
                     or '').strip() or _LEGACY_CANCEL_ID
    try:
        heartbeat_interval_sec = float(
            payload.get('heartbeat_interval_sec', 15))
    except (TypeError, ValueError):
        heartbeat_interval_sec = 15.0

    logger.info(
        f"截面评估请求: {len(factor_ids)} 个因子, {len(symbols)} 个币种, {start_date} ~ {end_date} (request_id={request_id})")

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
    if max_lookback > 5000:
        return jsonify({
            'success': False,
            'message': '参数 max_lookback 不能大于 5000（过大会导致内存压力，建议 100-500）'
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
        treat_zero_as_invalid = treat_zero_as_invalid.strip().lower() in ('1',
                                                                          'true', 'yes', 'y', 'on')
    else:
        treat_zero_as_invalid = bool(treat_zero_as_invalid)
    if isinstance(enable_data_cleaning, str):
        enable_data_cleaning = enable_data_cleaning.strip(
        ).lower() in ('1', 'true', 'yes', 'y', 'on')
    else:
        enable_data_cleaning = bool(enable_data_cleaning)
    if isinstance(remove_zero_volume, str):
        remove_zero_volume = remove_zero_volume.strip(
        ).lower() in ('1', 'true', 'yes', 'y', 'on')
    else:
        remove_zero_volume = bool(remove_zero_volume)
    if isinstance(enable_outlier_treatment, str):
        enable_outlier_treatment = enable_outlier_treatment.strip().lower() in ('1',
                                                                                'true', 'yes', 'y', 'on')
    else:
        enable_outlier_treatment = bool(enable_outlier_treatment)
    if isinstance(compute_fsc, str):
        compute_fsc = compute_fsc.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
    else:
        compute_fsc = bool(compute_fsc)
    if isinstance(compute_ic_decay_curve, str):
        compute_ic_decay_curve = compute_ic_decay_curve.strip().lower() in ('1',
                                                                            'true', 'yes', 'y', 'on')
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

    try:
        n_ic_segments = int(n_ic_segments)
    except (TypeError, ValueError):
        return jsonify({'success': False, 'message': '参数 n_ic_segments 必须为整数'}), 400
    if n_ic_segments < 2 or n_ic_segments > 12:
        return jsonify({'success': False, 'message': '参数 n_ic_segments 必须在 [2, 12] 区间'}), 400

    base_timeframe = str(base_timeframe or '1h').lower()
    factor_timeframe = str(factor_timeframe or base_timeframe).lower()
    if factor_bar_mode not in ('completed', 'intrabar', 'intrabar_strict', 'offset_resample'):
        return jsonify({
            'success': False,
            'message': '参数 factor_bar_mode 仅支持 completed / intrabar / intrabar_strict / offset_resample'
        }), 400

    # 解析 oos_start_date：要求严格落在 (start_date, end_date) 内
    oos_start_dt = None
    if oos_start_date:
        try:
            oos_start_dt = pd.Timestamp(oos_start_date)
        except Exception:
            return jsonify({
                'success': False,
                'message': '参数 oos_start_date 无法解析为时间戳'
            }), 400
        try:
            _start_dt = pd.Timestamp(start_date)
            _end_dt = pd.Timestamp(end_date)
        except Exception:
            return jsonify({
                'success': False,
                'message': 'start_date / end_date 格式无效'
            }), 400
        if not (_start_dt < oos_start_dt < _end_dt):
            return jsonify({
                'success': False,
                'message': 'oos_start_date 必须严格位于 start_date 与 end_date 之间'
            }), 400
    oos_enabled = oos_start_dt is not None

    cancel_event = _get_cancel_event(request_id)
    cancel_event.clear()

    def generate():
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
                n_ic_segments=n_ic_segments,
            )
            total = len(factor_ids)
            logger.info(f"开始截面评估，共 {total} 个因子")

            # 根据因子 metadata.requires_extras 汇总本轮要 join 的额外数据。
            # 细粒度：metrics（OI/LSR 等 5m feather）、funding、basis（mark+index 派生基差）。
            # 兼容旧标记 derivatives → metrics+basis；未知 token 记 debug 日志并跳过。
            _VALID_EXTRA_KINDS = frozenset({'metrics', 'funding', 'basis', 'mark', 'index'})
            _LEGACY_DERIVATIVES_EXPAND = frozenset({'metrics', 'basis'})
            _extras_include = set()
            for _fid in factor_ids:
                try:
                    _fdef = _get_repo().load_definition(_fid)
                    if not _fdef or not _fdef.metadata:
                        continue
                    for _req in (_fdef.metadata.get('requires_extras') or []):
                        if _req == 'derivatives':
                            _extras_include.update(_LEGACY_DERIVATIVES_EXPAND)
                        elif _req in _VALID_EXTRA_KINDS:
                            _extras_include.add(_req)
                        else:
                            logger.debug(
                                "截面评估: 因子 %s 的 requires_extras 含未知项 %r，已忽略",
                                _fid, _req,
                            )
                except Exception:
                    pass
            _extras_include = sorted(_extras_include) if _extras_include else None
            logger.info(f"截面评估额外数据 include={_extras_include}")

            _data_loader = DataLoader()

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
                if cancel_event.is_set():
                    return (symbol, None)
                try:
                    md = load_local_market_data(
                        symbol, base_timeframe, start_date, end_date, exchange, trade_type
                    )
                    if md is not None and not md.empty:
                        if _extras_include:
                            md = _data_loader.join_extras(
                                md, symbol, interval=base_timeframe,
                                include=_extras_include,
                            )
                        return (symbol, md)
                except Exception as e:
                    logger.warning(f"加载 {symbol} 失败: {e}")
                return (symbol, None)

            load_workers = min(total_symbols, os.cpu_count() or 4, 8)
            loader = ThreadPoolExecutor(max_workers=load_workers)
            try:
                load_futures = {loader.submit(
                    load_single_symbol, s): s for s in symbols}
                loaded_count = 0
                for future in as_completed(load_futures):
                    if cancel_event.is_set():
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
                cancel_event.set()
                for f in load_futures:
                    f.cancel()
                loader.shutdown(wait=False, cancel_futures=True)
                raise
            except KeyboardInterrupt:
                cancel_event.set()
                for f in load_futures:
                    f.cancel()
                loader.shutdown(wait=False, cancel_futures=True)
                raise
            loader.shutdown(wait=True)

            load_elapsed = time.time() - load_t0
            logger.info(
                f"市场数据加载完成: {len(data_dict)}/{len(symbols)} 个币种, 耗时 {load_elapsed:.1f}s")

            if len(data_dict) < 2:
                logger.error(f"有效币种数量不足: {len(data_dict)} < 2")
                yield _sse_event('error', {
                    'message': f'有效币种数量不足（{len(data_dict)} < 2）'
                })
                yield _sse_event('done', {
                    'total': total,
                    'batch_total': total,
                    'success_count': 0,
                    'fail_count': 0,
                    'eval_elapsed': 0,
                    'total_elapsed': round(time.time() - load_t0, 2) if 'load_t0' in locals() else 0,
                    'workers': 0,
                    'parallel_backend': parallel_backend,  # effective_backend 尚未确定，用原始请求值
                    'request_id': request_id,
                    'oos_enabled': oos_enabled,
                    'oos_start_date': str(oos_start_dt) if oos_start_dt is not None else None,
                })
                return

            # OOS 模式：按 oos_start_dt 切成 IS / OOS 两份 data_dict
            # 切片发生在传入评估器之前，因此 IS 段的 future_returns 不会偷看 OOS 数据
            # （因为 pct_change+shift 需要未来价格，切掉之后对应的标签会变 NaN 被丢弃）。
            data_dict_is = None
            data_dict_oos = None
            if oos_enabled:
                data_dict_is = {}
                data_dict_oos = {}
                for _sym, _df in data_dict.items():
                    if _df is None or _df.empty:
                        continue
                    try:
                        _idx = _df.index
                        if not isinstance(_idx, pd.DatetimeIndex):
                            _idx = pd.to_datetime(_idx, errors='coerce')
                            _df = _df.copy()
                            _df.index = _idx
                        if isinstance(_df.index, pd.DatetimeIndex):
                            _df = _df.loc[_df.index.notna()]
                        _cut = oos_start_dt
                        if isinstance(_df.index, pd.DatetimeIndex) and _df.index.tz is not None:
                            if _cut.tz is None:
                                _cut = _cut.tz_localize(_df.index.tz)
                            else:
                                _cut = _cut.tz_convert(_df.index.tz)
                        elif _cut.tz is not None:
                            _cut = _cut.tz_convert(None)

                        _is_part = _df.loc[_df.index < _cut]
                        _oos_part = _df.loc[_df.index >= _cut]
                    except Exception:
                        continue
                    if _is_part is not None and not _is_part.empty:
                        data_dict_is[_sym] = _is_part
                    if _oos_part is not None and not _oos_part.empty:
                        data_dict_oos[_sym] = _oos_part
                if len(data_dict_is) < 2 or len(data_dict_oos) < 2:
                    yield _sse_event('error', {
                        'message': (f'OOS 切分后有效币种不足：IS={len(data_dict_is)}, '
                                    f'OOS={len(data_dict_oos)}（均需 ≥2）')
                    })
                    yield _sse_event('done', {
                        'total': total,
                        'batch_total': total,
                        'success_count': 0,
                        'fail_count': 0,
                        'eval_elapsed': 0,
                        'total_elapsed': round(time.time() - load_t0, 2) if 'load_t0' in locals() else 0,
                        'workers': 0,
                        'parallel_backend': parallel_backend,  # effective_backend 尚未确定，用原始请求值
                        'request_id': request_id,
                        'oos_enabled': oos_enabled,
                        'oos_start_date': str(oos_start_dt) if oos_start_dt is not None else None,
                    })
                    return
                logger.info(
                    f"OOS 切分完成: IS 段 {len(data_dict_is)} 币种, OOS 段 {len(data_dict_oos)} 币种, "
                    f"切点 = {oos_start_dt}"
                )

            yield _sse_event('progress', {
                'phase': 'evaluating',
                'message': (
                    f'数据加载完成 ({len(data_dict)} 个币种, {load_elapsed:.1f}s)，'
                    + (f'OOS 切点 {oos_start_dt.strftime("%Y-%m-%d")}，每因子将分别跑 IS/OOS 两次。' if oos_enabled else '')
                    + f'开始评估 {total} 个因子...'
                ),
                'completed': 0,
                'total': total,
                'oos_enabled': oos_enabled,
            })

            def _run_single_slice(factor_id, slice_data_dict, eval_start_date=None):
                """在给定的 data_dict 切片上跑一次截面评估，返回统一结构。"""
                t0 = time.time()
                try:
                    result = cs_evaluator.evaluate_cross_sectional(
                        slice_data_dict, factor_id, engine, timeframe=base_timeframe,
                        eval_start_date=eval_start_date,
                    )
                    elapsed = time.time() - t0
                    if result.get('success'):
                        ic_payload = result.get('ic') or {}
                        # rank_ic_series 不在 SSE 回传，IC 时序相关性按需重算
                        if isinstance(ic_payload, dict) and 'rank_ic_series' in ic_payload:
                            ic_payload = {
                                k: v for k, v in ic_payload.items() if k != 'rank_ic_series'}
                        return {
                            'success': True,
                            'n_symbols': result.get('n_symbols'),
                            'n_periods': result.get('n_periods'),
                            'n_periods_total': result.get('n_periods_total'),
                            'n_periods_ic': result.get('n_periods_ic'),
                            'n_periods_returns': result.get('n_periods_returns'),
                            'ic': ic_payload,
                            'returns': result.get('returns'),
                            'coverage': result.get('coverage'),
                            'summary': result.get('summary'),
                            'elapsed': round(elapsed, 2),
                        }
                    return {
                        'success': False,
                        'message': result.get('message', '评估失败'),
                        'elapsed': round(elapsed, 2),
                    }
                except Exception as ex:
                    elapsed = time.time() - t0
                    return {
                        'success': False,
                        'message': str(ex),
                        'elapsed': round(elapsed, 2),
                    }

            def evaluate_one_factor(factor_id):
                """单个因子的截面评估（线程安全）。OOS 开启时跑 IS / OOS 两次。"""
                if cancel_event.is_set():
                    return {
                        'factor_id': factor_id,
                        'success': False,
                        'message': '评估已取消',
                        'elapsed': 0
                    }
                if not oos_enabled:
                    r = _run_single_slice(factor_id, data_dict)
                    if r.get('success'):
                        logger.debug(
                            f"因子 {factor_id} 评估成功，耗时 {r.get('elapsed')}s")
                    else:
                        logger.warning(
                            f"因子 {factor_id} 评估失败: {r.get('message')}")
                    r['factor_id'] = factor_id
                    r['oos_enabled'] = False
                    return r

                # OOS 模式：依次跑 IS / OOS
                is_res = _run_single_slice(factor_id, data_dict_is)
                if cancel_event.is_set():
                    return {
                        'factor_id': factor_id,
                        'success': False,
                        'message': '评估已取消',
                        'elapsed': round(is_res.get('elapsed') or 0, 2),
                        'oos_enabled': True,
                        'is': is_res,
                    }
                oos_res = _run_single_slice(factor_id, data_dict, eval_start_date=oos_start_dt)

                combined = {
                    'factor_id': factor_id,
                    'oos_enabled': True,
                    # 以 OOS 的成功状态作为因子是否通过的依据（OOS 是最终判据）
                    'success': bool(oos_res.get('success')),
                    'elapsed': round((is_res.get('elapsed') or 0) + (oos_res.get('elapsed') or 0), 2),
                    'is': is_res,
                    'oos': oos_res,
                }
                if oos_res.get('success'):
                    # 同时把 OOS 的核心字段铺到顶层，便于现有前端逻辑兜底兼容
                    for _key in ('n_symbols', 'n_periods', 'n_periods_total',
                                 'n_periods_ic', 'n_periods_returns',
                                 'ic', 'returns', 'coverage', 'summary'):
                        combined[_key] = oos_res.get(_key)
                else:
                    combined['message'] = oos_res.get('message', 'OOS 段评估失败')

                if combined['success']:
                    logger.debug(
                        f"因子 {factor_id} IS/OOS 评估成功，耗时 {combined['elapsed']}s "
                        f"(IS={is_res.get('elapsed')}s, OOS={oos_res.get('elapsed')}s)"
                    )
                else:
                    logger.warning(
                        f"因子 {factor_id} OOS 段评估失败: {combined.get('message')} "
                        f"(IS success={is_res.get('success')})"
                    )
                return combined

            cpu_count = os.cpu_count() or 4

            data_dict_size_mb = sum(df.memory_usage(deep=True).sum(
            ) for df in data_dict.values()) / (1024 * 1024) if data_dict else 0
            _PROCESS_DATA_SIZE_LIMIT_MB = 50
            _PROCESS_MIN_FACTOR_COUNT = 8

            effective_backend = parallel_backend
            if parallel_backend == 'auto':
                if os.name == 'nt':
                    effective_backend = 'thread'
                    logger.info("auto 模式: 检测到 Windows，强制使用 thread 后端（避免进程池在 Ctrl+C 时残留）")
                elif (data_dict_size_mb < _PROCESS_DATA_SIZE_LIMIT_MB
                        and len(factor_ids) >= _PROCESS_MIN_FACTOR_COUNT):
                    effective_backend = 'process'
                    logger.info(f"auto 模式: 数据量 {data_dict_size_mb:.1f}MB < {_PROCESS_DATA_SIZE_LIMIT_MB}MB, "
                                f"因子数 {len(factor_ids)} >= {_PROCESS_MIN_FACTOR_COUNT}, 选择 process 后端")
                else:
                    effective_backend = 'thread'
                    logger.info(
                        f"auto 模式: 数据量 {data_dict_size_mb:.1f}MB, 因子数 {len(factor_ids)}, 选择 thread 后端")
            elif parallel_backend == 'process' and data_dict_size_mb >= _PROCESS_DATA_SIZE_LIMIT_MB:
                logger.warning(f"数据量 {data_dict_size_mb:.1f}MB >= {_PROCESS_DATA_SIZE_LIMIT_MB}MB, "
                               f"进程模式序列化开销过大，自动降级为 thread 后端")
                effective_backend = 'thread'
            elif parallel_backend == 'process' and os.name == 'nt':
                logger.warning("检测到 Windows，process 后端可能在 Ctrl+C 时残留，已自动降级为 thread 后端")
                effective_backend = 'thread'

            if oos_enabled and effective_backend == 'process':
                # OOS 模式需要在两份独立的 data_dict 上跑两次评估；
                # 当前 _evaluate_factor_process_worker 签名仅接受单份 data_dict，
                # 为避免重复 IPC 序列化开销与复杂度，这里直接降级为 thread 后端。
                logger.info("OOS 模式启用，自动从 process 降级为 thread 后端（简化数据共享）")
                effective_backend = 'thread'

            if effective_backend == 'process':
                max_workers = min(len(factor_ids), max(cpu_count - 1, 1), 4)
                executor_cls = ProcessPoolExecutor
            else:
                max_workers = min(len(factor_ids), cpu_count, 8)
                executor_cls = ThreadPoolExecutor
            completed_count = 0
            all_results = []
            eval_t0 = time.time()
            logger.info(
                f"开始并行评估，后端={effective_backend}, workers={max_workers}, 数据量={data_dict_size_mb:.1f}MB")

            executor = executor_cls(max_workers=max_workers)
            pending = set()
            try:
                _register_active_executor(request_id, executor)
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
                            n_ic_segments,
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
                    done, pending = wait(
                        pending, timeout=1.0, return_when=FIRST_COMPLETED)

                    if cancel_event.is_set():
                        logger.info("检测到取消信号，停止评估循环...")
                        break

                    if _shutdown_event.is_set():
                        logger.info("检测到全局关闭信号，停止评估循环...")
                        for f in pending:
                            f.cancel()
                        _force_stop_executor(executor)
                        raise GeneratorExit("服务关闭")

                    if not done:
                        now = time.time()
                        if now - heartbeat_ts >= max(heartbeat_interval_sec, 1.0):
                            elapsed = now - eval_t0
                            avg_sec = (
                                elapsed / completed_count) if completed_count > 0 else 0
                            eta_sec = (avg_sec * (total - completed_count)
                                       ) if completed_count > 0 else None
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

                        if result.get('success'):
                            try:
                                save_evaluation_results(factor_id, result, {
                                    'evaluation_type': 'cross_sectional',
                                    'timeframe': base_timeframe,
                                    'n_groups': n_groups,
                                    'normalize_method': normalize_method,
                                    'predict_step': predict_step,
                                    'sample_step': sample_step,
                                    'oos_enabled': oos_enabled,
                                    'oos_start_date': str(oos_start_dt) if oos_start_dt is not None else None,
                                })
                            except Exception as save_ex:
                                logger.warning(f"持久化因子 {factor_id} 截面评估结果失败: {save_ex}")
            except GeneratorExit:
                logger.info("截面评估被中断（GeneratorExit），正在取消所有待处理任务...")
                cancel_event.set()
                for f in pending:
                    f.cancel()
                _force_stop_executor(executor)
                raise
            except KeyboardInterrupt:
                logger.info("截面评估被中断（KeyboardInterrupt），正在取消所有待处理任务...")
                cancel_event.set()
                for f in pending:
                    f.cancel()
                _force_stop_executor(executor)
                raise
            else:
                executor.shutdown(wait=True)

            eval_elapsed = time.time() - eval_t0
            total_elapsed = time.time() - load_t0

            success_count = sum(1 for r in all_results if r.get('success'))
            fail_count = sum(1 for r in all_results if not r.get('success'))
            logger.info(
                f"截面评估完成: 成功 {success_count}/{total}, 失败 {fail_count}, 总耗时 {total_elapsed:.1f}s")

            yield _sse_event('done', {
                'total': total,
                'batch_total': total,
                'success_count': success_count,
                'fail_count': fail_count,
                'eval_elapsed': round(eval_elapsed, 2),
                'total_elapsed': round(total_elapsed, 2),
                'workers': max_workers,
                'parallel_backend': effective_backend,
                'request_id': request_id,
                'oos_enabled': oos_enabled,
                'oos_start_date': str(oos_start_dt) if oos_start_dt is not None else None,
            })
        except Exception as e:
            logger.error(f"截面评估过程发生异常: {e}", exc_info=True)
            yield _sse_event('error', {
                'message': f'评估过程发生错误: {str(e)}'
            })
            yield _sse_event('done', {
                'total': len(factor_ids),
                'batch_total': len(factor_ids),
                'success_count': len([r for r in all_results if r.get('success')]) if 'all_results' in locals() else 0,
                'fail_count': len([r for r in all_results if not r.get('success')]) if 'all_results' in locals() else len(factor_ids),
                'eval_elapsed': round(time.time() - eval_t0, 2) if 'eval_t0' in locals() else 0,
                'total_elapsed': round(time.time() - load_t0, 2) if 'load_t0' in locals() else 0,
                'error': str(e),
                'request_id': request_id,
            })
        finally:
            _pop_active_executor(request_id)
            _discard_cancel_event(request_id)

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
    """取消正在进行的截面评估 / 组合回测 / 相关性计算

    - 若请求体中带 request_id，仅取消对应请求（推荐）
    - 未带 request_id 时回退为"取消所有"，用于兼容旧前端
    """
    payload = request.get_json(silent=True) or {}
    request_id = str(payload.get('request_id') or '').strip()
    triggered = _set_cancel(request_id) if request_id else _set_cancel('')
    if request_id:
        _force_stop_executor(_pop_active_executor(request_id))
    else:
        for _, ex in _pop_all_active_executors():
            _force_stop_executor(ex)
    return jsonify({
        'success': True,
        'message': '取消信号已发送',
        'request_id': request_id or None,
        'triggered': triggered,
    })


def _sse_event(event_type: str, data: dict) -> str:
    """构造 SSE 事件字符串"""
    payload = json.dumps(_sanitize_for_json(
        {'type': event_type, **data}), ensure_ascii=False)
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
    crypto_symbols = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA',
                      'DOGE', 'LINK', 'LPT', 'MOVR', 'PEOPLE', 'SUI', 'FIL']

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
    """转调核心层的评估结果保存，并刷新索引缓存"""
    try:
        core_save_evaluation_results(factor_id, results, metadata)
        with _factor_list_cache_lock:
            _factor_list_cache["payload"] = None
            _factor_list_cache["expires_at"] = 0.0
        try:
            catalog = FactorCatalogService()
            catalog.invalidate_index_cache()
            catalog.update_index_entry(factor_id)
        except Exception:
            pass
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
    """导出因子数据（v4：统一导出到 factorlib/exports 子目录）。"""
    export_dir = FACTOR_LIBRARY_DIR / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    identifier = factor_info.get(
        'identifier') or factor_info.get('id') or 'factor'
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
        data_dir = Path(__file__).parent.parent.parent / \
            "data" / exchange / trade_type

        # 解析交易对格式
        # API 返回的交易对格式是 BASE_QUOTE（如 BTC_USDT 或 ETH_BTC），
        # 实际文件名为 {BASE}_{QUOTE}_{SETTLE}-{timeframe}-{trade_type}.feather
        # 永续合约通常 settle == quote；现货也按此约定退化处理。
        if '_' in symbol:
            parts = [p for p in symbol.split('_') if p]
            base_symbol = parts[0] if len(parts) >= 1 else symbol
            quote_symbol = parts[1] if len(parts) >= 2 else 'USDT'
            settle_symbol = parts[2] if len(parts) >= 3 else quote_symbol
        else:
            base_symbol = symbol
            quote_symbol = 'USDT'
            settle_symbol = 'USDT'

        candidate_filenames = [
            f"{base_symbol}_{quote_symbol}_{settle_symbol}-{timeframe}-{trade_type}.feather",
        ]
        # 兼容旧目录里 settle 与 quote 相同时被写成 BASE_QUOTE_QUOTE 的命名
        if settle_symbol != quote_symbol:
            candidate_filenames.append(
                f"{base_symbol}_{quote_symbol}_{quote_symbol}-{timeframe}-{trade_type}.feather"
            )

        file_path = None
        for fname in candidate_filenames:
            cand = data_dir / fname
            if cand.exists():
                file_path = cand
                break

        if file_path is None:
            print(f"数据文件不存在: {data_dir / candidate_filenames[0]}")
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

        print(
            f"✅ {symbol} [{timeframe}]: {len(data)} 条 ({data.index.min().date()} ~ {data.index.max().date()})")
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
    request_id = _LEGACY_CANCEL_ID
    cancel_event = None
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
        enable_data_cleaning = payload.get('enable_data_cleaning', False)
        remove_zero_volume = payload.get('remove_zero_volume', True)
        liquidity_filter_ratio = payload.get('liquidity_filter_ratio', 0.5)
        request_id = str(payload.get('request_id')
                         or '').strip() or _LEGACY_CANCEL_ID

        try:
            max_lookback = int(max_lookback)
        except (TypeError, ValueError):
            max_lookback = 200
        if max_lookback < 1:
            max_lookback = 200
        if max_lookback > 5000:
            return jsonify({
                'success': False,
                'message': '参数 max_lookback 不能大于 5000（建议 100-500）'
            }), 400

        cancel_event = _get_cancel_event(request_id)
        cancel_event.clear()

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

        _VALID_EXTRA_KINDS_ENS = frozenset({'metrics', 'funding', 'basis', 'mark', 'index'})
        _LEGACY_DERIVATIVES_EXPAND_ENS = frozenset({'metrics', 'basis'})
        _extras_include_ens = set()
        for _fid in factor_ids:
            try:
                _fdef = _get_repo().load_definition(_fid)
                if not _fdef or not _fdef.metadata:
                    continue
                for _req in (_fdef.metadata.get('requires_extras') or []):
                    if _req == 'derivatives':
                        _extras_include_ens.update(_LEGACY_DERIVATIVES_EXPAND_ENS)
                    elif _req in _VALID_EXTRA_KINDS_ENS:
                        _extras_include_ens.add(_req)
            except Exception:
                pass
        _extras_include_ens = sorted(_extras_include_ens) if _extras_include_ens else []
        _data_loader_ens = DataLoader()

        from concurrent.futures import ThreadPoolExecutor, as_completed
        load_workers = min(len(symbols), 8)

        def _load_symbol_ens(sym):
            try:
                md = load_local_market_data(sym, timeframe, start_date, end_date, exchange, trade_type)
                if md is not None and not md.empty:
                    if _extras_include_ens:
                        md = _data_loader_ens.join_extras(md, sym, interval=timeframe, include=_extras_include_ens)
                    return (sym, md)
            except Exception as e:
                print(f"加载 {sym} 失败: {e}")
            return (sym, None)

        with ThreadPoolExecutor(max_workers=load_workers) as loader:
            load_futures = {loader.submit(_load_symbol_ens, s): s for s in symbols}
            for future in as_completed(load_futures):
                sym = load_futures[future]
                try:
                    sym_key, md = future.result()
                    if md is not None and not md.empty:
                        data_dict[sym_key] = md
                except Exception as e:
                    print(f"加载 {sym} 失败: {e}")

        if len(data_dict) < 2:
            return jsonify({
                'success': False,
                'message': f'加载到有效市场数据的币种仅 {len(data_dict)} 个，截面组合回测至少需要 2 个'
            })

        prep_evaluator = CrossSectionalEvaluator(
            n_groups=n_groups,
            normalize_method='rank',
            predict_step=predict_step,
            sample_step=sample_step,
            base_timeframe=timeframe,
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
        )

        # prepare_cross_sectional_data 已在内部对因子施加 trade_shift（信号延迟），
        # 返回的 factor_value 是已延迟的值，后续无需再 shift。
        all_factors = {}
        for factor_id in factor_ids:
            if cancel_event.is_set():
                return jsonify({'success': False, 'message': '组合回测已取消', 'cancelled': True})
            try:
                cs_df = prep_evaluator.prepare_cross_sectional_data(
                    data_dict, factor_id, engine)
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

        # 仅对线性等权 / IC 加权方法做"按 IC 符号反转"的预处理。
        # ml_weight / max_icir_weight 让优化器自行学习正负权重，避免重复反转或方向冲突。
        method_uses_reverse = ensemble_method in (
            'equal_weight', 'ic_weight') and auto_reverse
        if method_uses_reverse:
            for symbol, factors_df in all_factors_dfs.items():
                for fid in factors_df.columns:
                    try:
                        ic_value = float(factor_ic_dict.get(fid, 0) or 0)
                    except (TypeError, ValueError):
                        ic_value = 0.0
                    if ic_value < 0:
                        factors_df[fid] = -factors_df[fid]

        ensemble_factors = {}
        reversed_factors = []
        max_icir_weights_per_symbol = {}

        # 保存原始 IC 字典，用于后续判断反转方向（ic_source='backtest' 模式下会更新 factor_ic_dict）
        original_ic_dict = dict(factor_ic_dict)

        if ic_source == 'backtest' and ensemble_method == 'ic_weight':
            recalc_ic_dict = {}
            for fid in factor_ids:
                if cancel_event.is_set():
                    return jsonify({'success': False, 'message': '组合回测已取消', 'cancelled': True})
                all_fv = []
                all_rt = []
                for sym, fdf in all_factors_dfs.items():
                    if fid in fdf.columns:
                        md = data_dict[sym].copy().sort_index()
                        md['future_returns'] = md['close'].pct_change(
                            periods=predict_step).shift(-predict_step)
                        common_idx = fdf[fid].index.intersection(md.index)
                        if len(common_idx) > 0:
                            # prepare_cross_sectional_data 已完成 trade_shift 对齐，
                            # 这里直接复用同口径因子值，避免重算 IC 时额外滞后一期。
                            fv = fdf.loc[common_idx, fid]
                            rt = md.loc[common_idx, 'future_returns']
                            mask = fv.notna() & rt.notna()
                            if mask.sum() > 10:
                                all_fv.extend(fv[mask].values.tolist())
                                all_rt.extend(rt[mask].values.tolist())
                if len(all_fv) >= 20:
                    try:
                        ic_val = float(
                            pd.Series(all_fv).corr(pd.Series(all_rt)))
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
            # 权重始终使用 |IC| 表示“强度”；方向是否反转仅由 auto_reverse 控制。
            # 否则 auto_reverse=False 时负权重仍会隐式把负 IC 因子反向使用。
            weight_ic = abs(ic_val)
            ic_weight_map[fid] = weight_ic
            abs_ic_sum += abs(weight_ic)
        if abs_ic_sum > 0:
            ic_weight_map = {k: v / abs_ic_sum for k,
                             v in ic_weight_map.items()}
        else:
            eq_w = 1.0 / max(len(factor_ids), 1)
            ic_weight_map = {fid: eq_w for fid in factor_ids}

        for symbol, factors_df in all_factors_dfs.items():
            try:
                market_data = data_dict[symbol]
                if ensemble_method == 'equal_weight':
                    ensemble_factor = factors_df.mean(axis=1)
                elif ensemble_method == 'ic_weight':
                    active_cols = [
                        c for c in factors_df.columns if c in ic_weight_map]
                    if not active_cols:
                        ensemble_factor = factors_df.mean(axis=1)
                    else:
                        weights = np.array([ic_weight_map[c]
                                           for c in active_cols], dtype=float)
                        weight_sum = float(weights.sum())
                        if weight_sum <= 0:
                            ensemble_factor = factors_df[active_cols].mean(
                                axis=1)
                        else:
                            weights = weights / weight_sum
                            ensemble_factor = factors_df[active_cols].mul(
                                weights, axis=1).sum(axis=1)
                elif ensemble_method == 'ml_weight':
                    md = market_data.copy().sort_index()
                    ml_returns = md['close'].pct_change(
                        periods=predict_step).shift(-predict_step)
                    optimizer = FactorOptimizer()
                    optimizer.set_data(None, ml_returns)
                    ensemble_factor = optimizer._create_ml_weighted_factor_walk_forward(
                        factors_df)
                elif ensemble_method in ('max_icir_weight', 'max_icir'):
                    md = market_data.copy().sort_index()
                    icir_returns = md['close'].pct_change(
                        periods=predict_step).shift(-predict_step)
                    optimizer = FactorOptimizer()
                    optimizer.set_data(None, icir_returns)
                    ensemble_factor, sym_weights = optimizer._create_max_icir_weighted_factor(
                        factors_df,
                        return_weights=True
                    )
                    max_icir_weights_per_symbol[symbol] = sym_weights
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
                    md['close'].pct_change(
                        periods=predict_step).shift(-predict_step)
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
        # 注意：sample_step 已在 prep_evaluator.prepare_cross_sectional_data 中按时间稀疏，
        # 这里不再重复二次过滤，避免间隔翻倍。

        cs_evaluator = CrossSectionalEvaluator(
            n_groups=effective_n_groups,
            normalize_method='rank',
            predict_step=predict_step,
            # cs_data 已由 prepare_cross_sectional_data 按 sample_step 做过稀疏，
            # 这里传入 sample_step=1 避免 calculate_cross_sectional_ic/returns 内部再次过滤
            sample_step=1,
            min_coverage=min_coverage,
            min_valid_count=min_valid_count,
            min_group_size=min_group_size,
            treat_zero_as_invalid=treat_zero_as_invalid,
            transaction_cost=transaction_cost,
            enable_data_cleaning=False,
        )
        ic_results = cs_evaluator.calculate_cross_sectional_ic(cs_data)
        returns_results = cs_evaluator.calculate_cross_sectional_returns(
            cs_data, timeframe=timeframe)

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
            # 使用原始 IC 字典判断反转方向，避免 ic_source='backtest' 更新后导致列表为空
            reversed_factors = [fid for fid in factor_ids if float(
                original_ic_dict.get(fid, 0) or 0) < 0]

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

        factor_weight_map = None
        if ensemble_method == 'equal_weight':
            eq_w = 1.0 / max(len(factor_ids), 1)
            factor_weight_map = {fid: eq_w for fid in factor_ids}
        elif ensemble_method == 'ic_weight':
            factor_weight_map = ic_weight_map
        elif ensemble_method in ('max_icir_weight', 'max_icir') and max_icir_weights_per_symbol:
            # 多 symbol 时，max_icir_weight 是按 symbol 分别求解的。
            # 这里对各 symbol 的权重做算术平均后再 L1 归一化，作为可导出的全局静态权重。
            agg = {fid: 0.0 for fid in factor_ids}
            counts = {fid: 0 for fid in factor_ids}
            for _sym, w_map in max_icir_weights_per_symbol.items():
                for fid, w in (w_map or {}).items():
                    if fid in agg:
                        try:
                            agg[fid] += float(w)
                            counts[fid] += 1
                        except (TypeError, ValueError):
                            continue
            avg_weights = {
                fid: (agg[fid] / counts[fid]) if counts[fid] > 0 else 0.0
                for fid in factor_ids
            }
            l1 = sum(abs(v) for v in avg_weights.values())
            if l1 > 0:
                factor_weight_map = {
                    fid: v / l1 for fid, v in avg_weights.items()}
            else:
                eq_w = 1.0 / max(len(factor_ids), 1)
                factor_weight_map = {fid: eq_w for fid in factor_ids}

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
            'factor_weight_map': _clean_nan(factor_weight_map),
            'reversed_factors': reversed_factors,
            'predict_step': predict_step,
            'sample_step': sample_step,
            'n_groups_effective': effective_n_groups,
            'request_id': request_id,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'组合回测失败: {str(e)}'
        })
    finally:
        try:
            _discard_cancel_event(request_id)
        except Exception:
            pass


@bp.route('/factor_correlation', methods=['POST'])
def factor_correlation():
    """
    计算选中因子的截面相关性矩阵
    - 对每个币种分别计算因子值，然后合并为截面数据
    - 计算因子间的 Pearson 和 Spearman 相关系数
    - 返回相关性矩阵和高相关因子对
    """
    request_id = _LEGACY_CANCEL_ID
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
        # 与截面评估保持一致的因子计算配置（解决因子值口径不同导致的相关性失真）
        factor_timeframe = payload.get('factor_timeframe', timeframe)
        factor_bar_mode = str(payload.get(
            'factor_bar_mode', 'completed')).lower()
        max_lookback = payload.get('max_lookback', 200)
        predict_step = payload.get('predict_step', 1)
        sample_step = payload.get('sample_step', 1)
        min_coverage = payload.get('min_coverage', 0.3)
        min_valid_count = payload.get('min_valid_count', 30)
        min_group_size = payload.get('min_group_size', 5)
        treat_zero_as_invalid = payload.get('treat_zero_as_invalid', True)
        enable_data_cleaning = payload.get('enable_data_cleaning', False)
        remove_zero_volume = payload.get('remove_zero_volume', True)
        liquidity_filter_ratio = payload.get('liquidity_filter_ratio', 0.5)
        request_id = str(payload.get('request_id')
                         or '').strip() or _LEGACY_CANCEL_ID

        try:
            corr_threshold = float(corr_threshold)
        except (TypeError, ValueError):
            corr_threshold = 0.7
        if not (0 < corr_threshold <= 1):
            corr_threshold = 0.7

        try:
            max_lookback = int(max_lookback)
        except (TypeError, ValueError):
            max_lookback = 200
        if max_lookback < 1:
            max_lookback = 200
        if max_lookback > 5000:
            return jsonify({
                'success': False,
                'message': '参数 max_lookback 不能大于 5000（建议 100-500）'
            }), 400

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

        cancel_event = _get_cancel_event(request_id)
        cancel_event.clear()

        engine = get_global_engine()
        data_dict = {}

        _VALID_EXTRA_KINDS = frozenset({'metrics', 'funding', 'basis', 'mark', 'index'})
        _LEGACY_DERIVATIVES_EXPAND = frozenset({'metrics', 'basis'})
        _extras_include = set()
        for _fid in factor_ids:
            try:
                _fdef = _get_repo().load_definition(_fid)
                if not _fdef or not _fdef.metadata:
                    continue
                for _req in (_fdef.metadata.get('requires_extras') or []):
                    if _req == 'derivatives':
                        _extras_include.update(_LEGACY_DERIVATIVES_EXPAND)
                    elif _req in _VALID_EXTRA_KINDS:
                        _extras_include.add(_req)
            except Exception:
                pass
        _extras_include = sorted(_extras_include) if _extras_include else []
        _data_loader = DataLoader()

        from concurrent.futures import ThreadPoolExecutor, as_completed
        load_workers = min(len(symbols), 8)

        def _load_symbol_corr(sym):
            try:
                md = load_local_market_data(sym, timeframe, start_date, end_date, exchange, trade_type)
                if md is not None and not md.empty:
                    if _extras_include:
                        md = _data_loader.join_extras(md, sym, interval=timeframe, include=_extras_include)
                    return (sym, md)
            except Exception:
                pass
            return (sym, None)

        with ThreadPoolExecutor(max_workers=load_workers) as loader:
            load_futures = {loader.submit(_load_symbol_corr, s): s for s in symbols}
            for future in as_completed(load_futures):
                sym = load_futures[future]
                try:
                    sym_key, md = future.result()
                    if md is not None and not md.empty:
                        data_dict[sym_key] = md
                except Exception:
                    continue

        if len(data_dict) < 2:
            return jsonify({
                'success': False,
                'message': '成功加载市场数据的币种不足2个'
            })

        # 使用与截面评估相同的预处理流水线（trade_shift / 数据清洗 / 异常值处理 / sample_step 等），
        # 保证相关性矩阵反映的是"评估时实际使用的因子值"。
        prep_evaluator = CrossSectionalEvaluator(
            n_groups=5,
            normalize_method='rank',
            predict_step=predict_step,
            sample_step=sample_step,
            base_timeframe=timeframe,
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
        )

        # rows: list of DataFrame[date, symbol, factor_id -> value]
        long_rows = []
        used_symbols = set()
        for factor_id in factor_ids:
            if cancel_event.is_set():
                return jsonify({'success': False, 'message': '相关性计算已取消', 'cancelled': True})
            try:
                cs_df = prep_evaluator.prepare_cross_sectional_data(
                    data_dict, factor_id, engine)
            except Exception as ex:
                print(f"[corr] 因子 {factor_id} 截面数据准备失败: {ex}")
                continue
            if cs_df is None or cs_df.empty:
                continue
            sub = cs_df[['date', 'symbol', 'factor_value']].copy()
            if auto_reverse:
                try:
                    ic_value = float(factor_ic_dict.get(factor_id, 0) or 0)
                except (TypeError, ValueError):
                    ic_value = 0.0
                if ic_value < 0:
                    sub['factor_value'] = -sub['factor_value']
            sub = sub.rename(columns={'factor_value': factor_id})
            used_symbols.update(sub['symbol'].unique().tolist())
            long_rows.append(sub)

        if len(long_rows) < 2:
            return jsonify({
                'success': False,
                'message': '成功计算因子值的因子不足 2 个，无法计算相关性'
            })

        # 在 (date, symbol) 维度合并所有因子，构成宽表
        merged = long_rows[0]
        for nxt in long_rows[1:]:
            merged = merged.merge(nxt, on=['date', 'symbol'], how='outer')

        factor_cols = [fid for fid in factor_ids if fid in merged.columns]
        if len(factor_cols) < 2:
            return jsonify({
                'success': False,
                'message': '有效因子不足2个'
            })

        # 使用 pairwise 模式：不做全行 dropna，让 pandas 按因子对删除缺失值，
        # 避免某个因子缺失值过多时大量丢失其他因子的有效行。
        factor_matrix_full = merged[factor_cols]

        # 最小样本检查：至少需要已有 10 行痞不为 NaN 的数据
        min_required = max(int(min_valid_count), 10)
        non_null_counts = factor_matrix_full.count()
        if non_null_counts.min() < min_required:
            return jsonify({
                'success': False,
                'message': f'有因子有效数据点不足（最少 {non_null_counts.min()} ＜ {min_required}），请检查因子或数据覆盖率'
            })

        n_data_points = int(factor_matrix_full.dropna().shape[0])  # 仅用于展示

        pearson_corr = factor_matrix_full.corr(
            method='pearson', min_periods=min_required)
        spearman_corr = factor_matrix_full.corr(
            method='spearman', min_periods=min_required)

        pearson_matrix = []
        spearman_matrix = []
        for i, fid_i in enumerate(factor_cols):
            pearson_row = []
            spearman_row = []
            for j, fid_j in enumerate(factor_cols):
                p_val = pearson_corr.iloc[i, j]
                s_val = spearman_corr.iloc[i, j]
                pearson_row.append(
                    float(p_val) if np.isfinite(p_val) else None)
                spearman_row.append(
                    float(s_val) if np.isfinite(s_val) else None)
            pearson_matrix.append(pearson_row)
            spearman_matrix.append(spearman_row)

        # 高相关识别：取 max(|pearson|, |spearman|)，更全面捕捉线性 + 单调相关
        high_corr_pairs = []
        for i in range(len(factor_cols)):
            for j in range(i + 1, len(factor_cols)):
                p_val = pearson_matrix[i][j]
                s_val = spearman_matrix[i][j]
                p_abs = abs(p_val) if p_val is not None else 0.0
                s_abs = abs(s_val) if s_val is not None else 0.0
                composite = max(p_abs, s_abs)
                if composite >= corr_threshold:
                    high_corr_pairs.append({
                        'factor_1': factor_cols[i],
                        'factor_2': factor_cols[j],
                        'pearson': p_val,
                        'spearman': s_val,
                        'max_abs': composite,
                    })

        high_corr_pairs.sort(key=lambda x: x['max_abs'], reverse=True)

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
            'n_symbols': len(used_symbols),
            'n_data_points': n_data_points,
            'pearson_matrix': pearson_matrix,
            'spearman_matrix': spearman_matrix,
            'high_corr_pairs': high_corr_pairs,
            'corr_threshold': corr_threshold,
            'request_id': request_id,
        }))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'相关性计算失败: {str(e)}'
        })
    finally:
        try:
            _discard_cancel_event(request_id)
        except Exception:
            pass


@bp.route('/factor_ic_correlation', methods=['POST'])
def factor_ic_correlation():
    """
    计算因子间 Rank IC 时序相关性（方案 B+）

    不同于 /factor_correlation 以"因子值"为维度的截面相关性，这里以"每个时间点
    的 Rank IC"为维度，衡量两个因子在"对未来收益率的截面预测"这件事上是否
    同步波动。Rank IC 相关高 => 两个因子贡献的 alpha 源几乎同质。
    """
    request_id = _LEGACY_CANCEL_ID
    try:
        import numpy as np

        payload = request.get_json() or {}
        factor_ids = payload.get('factor_ids', [])
        symbols = payload.get('symbols', [])
        timeframe = payload.get('timeframe', '1h')
        start_date = payload.get('start_date')
        end_date = payload.get('end_date')
        exchange = payload.get('exchange', 'binance')
        trade_type = payload.get('trade_type', 'futures')
        corr_threshold = payload.get('corr_threshold', 0.7)
        factor_timeframe = payload.get('factor_timeframe', timeframe)
        factor_bar_mode = str(payload.get(
            'factor_bar_mode', 'completed')).lower()
        max_lookback = payload.get('max_lookback', 200)
        predict_step = payload.get('predict_step', 1)
        sample_step = payload.get('sample_step', 1)
        min_coverage = payload.get('min_coverage', 0.3)
        min_valid_count = payload.get('min_valid_count', 30)
        min_group_size = payload.get('min_group_size', 5)
        treat_zero_as_invalid = payload.get('treat_zero_as_invalid', True)
        n_ic_segments = payload.get('n_ic_segments', 4)
        enable_data_cleaning = payload.get('enable_data_cleaning', False)
        remove_zero_volume = payload.get('remove_zero_volume', True)
        liquidity_filter_ratio = payload.get('liquidity_filter_ratio', 0.5)
        request_id = str(payload.get('request_id')
                         or '').strip() or _LEGACY_CANCEL_ID

        try:
            corr_threshold = float(corr_threshold)
        except (TypeError, ValueError):
            corr_threshold = 0.7
        if not (0 < corr_threshold <= 1):
            corr_threshold = 0.7

        try:
            max_lookback = int(max_lookback)
        except (TypeError, ValueError):
            max_lookback = 200
        if max_lookback < 1:
            max_lookback = 200
        if max_lookback > 5000:
            return jsonify({
                'success': False,
                'message': '参数 max_lookback 不能大于 5000（建议 100-500）'
            }), 400

        try:
            n_ic_segments = int(n_ic_segments)
        except (TypeError, ValueError):
            n_ic_segments = 4

        if not factor_ids or len(factor_ids) < 2:
            return jsonify({
                'success': False,
                'message': '至少需要选择 2 个因子才能计算 IC 时序相关性'
            }), 400

        if not symbols or len(symbols) < 2:
            return jsonify({
                'success': False,
                'message': '至少需要 2 个币种来计算截面 IC'
            }), 400

        if not start_date or not end_date:
            return jsonify({
                'success': False,
                'message': '请选择时间范围'
            })

        cancel_event = _get_cancel_event(request_id)
        cancel_event.clear()

        engine = get_global_engine()
        data_dict = {}

        _VALID_EXTRA_KINDS_IC = frozenset({'metrics', 'funding', 'basis', 'mark', 'index'})
        _LEGACY_DERIVATIVES_EXPAND_IC = frozenset({'metrics', 'basis'})
        _extras_include_ic = set()
        for _fid in factor_ids:
            try:
                _fdef = _get_repo().load_definition(_fid)
                if not _fdef or not _fdef.metadata:
                    continue
                for _req in (_fdef.metadata.get('requires_extras') or []):
                    if _req == 'derivatives':
                        _extras_include_ic.update(_LEGACY_DERIVATIVES_EXPAND_IC)
                    elif _req in _VALID_EXTRA_KINDS_IC:
                        _extras_include_ic.add(_req)
            except Exception:
                pass
        _extras_include_ic = sorted(_extras_include_ic) if _extras_include_ic else []
        _data_loader_ic = DataLoader()

        from concurrent.futures import ThreadPoolExecutor, as_completed
        load_workers = min(len(symbols), 8)

        def _load_symbol_ic_corr(sym):
            try:
                md = load_local_market_data(sym, timeframe, start_date, end_date, exchange, trade_type)
                if md is not None and not md.empty:
                    if _extras_include_ic:
                        md = _data_loader_ic.join_extras(md, sym, interval=timeframe, include=_extras_include_ic)
                    return (sym, md)
            except Exception:
                pass
            return (sym, None)

        with ThreadPoolExecutor(max_workers=load_workers) as loader:
            load_futures = {loader.submit(_load_symbol_ic_corr, s): s for s in symbols}
            for future in as_completed(load_futures):
                sym = load_futures[future]
                try:
                    sym_key, md = future.result()
                    if md is not None and not md.empty:
                        data_dict[sym_key] = md
                except Exception:
                    continue

        if len(data_dict) < 2:
            return jsonify({
                'success': False,
                'message': '成功加载市场数据的币种不足 2 个'
            })

        prep_evaluator = CrossSectionalEvaluator(
            n_groups=5,
            normalize_method='rank',
            predict_step=predict_step,
            sample_step=sample_step,
            base_timeframe=timeframe,
            factor_timeframe=factor_timeframe,
            factor_bar_mode=factor_bar_mode,
            max_lookback=max_lookback,
            min_coverage=min_coverage,
            min_valid_count=min_valid_count,
            min_group_size=min_group_size,
            treat_zero_as_invalid=treat_zero_as_invalid,
            n_ic_segments=n_ic_segments,
            enable_data_cleaning=enable_data_cleaning,
            remove_zero_volume=remove_zero_volume,
            liquidity_filter_ratio=liquidity_filter_ratio,
        )

        # 收集每个因子的 Rank IC 时序 -> Series(index=date, values=rank_ic)
        rank_ic_series_map = {}
        usable_factor_ids = []
        failed_factor_ids = []
        for factor_id in factor_ids:
            if cancel_event.is_set():
                return jsonify({
                    'success': False,
                    'message': 'IC 时序相关性计算已取消',
                    'cancelled': True,
                })
            try:
                cs_df = prep_evaluator.prepare_cross_sectional_data(
                    data_dict, factor_id, engine)
                if cs_df is None or cs_df.empty:
                    failed_factor_ids.append(factor_id)
                    continue
                ic_res = prep_evaluator.calculate_cross_sectional_ic(cs_df)
                series_list = ic_res.get('rank_ic_series') or []
                if not series_list:
                    failed_factor_ids.append(factor_id)
                    continue
                dates = []
                values = []
                for item in series_list:
                    try:
                        d = pd.Timestamp(item.get('date'))
                    except Exception:
                        continue
                    v = item.get('rank_ic')
                    if v is None or not np.isfinite(v):
                        continue
                    dates.append(d)
                    values.append(float(v))
                if len(values) < max(int(min_valid_count // 2), 10):
                    failed_factor_ids.append(factor_id)
                    continue
                s = pd.Series(values, index=pd.DatetimeIndex(dates))
                s = s[~s.index.duplicated(keep='last')].sort_index()
                rank_ic_series_map[factor_id] = s
                usable_factor_ids.append(factor_id)
            except Exception as ex:
                print(f"[ic_corr] 因子 {factor_id} Rank IC 时序准备失败: {ex}")
                failed_factor_ids.append(factor_id)
                continue

        if len(usable_factor_ids) < 2:
            return jsonify({
                'success': False,
                'message': (
                    f'成功计算 Rank IC 时序的因子不足 2 个（可用 {len(usable_factor_ids)}，'
                    f'失败 {len(failed_factor_ids)}）'
                ),
                'failed_factor_ids': failed_factor_ids,
            })

        # 对齐所有因子的 Rank IC 时间序列为宽表（index=date, columns=factor_id）
        rank_ic_wide = pd.concat(
            [rank_ic_series_map[fid].rename(fid) for fid in usable_factor_ids],
            axis=1,
            join='outer',
        ).sort_index()

        # 只保留全部因子都有 IC 的时间点（确保相关性口径一致）
        complete = rank_ic_wide.dropna(how='any')
        # 若共同有效时点过少（极端工况），退化为成对删除
        if len(complete) < 30 and len(rank_ic_wide) >= 30:
            pearson_corr = rank_ic_wide.corr(method='pearson', min_periods=30)
            spearman_corr = rank_ic_wide.corr(
                method='spearman', min_periods=30)
            n_aligned = int(len(rank_ic_wide))
            alignment_mode = 'pairwise'
        else:
            pearson_corr = complete.corr(method='pearson')
            spearman_corr = complete.corr(method='spearman')
            n_aligned = int(len(complete))
            alignment_mode = 'intersection'

        factor_cols = list(usable_factor_ids)
        pearson_matrix = []
        spearman_matrix = []
        for i, fid_i in enumerate(factor_cols):
            prow = []
            srow = []
            for j, fid_j in enumerate(factor_cols):
                p_val = pearson_corr.loc[fid_i,
                                         fid_j] if fid_i in pearson_corr.index and fid_j in pearson_corr.columns else np.nan
                s_val = spearman_corr.loc[fid_i,
                                          fid_j] if fid_i in spearman_corr.index and fid_j in spearman_corr.columns else np.nan
                prow.append(float(p_val) if np.isfinite(p_val) else None)
                srow.append(float(s_val) if np.isfinite(s_val) else None)
            pearson_matrix.append(prow)
            spearman_matrix.append(srow)

        high_corr_pairs = []
        for i in range(len(factor_cols)):
            for j in range(i + 1, len(factor_cols)):
                p_val = pearson_matrix[i][j]
                s_val = spearman_matrix[i][j]
                p_abs = abs(p_val) if p_val is not None else 0.0
                s_abs = abs(s_val) if s_val is not None else 0.0
                composite = max(p_abs, s_abs)
                if composite >= corr_threshold:
                    high_corr_pairs.append({
                        'factor_1': factor_cols[i],
                        'factor_2': factor_cols[j],
                        'pearson': p_val,
                        'spearman': s_val,
                        'max_abs': composite,
                    })
        high_corr_pairs.sort(key=lambda x: x['max_abs'], reverse=True)

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
            'mode': 'rank_ic_series',
            'factor_ids': factor_cols,
            'failed_factor_ids': failed_factor_ids,
            'n_aligned_periods': n_aligned,
            'alignment_mode': alignment_mode,
            'pearson_matrix': pearson_matrix,
            'spearman_matrix': spearman_matrix,
            'high_corr_pairs': high_corr_pairs,
            'corr_threshold': corr_threshold,
            'request_id': request_id,
        }))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'IC 时序相关性计算失败: {str(e)}'
        })
    finally:
        try:
            _discard_cancel_event(request_id)
        except Exception:
            pass


# ============================================================================
# 方法对比（walk-forward 对照实验）
# ----------------------------------------------------------------------------
# 在完全相同的 rolling (train, oos) 切分上，对三种组合权重估计方法：
#   - equal_weight
#   - ic_weight           （训练段平均 Rank IC，L1 归一化）
#   - max_icir_weight     （训练段 μ=mean IC, Σ=cov(IC)；w=project_l1(Σ⁻¹ μ)）
# 进行严格样本外比较，返回每种方法的 OOS 累计收益曲线、聚合指标与两两配对 t 检验。
# ============================================================================

def _mc_compute_daily_ic_matrix(ranks_by_factor, returns_rank):
    """
    vectorized 地计算每日 Rank IC 矩阵（T 行 × N 列）。

    参数：
        ranks_by_factor: {factor_id: DataFrame(T×S)}，每个因子的截面秩（已对日期做 rank(axis=1)）
        returns_rank: DataFrame(T×S)，未来收益的截面秩
    返回：
        DataFrame(index=date, columns=factor_id)：每日每因子的 Rank IC
    """
    ic_cols = {}
    for fid, rdf in ranks_by_factor.items():
        # pandas corrwith(..., axis=1) 会按行（每个日期）计算两张表的相关性，
        # 两张表都已按行 rank → pearson(rank) = spearman = Rank IC
        try:
            ic_cols[fid] = rdf.corrwith(returns_rank, axis=1, method='pearson')
        except Exception:
            ic_cols[fid] = pd.Series(index=rdf.index, dtype=float)
    return pd.DataFrame(ic_cols)


def _mc_project_l1(w, enforce_non_negative=False):
    """把一个权重向量归一到 ||w||_1 = 1。全 0 时退化为等权。"""
    import numpy as np
    w = np.asarray(w, dtype=float).ravel()
    if enforce_non_negative:
        w = np.clip(w, a_min=0.0, a_max=None)
    s = np.sum(np.abs(w))
    if not np.isfinite(s) or s <= 0:
        n = len(w) if len(w) > 0 else 1
        return np.ones(n) / n
    return w / s


def _mc_compute_weights(ic_train_matrix, method, ridge=1e-6, use_ledoit_wolf=True):
    """
    从训练段的 T×N IC 矩阵估一组全局静态权重。

    - equal_weight:     w_i = 1/N
    - ic_weight:        w_i ∝ mean(IC_i)，L1 归一化
    - max_icir_weight:  w ∝ Σ⁻¹ μ，L1 归一化
    """
    import numpy as np
    factors = list(ic_train_matrix.columns)
    n = len(factors)
    if n == 0:
        return {}, '空因子'

    if method == 'equal_weight':
        w = np.ones(n) / n
        return {f: float(w[i]) for i, f in enumerate(factors)}, None

    mat = ic_train_matrix.to_numpy(dtype=float)
    # 丢掉 IC 全为 NaN 的行
    mat = mat[~np.all(~np.isfinite(mat), axis=1)]
    if mat.shape[0] < max(5, n):
        # 训练段 IC 行数太少：退化等权
        w = np.ones(n) / n
        return {f: float(w[i]) for i, f in enumerate(factors)}, '训练段 IC 行数不足，退化等权'

    # 均值：用 nanmean；NaN 权重按 0 处理
    mu = np.nanmean(mat, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)

    if method == 'ic_weight':
        w = _mc_project_l1(mu, enforce_non_negative=False)
        return {f: float(w[i]) for i, f in enumerate(factors)}, None

    if method == 'max_icir_weight':
        # 协方差：用有限行构造
        finite_mat = np.where(np.isfinite(mat), mat, 0.0)
        cov = None
        if use_ledoit_wolf and finite_mat.shape[0] >= max(2, n + 1):
            try:
                from sklearn.covariance import LedoitWolf
                cov = LedoitWolf().fit(finite_mat).covariance_
            except Exception:
                cov = None
        if cov is None:
            cov = np.cov(finite_mat, rowvar=False)
        cov = np.asarray(cov, dtype=float)
        if cov.ndim != 2 or cov.shape[0] != n:
            w = _mc_project_l1(mu)
            return {f: float(w[i]) for i, f in enumerate(factors)}, 'cov 估计失败，回退 ic_weight'
        cov = cov + np.eye(n, dtype=float) * float(ridge)
        try:
            raw_w = np.linalg.solve(cov, mu)
        except np.linalg.LinAlgError:
            raw_w, *_ = np.linalg.lstsq(cov, mu, rcond=None)
        w = _mc_project_l1(raw_w, enforce_non_negative=False)
        return {f: float(w[i]) for i, f in enumerate(factors)}, None

    # 未知方法：等权兜底
    w = np.ones(n) / n
    return {f: float(w[i]) for i, f in enumerate(factors)}, f'未知方法 {method}，已退化等权'


def _mc_apply_and_measure(ranks_by_factor_oos, returns_pivot_oos, returns_rank_oos,
                          weights, n_groups, transaction_cost):
    """
    给定一组权重，在 OOS 段 ranks / returns 面板上跑组合因子，返回：
      - 每日 Rank IC Series
      - 每日多空收益 Series（未扣费）
      - 每日换手率 Series
      - 每日多空收益 Series（扣费后）
    """
    import numpy as np
    factors = list(weights.keys())
    if not factors:
        return None

    # 合成因子的截面"得分"= Σ w_i * rank_i（秩 in [1, n_symbols]）
    composite_rank = None
    for f in factors:
        w = float(weights.get(f, 0.0))
        if w == 0:
            continue
        rdf = ranks_by_factor_oos.get(f)
        if rdf is None:
            continue
        contrib = rdf * w
        composite_rank = contrib if composite_rank is None else composite_rank.add(
            contrib, fill_value=0.0)

    if composite_rank is None or composite_rank.empty:
        return None

    # 每日 Rank IC = pearson(composite_rank_row, returns_rank_row)
    # 二者均已是同量纲的秩
    daily_ic = composite_rank.corrwith(
        returns_rank_oos, axis=1, method='pearson')

    # 按每日 composite_rank 排序分组 → 多空收益
    # 每行保留有效列（非 NaN），按分位分组
    def _long_short_return(row_composite, row_returns):
        mask = row_composite.notna() & row_returns.notna()
        if mask.sum() < max(n_groups * 2, 4):
            return np.nan, np.nan  # long_short, long_weights (None)
        comp = row_composite[mask]
        rets = row_returns[mask]
        n = len(comp)
        n_per_group = max(1, n // n_groups)
        order = comp.sort_values(ascending=False)
        top_syms = order.iloc[:n_per_group].index
        bot_syms = order.iloc[-n_per_group:].index
        top_ret = rets.loc[top_syms].mean()
        bot_ret = rets.loc[bot_syms].mean()
        return top_ret - bot_ret, (top_syms, bot_syms)

    # 逐日计算并累积，同时记录上一期多空持仓用于 turnover
    dates = composite_rank.index
    ls_returns = []
    turnover_list = []
    prev_weights = None  # dict {symbol: weight}，多头正、空头负；仓位各自 1/n_per_group
    for d in dates:
        comp_row = composite_rank.loc[d]
        ret_row = returns_pivot_oos.loc[d] if d in returns_pivot_oos.index else None
        if ret_row is None:
            ls_returns.append(np.nan)
            turnover_list.append(np.nan)
            continue
        ls, syms = _long_short_return(comp_row, ret_row)
        ls_returns.append(ls)
        if isinstance(syms, tuple):
            top_syms, bot_syms = syms
            n_long = max(len(top_syms), 1)
            n_short = max(len(bot_syms), 1)
            cur_weights = {}
            for s in top_syms:
                cur_weights[s] = cur_weights.get(s, 0.0) + 1.0 / n_long
            for s in bot_syms:
                cur_weights[s] = cur_weights.get(s, 0.0) - 1.0 / n_short
            if prev_weights is None:
                turnover_list.append(np.nan)
            else:
                all_syms = set(cur_weights.keys()) | set(prev_weights.keys())
                diff = sum(abs(cur_weights.get(s, 0.0) -
                           prev_weights.get(s, 0.0)) for s in all_syms)
                # 0.5 * L1 差：得到"组合变动比例"，0=不换仓，1=完全换仓
                turnover_list.append(0.5 * diff)
            prev_weights = cur_weights
        else:
            turnover_list.append(np.nan)

    ls_series = pd.Series(ls_returns, index=dates, dtype=float)
    turnover_series = pd.Series(turnover_list, index=dates, dtype=float)

    # 扣费：每期扣 turnover * transaction_cost（单边成本；double-sided 可按需调 ×2）
    cost_cost = turnover_series.fillna(0.0) * float(transaction_cost)
    ls_after_cost = ls_series - cost_cost

    return {
        'daily_ic': daily_ic,
        'long_short_return': ls_series,
        'long_short_return_after_cost': ls_after_cost,
        'turnover': turnover_series,
    }


@bp.route('/method_comparison_stream', methods=['POST'])
def method_comparison_stream():
    """
    方法对比：rolling walk-forward 对照实验（SSE 流式）

    请求 JSON：
    {
        factor_ids: [..],
        symbols: [..],
        timeframe: '1h',
        start_date, end_date: 'YYYY-MM-DD',
        predict_step, sample_step, n_groups,
        min_coverage, min_valid_count, min_group_size,
        treat_zero_as_invalid, factor_timeframe, factor_bar_mode, max_lookback,
        train_window_days, oos_window_days, step_days,   # walk-forward 切分
        transaction_cost,                                 # 交易成本，比如 0.001
        methods: ['equal_weight', 'ic_weight', 'max_icir_weight']   # 默认全选
        request_id
    }
    """
    import logging
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from flask import Response, stream_with_context
    import numpy as np

    logger = logging.getLogger(__name__)

    payload = request.get_json() or {}
    factor_ids = payload.get('factor_ids') or []
    symbols = payload.get('symbols') or []
    timeframe = payload.get('timeframe') or payload.get('base_timeframe', '1h')
    factor_timeframe = payload.get('factor_timeframe') or timeframe
    factor_bar_mode = str(payload.get('factor_bar_mode', 'completed')).lower()
    start_date = payload.get('start_date')
    end_date = payload.get('end_date')
    exchange = payload.get('exchange', 'binance')
    trade_type = payload.get('trade_type', 'futures')
    predict_step = int(payload.get('predict_step', 1) or 1)
    sample_step = int(payload.get('sample_step', 1) or 1)
    n_groups = int(payload.get('n_groups', 5) or 5)
    max_lookback = int(payload.get('max_lookback', 200) or 200)
    min_coverage = float(payload.get('min_coverage', 0.3) or 0.3)
    min_valid_count = int(payload.get('min_valid_count', 30) or 30)
    min_group_size = int(payload.get('min_group_size', 3) or 3)
    treat_zero_as_invalid = bool(payload.get('treat_zero_as_invalid', True))
    train_window_days = float(payload.get('train_window_days', 60) or 60)
    oos_window_days = float(payload.get('oos_window_days', 14) or 14)
    step_days = float(payload.get(
        'step_days', oos_window_days) or oos_window_days)
    transaction_cost = float(payload.get('transaction_cost', 0.001) or 0.001)
    methods = payload.get('methods') or [
        'equal_weight', 'ic_weight', 'max_icir_weight']
    methods = [m for m in methods if m in (
        'equal_weight', 'ic_weight', 'max_icir_weight')]
    if not methods:
        methods = ['equal_weight', 'ic_weight', 'max_icir_weight']
    request_id = str(payload.get('request_id')
                     or '').strip() or _LEGACY_CANCEL_ID

    # ── 基础参数校验 ──
    if not factor_ids or len(factor_ids) < 2:
        return jsonify({'success': False, 'message': '至少选择 2 个因子'}), 400
    if not symbols or len(symbols) < 2:
        return jsonify({'success': False, 'message': '至少选择 2 个币种'}), 400
    if not start_date or not end_date:
        return jsonify({'success': False, 'message': '请选择日期范围'}), 400
    if train_window_days < 1 or oos_window_days < 1 or step_days < 1:
        return jsonify({'success': False, 'message': 'train/oos/step 窗口必须 ≥ 1 天'}), 400

    cancel_event = _get_cancel_event(request_id)
    cancel_event.clear()

    def generate():
        import time as _time
        try:
            t_start = _time.time()
            total_factors = len(factor_ids)
            yield _sse_event('progress', {
                'phase': 'loading',
                'message': f'准备加载 {len(symbols)} 个币种的市场数据...',
                'completed': 0,
                'total': total_factors,
            })

            engine = get_global_engine()

            # Phase 1: 加载所有 symbol 的市场数据
            data_dict = {}
            load_workers = min(len(symbols), 8)
            with ThreadPoolExecutor(max_workers=load_workers) as pool:
                futures = {
                    pool.submit(load_local_market_data, s, timeframe, start_date, end_date, exchange, trade_type): s
                    for s in symbols
                }
                loaded = 0
                for fut in as_completed(futures):
                    if cancel_event.is_set():
                        break
                    loaded += 1
                    sym = futures[fut]
                    try:
                        md = fut.result()
                        if md is not None and not md.empty:
                            data_dict[sym] = md
                    except Exception as ex:
                        logger.warning(
                            f"[method_comparison] 加载 {sym} 失败: {ex}")
                    if loaded % 5 == 0 or loaded == len(symbols):
                        yield _sse_event('progress', {
                            'phase': 'loading',
                            'message': f'已加载 {loaded}/{len(symbols)}，有效 {len(data_dict)}',
                            'completed': 0,
                            'total': total_factors,
                        })

            if len(data_dict) < 2:
                yield _sse_event('error', {'message': f'有效币种不足 ({len(data_dict)} < 2)'})
                yield _sse_event('done', {'success': False})
                return

            # Phase 2: 对每个因子计算截面面板（date, symbol, factor_value, returns）
            cs_evaluator = CrossSectionalEvaluator(
                n_groups=n_groups,
                normalize_method='rank',
                predict_step=predict_step,
                sample_step=sample_step,
                base_timeframe=timeframe,
                factor_timeframe=factor_timeframe,
                factor_bar_mode=factor_bar_mode,
                max_lookback=max_lookback,
                min_coverage=min_coverage,
                min_valid_count=min_valid_count,
                min_group_size=min_group_size,
                treat_zero_as_invalid=treat_zero_as_invalid,
                enable_data_cleaning=False,
            )

            factor_panels = {}
            for i, fid in enumerate(factor_ids):
                if cancel_event.is_set():
                    yield _sse_event('error', {'message': '已取消'})
                    yield _sse_event('done', {'success': False})
                    return
                try:
                    cs_df = cs_evaluator.prepare_cross_sectional_data(
                        data_dict, fid, engine)
                    if cs_df is None or cs_df.empty:
                        logger.warning(f"[method_comparison] 因子 {fid} 无可用截面数据")
                        continue
                    factor_panels[fid] = cs_df[['date', 'symbol',
                                                'factor_value', 'returns']].copy()
                except Exception as ex:
                    logger.warning(
                        f"[method_comparison] 因子 {fid} 面板准备失败: {ex}")
                yield _sse_event('progress', {
                    'phase': 'preparing',
                    'message': f'面板准备中 {i+1}/{total_factors}: {fid}',
                    'completed': i + 1,
                    'total': total_factors,
                })

            usable_factors = [f for f in factor_ids if f in factor_panels]
            if len(usable_factors) < 2:
                yield _sse_event('error', {'message': f'成功准备面板的因子不足 2 ({len(usable_factors)})'})
                yield _sse_event('done', {'success': False})
                return

            # Phase 3: pivot 成 wide form：index=date, columns=symbol, values=factor_i / returns
            yield _sse_event('progress', {
                'phase': 'pivoting',
                'message': f'因子面板 pivot / 排名归一化（{len(usable_factors)} 个因子）...',
            })

            # returns pivot 取第一个因子的（它们的 returns 应该一致；以交集日期为准）
            first_panel = factor_panels[usable_factors[0]]
            returns_pivot = first_panel.pivot_table(
                index='date', columns='symbol', values='returns', aggfunc='last')
            returns_pivot = returns_pivot.sort_index()

            ranks_by_factor = {}
            for fid in usable_factors:
                p = factor_panels[fid]
                fv_pivot = p.pivot_table(
                    index='date', columns='symbol', values='factor_value', aggfunc='last')
                # 截面秩：对每一行（日期）rank，NaN 保持为 NaN
                ranks_by_factor[fid] = fv_pivot.rank(axis=1, method='average')

            # 对齐所有因子 + returns 的 (date × symbol) 索引：取公共日期 & 公共 symbol
            common_dates = returns_pivot.index
            for rdf in ranks_by_factor.values():
                common_dates = common_dates.intersection(rdf.index)
            common_dates = common_dates.sort_values()

            common_symbols = set(returns_pivot.columns)
            for rdf in ranks_by_factor.values():
                common_symbols &= set(rdf.columns)
            common_symbols = sorted(common_symbols)
            if len(common_dates) < 30 or len(common_symbols) < 3:
                yield _sse_event('error', {
                    'message': f'对齐后样本量过少：{len(common_dates)} 个截面 × {len(common_symbols)} 个 symbol'
                })
                yield _sse_event('done', {'success': False})
                return

            returns_pivot = returns_pivot.loc[common_dates, common_symbols]
            returns_rank = returns_pivot.rank(axis=1, method='average')
            for fid in usable_factors:
                ranks_by_factor[fid] = ranks_by_factor[fid].loc[common_dates,
                                                                common_symbols]

            # Phase 4: 计算全局每日 Rank IC 矩阵（供 train 段直接切片使用）
            yield _sse_event('progress', {
                'phase': 'ic_matrix',
                'message': f'计算每日 Rank IC 矩阵（{len(common_dates)} 天 × {len(usable_factors)} 因子）...',
            })
            full_ic_matrix = _mc_compute_daily_ic_matrix(
                ranks_by_factor, returns_rank)

            # Phase 5: 构造 walk-forward 切分
            date_index = pd.DatetimeIndex(common_dates)
            if len(date_index) < 2:
                yield _sse_event('error', {'message': '日期数不足'})
                yield _sse_event('done', {'success': False})
                return
            t0_date = date_index[0]
            t_end_date = date_index[-1]
            train_td = pd.Timedelta(days=float(train_window_days))
            oos_td = pd.Timedelta(days=float(oos_window_days))
            step_td = pd.Timedelta(days=float(step_days))

            splits = []
            train_start = t0_date
            while True:
                train_end = train_start + train_td
                oos_start = train_end
                oos_end = oos_start + oos_td
                if oos_end > t_end_date + pd.Timedelta(hours=23):
                    break
                splits.append({
                    'train_start': train_start,
                    'train_end': train_end,
                    'oos_start': oos_start,
                    'oos_end': oos_end,
                })
                train_start = train_start + step_td

            if len(splits) == 0:
                yield _sse_event('error', {
                    'message': (f'无法构造任何切分。数据范围 {t0_date.date()} ~ {t_end_date.date()}，'
                                f'train={train_window_days}d + oos={oos_window_days}d 超出范围')
                })
                yield _sse_event('done', {'success': False})
                return

            yield _sse_event('progress', {
                'phase': 'splitting',
                'message': f'生成 {len(splits)} 个切分，开始 walk-forward 评估...',
                'n_splits': len(splits),
            })

            # Phase 6: 遍历切分 × 方法，计算每段 OOS 指标
            # list of pd.Series，拼接后得到总 OOS LS 收益
            per_method_ls = {m: [] for m in methods}
            per_method_ls_after = {m: [] for m in methods}
            per_method_ic = {m: [] for m in methods}
            per_method_turnover = {m: [] for m in methods}
            per_method_weights_by_split = {m: [] for m in methods}

            for split_idx, sp in enumerate(splits):
                if cancel_event.is_set():
                    yield _sse_event('error', {'message': '已取消'})
                    yield _sse_event('done', {'success': False})
                    return

                # 训练段：取 full_ic_matrix 中该区间
                train_mask = (full_ic_matrix.index >= sp['train_start']) & (
                    full_ic_matrix.index < sp['train_end'])
                ic_train = full_ic_matrix.loc[train_mask]

                # OOS 段面板切片
                oos_mask = (common_dates >= sp['oos_start']) & (
                    common_dates < sp['oos_end'])
                oos_dates = common_dates[oos_mask]
                if len(oos_dates) == 0:
                    continue
                returns_pivot_oos = returns_pivot.loc[oos_dates]
                returns_rank_oos = returns_rank.loc[oos_dates]
                ranks_by_factor_oos = {f: r.loc[oos_dates]
                                       for f, r in ranks_by_factor.items()}

                for m in methods:
                    weights, warn_msg = _mc_compute_weights(ic_train, m)
                    per_method_weights_by_split[m].append({
                        'split': split_idx + 1,
                        'train_start': str(sp['train_start']),
                        'train_end': str(sp['train_end']),
                        'oos_start': str(sp['oos_start']),
                        'oos_end': str(sp['oos_end']),
                        'weights': weights,
                        'warn': warn_msg,
                    })
                    res = _mc_apply_and_measure(
                        ranks_by_factor_oos, returns_pivot_oos, returns_rank_oos,
                        weights, n_groups, transaction_cost
                    )
                    if res is None:
                        continue
                    per_method_ls[m].append(res['long_short_return'])
                    per_method_ls_after[m].append(
                        res['long_short_return_after_cost'])
                    per_method_ic[m].append(res['daily_ic'])
                    per_method_turnover[m].append(res['turnover'])

                yield _sse_event('progress', {
                    'phase': 'walking',
                    'message': (f'切分 {split_idx+1}/{len(splits)}：'
                                f'train {sp["train_start"].date()}→{sp["train_end"].date()}, '
                                f'oos {sp["oos_start"].date()}→{sp["oos_end"].date()}'),
                    'completed': split_idx + 1,
                    'total': len(splits),
                })

            # Phase 7: 聚合每个方法的 OOS 指标
            def _agg_metrics(ls_series, ls_after_series, ic_series, turnover_series):
                ls = ls_series.dropna() if ls_series is not None else pd.Series(dtype=float)
                ls_after = ls_after_series.dropna(
                ) if ls_after_series is not None else pd.Series(dtype=float)
                ic = ic_series.dropna() if ic_series is not None else pd.Series(dtype=float)
                tv = turnover_series.dropna() if turnover_series is not None else pd.Series(dtype=float)

                def _sharpe(r):
                    if len(r) < 2 or r.std() == 0 or not np.isfinite(r.std()):
                        return None
                    # 粗略年化常数，前端同口径比较
                    return float(r.mean() / r.std() * np.sqrt(252))

                def _max_dd(r):
                    if r.empty:
                        return None
                    equity = (1.0 + r).cumprod()
                    peak = equity.cummax()
                    dd = (equity - peak) / peak
                    return float(dd.min()) if not dd.empty else None

                metrics = {
                    'n_periods': int(len(ls)),
                    'n_ic_periods': int(len(ic)),
                    'ic_mean': float(ic.mean()) if not ic.empty else None,
                    'ic_std': float(ic.std()) if len(ic) >= 2 else None,
                    'icir': float(ic.mean() / ic.std()) if (len(ic) >= 2 and ic.std() > 0) else None,
                    'ic_tstat': (float(ic.mean() * np.sqrt(len(ic)) / ic.std())
                                 if (len(ic) >= 2 and ic.std() > 0) else None),
                    'ls_return_mean': float(ls.mean()) if not ls.empty else None,
                    'ls_total_return': float((1.0 + ls).prod() - 1.0) if not ls.empty else None,
                    'ls_total_return_after_cost': (float((1.0 + ls_after).prod() - 1.0)
                                                   if not ls_after.empty else None),
                    'sharpe': _sharpe(ls),
                    'sharpe_after_cost': _sharpe(ls_after),
                    'max_drawdown': _max_dd(ls),
                    'max_drawdown_after_cost': _max_dd(ls_after),
                    'turnover_mean': float(tv.mean()) if not tv.empty else None,
                }
                return metrics

            per_method_summary = {}
            per_method_curves = {}
            per_method_dates = {}
            per_method_ic_agg = {}
            per_method_ls_agg = {}
            per_method_ls_after_agg = {}
            per_method_turnover_agg = {}

            for m in methods:
                ls_concat = pd.concat(per_method_ls[m]).sort_index(
                ) if per_method_ls[m] else pd.Series(dtype=float)
                ls_after_concat = pd.concat(per_method_ls_after[m]).sort_index(
                ) if per_method_ls_after[m] else pd.Series(dtype=float)
                ic_concat = pd.concat(per_method_ic[m]).sort_index(
                ) if per_method_ic[m] else pd.Series(dtype=float)
                tv_concat = pd.concat(per_method_turnover[m]).sort_index(
                ) if per_method_turnover[m] else pd.Series(dtype=float)

                # 同一时间戳可能出现在相邻切分的 OOS 边界，保留最后一个
                ls_concat = ls_concat[~ls_concat.index.duplicated(keep='last')]
                ls_after_concat = ls_after_concat[~ls_after_concat.index.duplicated(
                    keep='last')]
                ic_concat = ic_concat[~ic_concat.index.duplicated(keep='last')]
                tv_concat = tv_concat[~tv_concat.index.duplicated(keep='last')]

                per_method_summary[m] = _agg_metrics(
                    ls_concat, ls_after_concat, ic_concat, tv_concat)

                # 累计净值曲线（基于 ls_after_cost）；时间戳 ISO
                ls_after_plot = ls_after_concat.fillna(0.0)
                equity_after = (1.0 + ls_after_plot).cumprod()
                per_method_curves[m] = {
                    'equity_after_cost': [float(v) for v in equity_after.values],
                    'equity_before_cost': [float(v) for v in (1.0 + ls_concat.fillna(0.0)).cumprod().values],
                }
                per_method_dates[m] = [pd.Timestamp(
                    d).isoformat() for d in equity_after.index]

                per_method_ic_agg[m] = ic_concat
                per_method_ls_agg[m] = ls_concat
                per_method_ls_after_agg[m] = ls_after_concat
                per_method_turnover_agg[m] = tv_concat

            # Phase 8: 配对 t 检验（基于 ls_after_cost 的同步日期差值）
            paired_tests = []
            try:
                from scipy.stats import ttest_rel, wilcoxon
            except Exception:
                ttest_rel = None
                wilcoxon = None

            for i, m_a in enumerate(methods):
                for m_b in methods[i + 1:]:
                    sa = per_method_ls_after_agg.get(m_a)
                    sb = per_method_ls_after_agg.get(m_b)
                    if sa is None or sb is None or sa.empty or sb.empty:
                        continue
                    aligned = pd.concat(
                        [sa, sb], axis=1, join='inner').dropna()
                    if len(aligned) < 10:
                        paired_tests.append({
                            'method_a': m_a, 'method_b': m_b,
                            'n': int(len(aligned)),
                            't_stat': None, 'p_value': None,
                            'mean_diff': None, 'message': '对齐后期数不足 10，跳过检验',
                        })
                        continue
                    diff = aligned.iloc[:, 0].to_numpy(
                    ) - aligned.iloc[:, 1].to_numpy()
                    t_stat = p_val = None
                    if ttest_rel is not None:
                        try:
                            t_stat_, p_val_ = ttest_rel(
                                aligned.iloc[:, 0], aligned.iloc[:, 1])
                            t_stat = float(t_stat_) if np.isfinite(
                                t_stat_) else None
                            p_val = float(p_val_) if np.isfinite(
                                p_val_) else None
                        except Exception:
                            pass
                    paired_tests.append({
                        'method_a': m_a, 'method_b': m_b,
                        'n': int(len(aligned)),
                        'mean_diff': float(np.mean(diff)),
                        'std_diff': float(np.std(diff, ddof=1)) if len(diff) > 1 else None,
                        't_stat': t_stat,
                        'p_value': p_val,
                    })

            # Phase 9: 收尾 → done
            total_elapsed = round(_time.time() - t_start, 2)

            # 方便前端画一条对齐的日期轴：取所有方法并集的日期排序
            all_dates = sorted(set().union(
                *[set(per_method_dates[m]) for m in methods]))

            result = {
                'success': True,
                'methods': methods,
                'factor_ids': usable_factors,
                'n_splits': len(splits),
                'splits': [
                    {k: (str(v) if hasattr(v, 'isoformat') else v)
                     for k, v in s.items()}
                    for s in splits
                ],
                'summary': per_method_summary,
                'curves': per_method_curves,
                'dates_by_method': per_method_dates,
                'all_dates': all_dates,
                'paired_tests': paired_tests,
                'weights_by_split': per_method_weights_by_split,
                'config': {
                    'train_window_days': train_window_days,
                    'oos_window_days': oos_window_days,
                    'step_days': step_days,
                    'n_groups': n_groups,
                    'predict_step': predict_step,
                    'sample_step': sample_step,
                    'transaction_cost': transaction_cost,
                    'timeframe': timeframe,
                },
                'total_elapsed': total_elapsed,
                'request_id': request_id,
            }

            yield _sse_event('done', _sanitize_for_json(result))

        except Exception as ex:
            logger.error(f"[method_comparison] 异常: {ex}", exc_info=True)
            yield _sse_event('error', {'message': f'方法对比失败: {str(ex)}'})
            yield _sse_event('done', {'success': False})
        finally:
            try:
                _discard_cancel_event(request_id)
            except Exception:
                pass

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        }
    )


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

        # 查找因子定义（v4：按一级分类目录动态查找）
        factor_file = _find_factor_definition_file(factor_id)
        if factor_file is None:
            print(f"❌ 因子定义文件不存在: {factor_id}")
            return jsonify({'success': False, 'error': f'因子 {factor_id} 不存在'})

        with open(factor_file, 'r', encoding='utf-8') as f:
            factor_info = json.load(f)

        print(f"🔍 因子信息: {factor_info.get('name', factor_id)}")

        # 检查因子类型
        computation_type = factor_info.get('computation_type')

        if computation_type == 'formula':
            # 公式因子
            factor_values = calculate_formula_factor(
                factor_info, market_data, parameters)
        elif computation_type == 'ml':
            # ML因子
            factor_values = calculate_ml_factor(
                factor_info, market_data, parameters)
        else:
            # 默认使用函数计算
            factor_values = calculate_function_factor(
                factor_info, market_data, parameters)

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
        factor_values = [random.uniform(-1, 1)
                         for _ in range(len(close_prices))]
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

        # 构建函数文件路径（v4：按一级分类目录动态查找）
        function_file = _find_factor_function_file(factor_name)
        if function_file is None:
            print(f"❌ 因子函数文件不存在: {factor_name}")
            return None

        print(f"🔍 找到因子函数文件: {function_file}")

        # 动态导入因子函数
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            factor_name, function_file)
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


@bp.route('/realtime_scan', methods=['POST'])
def realtime_scan():
    """
    截面实时扫描：基于交易所实时数据计算因子并组合，返回当前做多/做空推荐列表。
    支持 completed 和 offset_resample 两种因子K线模式。
    """
    payload = request.get_json() or {}
    factor_ids = payload.get('factor_ids') or []
    symbols = payload.get('symbols') or []
    base_timeframe = payload.get('base_timeframe') or '15m'
    factor_timeframe = payload.get('factor_timeframe') or '1h'
    factor_bar_mode = str(payload.get('factor_bar_mode', 'completed')).lower()
    if factor_bar_mode not in ('completed', 'offset_resample'):
        factor_bar_mode = 'completed'
    data_source = 'realtime'
    realtime_limit = int(payload.get('realtime_limit', 200))
    combine_mode = str(payload.get('combine_mode', 'average')).lower()
    normalize_method = str(payload.get('normalize_method', 'rank_centered')).lower()
    min_valid_count = int(payload.get('min_valid_count', 10))
    long_count = int(payload.get('long_count', 5))
    short_count = int(payload.get('short_count', 5))
    factor_weights = payload.get('factor_weights') or {}
    factor_directions = payload.get('factor_directions') or {}
    outlier_method = str(payload.get('outlier_method', 'none')).lower()
    outlier_mad_n = float(payload.get('outlier_mad_n', 5.0))
    max_lookback = int(payload.get('max_lookback', 200))
    exchange = payload.get('exchange', 'binance')
    trade_type = payload.get('trade_type', 'futures')

    if not factor_ids or not symbols or len(symbols) < 2:
        return jsonify({'success': False, 'message': '需要至少1个因子和2个币种'}), 400

    try:
        engine = get_global_engine()
    except Exception as e:
        return jsonify({'success': False, 'message': f'引擎初始化失败: {e}'}), 500

    _VALID_EXTRA_KINDS = frozenset({'metrics', 'funding', 'basis', 'mark', 'index'})
    _LEGACY_DERIVATIVES_EXPAND = frozenset({'metrics', 'basis'})
    _extras_include = set()
    for _fid in factor_ids:
        try:
            _fdef = _get_repo().load_definition(_fid)
            if not _fdef or not _fdef.metadata:
                continue
            for _req in (_fdef.metadata.get('requires_extras') or []):
                if _req == 'derivatives':
                    _extras_include.update(_LEGACY_DERIVATIVES_EXPAND)
                elif _req in _VALID_EXTRA_KINDS:
                    _extras_include.add(_req)
        except Exception:
            pass
    _extras_include = sorted(_extras_include) if _extras_include else []

    data_dict = {}
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os

    def _parse_tf_to_timedelta(tf: str):
        try:
            s = str(tf or '').strip().lower()
            m = re.fullmatch(r'(\d+)\s*([mhdw])', s)
            if not m:
                return None
            n = int(m.group(1))
            u = m.group(2)
            if u == 'm':
                return pd.Timedelta(minutes=n)
            if u == 'h':
                return pd.Timedelta(hours=n)
            if u == 'd':
                return pd.Timedelta(days=n)
            if u == 'w':
                return pd.Timedelta(weeks=n)
            return None
        except Exception:
            return None

    def _normalize_utc_index(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        x = df.copy()
        idx = pd.to_datetime(x.index, errors='coerce')
        try:
            if getattr(idx, 'tz', None) is None:
                idx = idx.tz_localize('UTC')
            else:
                idx = idx.tz_convert('UTC')
        except Exception:
            pass
        x.index = idx
        x = x[x.index.notna()].sort_index()
        return x

    def _clip_to_completed_base_bars(df: pd.DataFrame, tf: str) -> pd.DataFrame:
        """
        以“基础K线开盘时间索引”口径截断：
        now=04:11、base=1h 时，仅保留开盘时间 < 04:00 的K线（即最新可用为03:00-04:00）。
        """
        if df is None or df.empty:
            return df
        delta = _parse_tf_to_timedelta(tf)
        if delta is None:
            return df
        now_utc = pd.Timestamp.now(tz='UTC')
        last_completed_open = now_utc.floor(delta)
        clipped = df[df.index < last_completed_open].copy()
        return clipped

    def _is_fresh_enough(df: pd.DataFrame, tf: str) -> bool:
        """
        要求至少包含“最近一根已完成基础K线”的开盘时间。
        """
        if df is None or df.empty:
            return False
        delta = _parse_tf_to_timedelta(tf)
        if delta is None:
            return True
        now_utc = pd.Timestamp.now(tz='UTC')
        last_completed_open = now_utc.floor(delta) - delta
        try:
            return pd.to_datetime(df.index.max(), utc=True) >= last_completed_open
        except Exception:
            return False

    from factor_miner.core.realtime_data_fetcher import fetch_realtime_data, _create_exchange

    _shared_exchange = _create_exchange(exchange)

    def _load_symbol(sym):
        try:
            md = fetch_realtime_data(
                sym, base_timeframe, realtime_limit,
                include=_extras_include, exchange_id=exchange,
                exchange=_shared_exchange,
            )
            if md is not None and not md.empty:
                return (sym, md)
        except Exception as e:
            logger.warning(f"实时获取 {sym} 失败: {e}")
        return (sym, None)

    load_workers = min(len(symbols), os.cpu_count() or 4, 8)
    with ThreadPoolExecutor(max_workers=load_workers) as loader:
        futures = {loader.submit(_load_symbol, s): s for s in symbols}
        for future in as_completed(futures):
            sym, md = future.result()
            if md is not None:
                data_dict[sym] = md

    if _shared_exchange is not None:
        if hasattr(_shared_exchange, 'close'):
            _shared_exchange.close()

    # 统一按“仅已完成基础K线”截断，并执行新鲜度校验。
    aligned_data_dict = {}
    stale_symbols = []
    for sym, md in data_dict.items():
        md_norm = _normalize_utc_index(md)
        md_clip = _clip_to_completed_base_bars(md_norm, base_timeframe)
        if md_clip is None or md_clip.empty:
            stale_symbols.append(sym)
            continue
        if not _is_fresh_enough(md_clip, base_timeframe):
            stale_symbols.append(sym)
            continue
        aligned_data_dict[sym] = md_clip
    data_dict = aligned_data_dict

    if len(data_dict) < 2:
        stale_msg = f"，新鲜度不足币种: {len(stale_symbols)}" if stale_symbols else ""
        return jsonify({'success': False, 'message': f'有效币种不足2个（加载{len(data_dict)}个）{stale_msg}'}), 400

    evaluator = CrossSectionalEvaluator(
        n_groups=5,
        normalize_method=normalize_method,
        predict_step=1,
        sample_step=1,
        base_timeframe=base_timeframe,
        factor_timeframe=factor_timeframe,
        factor_bar_mode=factor_bar_mode,
        max_lookback=max_lookback,
        min_coverage=0.1,
        min_valid_count=max(2, min_valid_count),
        min_group_size=2,
        treat_zero_as_invalid=False,
        enable_outlier_treatment=(outlier_method != 'none'),
        outlier_method=outlier_method,
        outlier_mad_n=outlier_mad_n,
    )

    factor_values_map = {}
    for sym, md in data_dict.items():
        factor_values_map[sym] = {}
        for fid in factor_ids:
            try:
                md_sorted = md.copy().sort_index()
                use_same_tf = factor_timeframe == base_timeframe
                if use_same_tf:
                    factor_raw = evaluator._to_series(engine.compute_single_factor(fid, md_sorted))
                    factor_on_base = factor_raw
                elif factor_bar_mode == 'offset_resample':
                    factor_on_base = evaluator._build_offset_resampled_factor_series(
                        md_sorted, fid, engine
                    )
                else:
                    factor_input = evaluator._resample_ohlcv_completed(md_sorted, factor_timeframe)
                    if factor_input is None or factor_input.empty:
                        factor_on_base = None
                    else:
                        factor_raw = evaluator._to_series(engine.compute_single_factor(fid, factor_input))
                        factor_on_base = factor_raw.reindex(md_sorted.index, method='ffill') if factor_raw is not None else None

                if factor_on_base is not None and not factor_on_base.empty:
                    val = factor_on_base.iloc[-1]
                    factor_values_map[sym][fid] = float(val) if math.isfinite(val) else None
                else:
                    factor_values_map[sym][fid] = None
            except Exception:
                factor_values_map[sym][fid] = None

    factor_scores_map = {sym: {} for sym in data_dict}
    factor_ranks_map = {sym: {} for sym in data_dict}
    factor_coverage = {}

    for fid in factor_ids:
        direction = int(factor_directions.get(fid, 1))
        vals = {}
        for sym in data_dict:
            v = factor_values_map[sym].get(fid)
            if v is not None and math.isfinite(v):
                vals[sym] = v
        factor_coverage[fid] = len(vals)

        if len(vals) < min_valid_count:
            continue

        sym_list = sorted(vals.keys())
        val_array = np.array([vals[s] for s in sym_list], dtype=float)

        if outlier_method == 'mad' and len(val_array) >= 3:
            med = np.median(val_array)
            mad = np.median(np.abs(val_array - med))
            if mad > 0:
                val_array = np.clip(val_array, med - outlier_mad_n * mad, med + outlier_mad_n * mad)
        elif outlier_method == 'winsor' and len(val_array) >= 3:
            lo = np.quantile(val_array, 0.01)
            hi = np.quantile(val_array, 0.99)
            if lo < hi:
                val_array = np.clip(val_array, lo, hi)

        n = len(val_array)
        if normalize_method == 'rank_centered':
            ranks = pd.Series(val_array).rank(method='average').values
            scores = 2 * (ranks - 1) / (n - 1) - 1
        elif normalize_method == 'rank':
            ranks = pd.Series(val_array).rank(method='average').values
            scores = (ranks - 1) / (n - 1)
        else:
            scores = val_array.astype(float, copy=True)

        if direction == -1:
            if normalize_method == 'rank_centered':
                scores = -scores
            elif normalize_method == 'rank':
                scores = 1 - scores
            else:
                scores = -scores

        raw_ranks = pd.Series(val_array).rank(method='average', ascending=False).values
        for i, sym in enumerate(sym_list):
            factor_scores_map[sym][fid] = float(scores[i])
            factor_ranks_map[sym][fid] = int(raw_ranks[i])

    composite_scores = {}
    for sym in data_dict:
        active = [f for f in factor_ids if f in factor_scores_map.get(sym, {})]
        if not active:
            composite_scores[sym] = float('nan')
            continue
        if combine_mode == 'weighted':
            weights = np.array([float(factor_weights.get(f, 1.0) or 0.0) for f in active], dtype=float)
            total_w = float(np.sum(np.abs(weights)))
            if total_w <= 0:
                composite_scores[sym] = float(np.mean([factor_scores_map[sym][f] for f in active]))
            else:
                weights = weights / total_w
                composite_scores[sym] = float(sum(factor_scores_map[sym][f] * w for f, w in zip(active, weights)))
        else:
            composite_scores[sym] = float(np.mean([factor_scores_map[sym][f] for f in active]))

    def _sort_key(sym):
        v = composite_scores[sym]
        return (0 if math.isfinite(v) else 1, -v if math.isfinite(v) else 0.0)

    sorted_symbols = sorted(data_dict.keys(), key=_sort_key)

    results = []
    for rank, sym in enumerate(sorted_symbols, 1):
        results.append({
            'symbol': sym,
            'composite_score': round(composite_scores[sym], 4) if math.isfinite(composite_scores[sym]) else None,
            'rank': rank,
            'factor_values': {k: round(v, 4) if v is not None and math.isfinite(v) else None for k, v in factor_values_map.get(sym, {}).items()},
            'factor_scores': {k: round(v, 4) if math.isfinite(v) else None for k, v in factor_scores_map.get(sym, {}).items()},
            'factor_ranks': {k: v for k, v in factor_ranks_map.get(sym, {}).items()},
        })

    long_list = results[:long_count]
    short_list = results[-short_count:][::-1] if short_count > 0 else []

    return jsonify({
        'success': True,
        'long': long_list,
        'short': short_list,
        'all_ranked': results,
        'total_candidates': len(results),
        'factor_coverage': factor_coverage,
        'n_symbols_loaded': len(data_dict),
        'data_source': data_source,
        'stale_symbols_dropped': len(stale_symbols) if data_source == 'realtime' else 0,
        'scan_time': pd.Timestamp.now(tz='Asia/Shanghai').strftime('%Y-%m-%d %H:%M:%S'),
    })


_EXPORTS_DIR = Path(__file__).parent.parent / "static" / "exports"
_VALID_EXTRA_KINDS = frozenset({'metrics', 'funding', 'basis', 'mark', 'index', 'onchain'})
_LEGACY_DERIVATIVES_EXPAND = frozenset({'metrics', 'basis'})


def _resolve_factor_data_requirements(factor_ids: list) -> dict:
    result = {}
    for fid in factor_ids:
        try:
            fdef = _get_repo().load_definition(fid)
            if not fdef or not fdef.metadata:
                continue
            raw_reqs = fdef.metadata.get('requires_extras') or []
            expanded = set()
            for r in raw_reqs:
                if r == 'derivatives':
                    expanded.update(_LEGACY_DERIVATIVES_EXPAND)
                elif r in _VALID_EXTRA_KINDS:
                    expanded.add(r)
            if expanded:
                result[fid] = sorted(expanded)
        except Exception:
            pass
    return result


def _embed_factor_definitions(factor_ids: list) -> dict:
    repo = _get_repo()
    result = {}
    for fid in factor_ids:
        try:
            fdef = repo.load_definition(fid)
            if not fdef:
                continue

            artifacts_dict = fdef.artifacts.to_dict() if fdef.artifacts else {}
            computation_type = fdef.computation_type
            computation_data = artifacts_dict.copy()

            if computation_type == 'formula':
                formula_text = (fdef.artifacts.formula_inline or '').strip()
                if not formula_text or formula_text.startswith('#'):
                    func_file = fdef.artifacts.function_file
                    if not func_file:
                        for group in repo.list_source_groups():
                            candidate = repo.storage_dir / group / 'functions' / f'{fid}.py'
                            if candidate.exists():
                                func_file = str(candidate.relative_to(repo.storage_dir))
                                break
                    if func_file:
                        try:
                            func_code = repo.load_text_artifact(func_file)
                            computation_type = 'function'
                            computation_data['function_code'] = func_code
                            computation_data['function_file'] = func_file
                            computation_data['entry_point'] = fdef.artifacts.entry_point or 'calculate'
                        except Exception:
                            pass

            if computation_type == 'function' and 'function_code' not in computation_data:
                func_file = fdef.artifacts.function_file
                if func_file:
                    try:
                        func_code = repo.load_text_artifact(func_file)
                        computation_data['function_code'] = func_code
                    except Exception:
                        pass

            defn = {
                'factor_id': fdef.factor_id,
                'name': fdef.name,
                'description': fdef.description,
                'source_group': fdef.source_group,
                'factor_kind': fdef.factor_kind,
                'computation_type': computation_type,
                'computation_data': computation_data,
                'parameters': fdef.parameters or {},
                'dependencies': fdef.dependencies or [],
                'output_type': fdef.output_type or 'series',
                'metadata': fdef.metadata or {},
            }
            result[fid] = defn
        except Exception:
            pass
    return result


@bp.route('/save_scan_config', methods=['POST'])
def save_scan_config():
    payload = request.get_json() or {}
    config = payload.get('config')
    if not config or not isinstance(config, dict):
        return jsonify({'success': False, 'message': '缺少 config 对象'}), 400

    meta = config.get('_meta', {})
    if meta.get('type') != 'realtime_scan_config':
        return jsonify({'success': False, 'message': 'config._meta.type 必须为 realtime_scan_config'}), 400

    factor_ids = config.get('factor_ids') or []
    resolved = _resolve_factor_data_requirements(factor_ids)
    if resolved:
        existing = config.get('factor_data_requirements') or {}
        existing.update(resolved)
        config['factor_data_requirements'] = existing

    embedded = _embed_factor_definitions(factor_ids)
    if embedded:
        config['factor_definitions'] = embedded

    if config.get('_meta', {}).get('version', 1) < 3:
        config.setdefault('_meta', {})['version'] = 3

    _EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    ts = pd.Timestamp.now(tz='Asia/Shanghai').strftime('%Y%m%d-%H%M%S')
    filename = f"scan_config_{ts}.json"
    filepath = _EXPORTS_DIR / filename

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        logger.info(f"扫描配置已保存到服务端: {filepath}")
        return jsonify({
            'success': True,
            'path': str(filepath),
            'filename': filename,
            'config': config,
        })
    except Exception as e:
        logger.error(f"保存扫描配置失败: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500
