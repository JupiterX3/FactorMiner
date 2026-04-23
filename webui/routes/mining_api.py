"""
因子挖掘API路由
"""

import sys
import os
import json
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from flask import Blueprint, request, jsonify

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 现在再导入新的API
from factor_miner.api.factor_mining_api import FactorMiningAPI

# 创建蓝图
bp = Blueprint('mining_api', __name__, url_prefix='/api/mining')

# 全局变量
_mining_api = None

# 挖掘会话管理
mining_sessions = {}
mining_progress = {}

def get_mining_api():
    """获取因子挖掘API实例"""
    global _mining_api
    if _mining_api is None:
        _mining_api = FactorMiningAPI()
    return _mining_api

@bp.route('/start', methods=['POST'])
def start_mining():
    """启动因子挖掘"""
    try:
        data = request.get_json()
        print(f"收到挖掘请求: {data}")
        
        # 验证必要参数
        if not data.get('symbols') or not data.get('timeframes') or not data.get('selected_algorithms'):
            return jsonify({
                'success': False,
                'error': '缺少必要参数：symbols, timeframes, selected_algorithms'
            }), 400
        
        # 生成会话ID
        session_id = str(uuid.uuid4())
        
        # 创建挖掘配置
        mining_config = {
            'mode': 'standard',
            'symbols': data['symbols'],
            'timeframes': data['timeframes'],
            'selected_algorithms': data['selected_algorithms'],
            'start_date': data.get('start_date'),
            'end_date': data.get('end_date'),
            'max_factors': data.get('max_factors', 15),
            'min_ic': data.get('min_ic', 0.02),
            'min_ir': data.get('min_ir', 0.1),
            'min_sample_size': data.get('min_sample_size', 30),
            'optimization_method': data.get('optimization_method', 'greedy'),
            'session_id': session_id
        }
        
        # 初始化会话
        mining_sessions[session_id] = {
            'status': 'running',
            'progress': 0,
            'current_step': 'initializing',
            'message': '正在初始化挖掘任务...',
            'start_time': datetime.now().isoformat(),
            'config': mining_config
        }
        
        # 启动后台任务
        thread = threading.Thread(
            target=_run_mining_background,
            args=(session_id, data, mining_config)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': '挖掘任务已启动'
        })
        
    except Exception as e:
        print(f"启动挖掘失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'启动挖掘失败: {str(e)}'
        }), 500

def _run_mining_background(session_id, data, mining_config):
    """后台运行挖掘任务"""
    print(f"开始后台挖掘任务: {session_id}")
    try:
        mining_sessions[session_id]['progress'] = 10
        mining_sessions[session_id]['current_step'] = 'data_loading'
        mining_sessions[session_id]['message'] = '正在加载市场数据...'

        mining_api = get_mining_api()

        symbols = data.get('symbols', [])
        timeframes = data.get('timeframes', [])
        all_market_data = {}
        total_combos = len(symbols) * len(timeframes)
        loaded_count = 0

        for symbol in symbols:
            for timeframe in timeframes:
                loaded_count += 1
                mining_sessions[session_id]['message'] = f'正在加载数据 ({loaded_count}/{total_combos}): {symbol} {timeframe}...'
                try:
                    md = mining_api.load_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=data.get('start_date'),
                        end_date=data.get('end_date')
                    )
                    if md is not None and len(md) > 0:
                        all_market_data[(symbol, timeframe)] = md
                        print(f"✅ {symbol}/{timeframe} 数据加载成功: {len(md)} 条")
                    else:
                        print(f"⚠️ {symbol}/{timeframe} 数据为空")
                except Exception as e:
                    print(f"⚠️ {symbol}/{timeframe} 数据加载失败: {e}")

                pct = 10 + int((loaded_count / max(total_combos, 1)) * 20)
                mining_sessions[session_id]['progress'] = pct

        if not all_market_data:
            raise ValueError("所有交易对/时间框架的数据加载均失败或为空")

        print(f"数据加载完成，共 {len(all_market_data)} 个组合")

        mining_sessions[session_id]['progress'] = 30
        mining_sessions[session_id]['current_step'] = 'factor_building'
        mining_sessions[session_id]['message'] = '正在构建因子...'

        from factor_miner.core.factor_builder import FactorBuilder
        factor_builder = FactorBuilder()

        all_factors = {}
        all_factors_df_parts = []
        all_algorithms_used = set()
        combo_list = list(all_market_data.items())
        total_factors_count = 0

        for idx, ((symbol, timeframe), market_data) in enumerate(combo_list):
            mining_sessions[session_id]['message'] = f'正在构建因子 ({idx+1}/{len(combo_list)}): {symbol} {timeframe}...'
            try:
                result = factor_builder.build_all_factors(
                    data=market_data,
                    selected_algorithms=data['selected_algorithms'],
                    save_to_storage=True
                )
                if result.get('success'):
                    for fname, fseries in result.get('factors', {}).items():
                        qualified_name = f"{symbol}_{timeframe}_{fname}" if len(combo_list) > 1 else fname
                        all_factors[qualified_name] = fseries
                    if result.get('factors_df') is not None:
                        df_part = result['factors_df'].copy()
                        if len(combo_list) > 1:
                            df_part.columns = [f"{symbol}_{timeframe}_{c}" for c in df_part.columns]
                        all_factors_df_parts.append(df_part)
                    all_algorithms_used.update(result.get('algorithms_used', []))
                    total_factors_count += result.get('total_factors', 0)
                    print(f"✅ {symbol}/{timeframe} 因子构建成功: {result.get('total_factors', 0)} 个因子")
                else:
                    print(f"⚠️ {symbol}/{timeframe} 因子构建失败")
            except Exception as e:
                print(f"⚠️ {symbol}/{timeframe} 因子构建异常: {e}")

            pct = 30 + int(((idx + 1) / len(combo_list)) * 30)
            mining_sessions[session_id]['progress'] = pct

        if not all_factors:
            raise ValueError("因子构建失败：未生成任何因子")

        import pandas as pd
        combined_factors_df = pd.concat(all_factors_df_parts, axis=1) if all_factors_df_parts else pd.DataFrame()

        print(f"因子构建完成，共生成 {len(all_factors)} 个因子")

        mining_sessions[session_id]['progress'] = 60
        mining_sessions[session_id]['current_step'] = 'evaluation'
        mining_sessions[session_id]['message'] = '正在评估因子...'

        from factor_miner.core.factor_evaluator import FactorEvaluator
        evaluator = FactorEvaluator()

        evaluation_results = {}
        first_market_data = list(all_market_data.values())[0]
        eval_target = first_market_data['close'].pct_change().shift(-1)

        for factor_name, factor_series in all_factors.items():
            try:
                eval_result = evaluator.evaluate_factor(factor_series, eval_target)
                evaluation_results[factor_name] = eval_result
            except Exception as e:
                print(f"评估因子 {factor_name} 失败: {e}")
                continue

        print(f"因子评估完成，共评估 {len(evaluation_results)} 个因子")

        mining_sessions[session_id]['progress'] = 80
        mining_sessions[session_id]['current_step'] = 'optimization'
        mining_sessions[session_id]['message'] = '正在优化因子...'

        from factor_miner.core.factor_optimizer import FactorOptimizer
        optimizer = FactorOptimizer()

        optimization_result = {}
        if not combined_factors_df.empty:
            optimization_result = optimizer.optimize_factors(
                combined_factors_df, eval_target
            )

        selected_count = len(optimization_result.get('selected_factors', []))
        print(f"因子优化完成，选择了 {selected_count} 个因子")

        mining_sessions[session_id]['progress'] = 90
        mining_sessions[session_id]['current_step'] = 'saving'
        mining_sessions[session_id]['message'] = '正在保存结果...'

        factors_for_storage = {}
        for factor_name, eval_data in evaluation_results.items():
            if isinstance(eval_data, dict):
                factors_for_storage[factor_name] = {
                    'name': factor_name,
                    'ic_pearson': eval_data.get('ic_pearson'),
                    'ic_spearman': eval_data.get('ic_spearman'),
                    'sharpe_ratio': eval_data.get('sharpe_ratio'),
                    'win_rate': eval_data.get('win_rate'),
                    'long_short_return': eval_data.get('long_short_return'),
                }

        final_result = {
            'mode': 'standard',
            'factors': factors_for_storage,
            'total_factors': len(all_factors),
            'algorithms_used': list(all_algorithms_used),
            'evaluation': evaluation_results,
            'optimization': optimization_result,
        }

        save_mining_result_to_file(session_id, {
            'session_id': session_id,
            'config': mining_config,
            'results': final_result,
            'status': 'completed',
            'completed_time': datetime.now().isoformat()
        })

        mining_sessions[session_id]['status'] = 'completed'
        mining_sessions[session_id]['progress'] = 100
        mining_sessions[session_id]['current_step'] = 'completed'
        mining_sessions[session_id]['message'] = '挖掘任务完成'
        mining_sessions[session_id]['completed_time'] = datetime.now().isoformat()
        mining_sessions[session_id]['results'] = final_result

        print(f"挖掘任务完成: {session_id}")

    except Exception as e:
        print(f"挖掘任务失败: {e}")
        import traceback
        traceback.print_exc()

        mining_sessions[session_id]['status'] = 'failed'
        mining_sessions[session_id]['progress'] = 0
        mining_sessions[session_id]['current_step'] = 'failed'
        mining_sessions[session_id]['message'] = f'挖掘失败: {str(e)}'
        mining_sessions[session_id]['error'] = str(e)

@bp.route('/stop/<session_id>', methods=['POST'])
def stop_mining(session_id):
    """停止标准挖掘任务"""
    if session_id not in mining_sessions:
        return jsonify({'success': False, 'error': '会话不存在'}), 404

    session = mining_sessions[session_id]
    if session['status'] != 'running':
        return jsonify({'success': False, 'error': '任务不在运行中'})

    session['status'] = 'stopped'
    session['message'] = '挖掘任务已停止'
    session['current_step'] = 'stopped'
    session['completed_time'] = datetime.now().isoformat()

    if session.get('results'):
        save_mining_result_to_file(session_id, {
            'session_id': session_id,
            'config': session.get('config', {}),
            'results': session.get('results', {}),
            'status': 'stopped',
            'completed_time': session['completed_time']
        })

    return jsonify({'success': True, 'message': '挖掘任务已停止'})

@bp.route('/status/<session_id>', methods=['GET'])
def get_mining_status(session_id):
    """获取挖掘状态"""
    if session_id not in mining_sessions:
        return jsonify({'success': False, 'error': '会话不存在'}), 404
    
    session = mining_sessions[session_id]
    return jsonify({
        'success': True,
        'status': session['status'],
        'progress': session['progress'],
        'current_step': session['current_step'],
        'message': session['message'],
        'start_time': session.get('start_time'),
        'completed_time': session.get('completed_time')
    })

@bp.route('/progress/<session_id>', methods=['GET'])
def get_mining_progress(session_id):
    """获取挖掘进度（SSE）"""
    def generate_progress():
        while True:
            if session_id not in mining_sessions:
                yield f"data: {json.dumps({'error': '会话不存在'})}\n\n"
                break
            
            session = mining_sessions[session_id]
            progress_data = {
                'status': session['status'],
                'progress': session['progress'],
                'current_step': session['current_step'],
                'message': session['message']
            }
            
            yield f"data: {json.dumps(progress_data)}\n\n"
            
            if session['status'] in ['completed', 'failed', 'stopped']:
                break
            
            import time
            time.sleep(1)
    
    from flask import Response
    return Response(generate_progress(), mimetype='text/event-stream')

@bp.route('/algorithms', methods=['GET'])
def get_algorithms():
    """获取可用算法列表"""
    try:
        from factor_miner.core.factor_builder import FactorBuilder
        builder = FactorBuilder()
        algorithms = builder.scan_all_algorithms()
        return jsonify({'success': True, 'algorithms': algorithms})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@bp.route('/algorithms/<algorithm_id>', methods=['GET'])
def get_algorithm_info(algorithm_id):
    """获取算法详细信息"""
    try:
        from factor_miner.core.factor_builder import FactorBuilder
        builder = FactorBuilder()
        algorithm_info = builder.get_algorithm_info(algorithm_id)
        if algorithm_info:
            return jsonify({'success': True, 'algorithm': algorithm_info})
        else:
            return jsonify({'success': False, 'error': '算法不存在'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@bp.route('/history', methods=['GET'])
def get_mining_history():
    """获取挖掘历史"""
    try:
        sessions = load_completed_mining_sessions()
        history = []

        def _append_session_row(session_id_key, session_data):
            if not isinstance(session_data, dict):
                return
            results = session_data.get('results', {}) if isinstance(session_data.get('results'), dict) else {}
            sid = session_data.get('session_id') or session_id_key
            if not sid or str(sid).lower() in ('metadata',):
                return
            completed = (
                session_data.get('completed_time')
                or session_data.get('timestamp')
            )
            config = session_data.get('config', {}) if isinstance(session_data.get('config'), dict) else {}
            mode_guess = results.get('mode', config.get('mode', 'unknown'))
            if mode_guess == 'unknown' and config.get('selected_algorithms'):
                mode_guess = 'standard'
            history.append({
                'session_id': sid,
                'config': config,
                'mode': mode_guess,
                'total_factors': results.get('total_factors', 0),
                'algorithms_used': results.get('algorithms_used', []),
                'completed_time': completed,
                'timestamp': completed,
                'status': session_data.get('status', 'unknown')
            })

        if isinstance(sessions, dict):
            if 'mining_sessions' in sessions and isinstance(sessions['mining_sessions'], list):
                for session_data in sessions['mining_sessions']:
                    _append_session_row(session_data.get('session_id', ''), session_data)
            for session_id, session_data in sessions.items():
                if session_id in ('mining_sessions', 'metadata'):
                    continue
                if isinstance(session_data, dict):
                    _append_session_row(session_id, session_data)
        elif isinstance(sessions, list):
            for session_data in sessions:
                is_dict = isinstance(session_data, dict)
                if not is_dict:
                    continue
                results = session_data.get('results', {}) if isinstance(session_data.get('results'), dict) else {}
                ct = session_data.get('completed_time') or session_data.get('timestamp')
                sid = session_data.get('session_id', '')
                if not sid or str(sid).lower() in ('metadata',):
                    continue
                config = session_data.get('config', {}) if isinstance(session_data.get('config'), dict) else {}
                mode_guess = results.get('mode', config.get('mode', 'unknown'))
                if mode_guess == 'unknown' and config.get('selected_algorithms'):
                    mode_guess = 'standard'
                history.append({
                    'session_id': sid,
                    'config': config,
                    'mode': mode_guess,
                    'total_factors': results.get('total_factors', 0),
                    'algorithms_used': results.get('algorithms_used', []),
                    'completed_time': ct,
                    'timestamp': ct,
                    'status': session_data.get('status', 'unknown')
                })

        # 按 session_id 去重（同一任务可能同时出现在 mining_sessions 与顶层键）
        dedup = {}
        for row in history:
            sid = row.get('session_id') or ''
            if not sid:
                continue
            prev = dedup.get(sid)
            if prev is None or str((row or {}).get('completed_time') or '') >= str((prev or {}).get('completed_time') or ''):
                dedup[sid] = row
        history = list(dedup.values())

        # 按完成时间倒序排列
        history.sort(key=lambda x: str((x or {}).get('completed_time') or ''), reverse=True)

        return jsonify({'success': True, 'history': history})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

def load_completed_mining_sessions():
    """加载已完成的挖掘会话"""
    try:
        # 优先从mining_sessions.json读取
        sessions_file = Path(__file__).parent.parent.parent / "factorlib" / "basic_kline" / "mining_history" / "mining_sessions.json"
        
        if sessions_file.exists():
            with open(sessions_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    return json.loads(content)
        
        # 如果mining_sessions.json为空或不存在，返回空字典
        return {}
        
    except Exception as e:
        print(f"加载挖掘会话失败: {e}")
        return {}

def _get_history_dir():
    return Path(__file__).parent.parent.parent / "factorlib" / "basic_kline" / "mining_history"

def _get_sessions_file():
    return _get_history_dir() / "mining_sessions.json"

def _save_sessions_file(sessions):
    sessions_file = _get_sessions_file()
    _get_history_dir().mkdir(parents=True, exist_ok=True)
    with open(sessions_file, 'w', encoding='utf-8') as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)

@bp.route('/history/delete/<session_id>', methods=['POST'])
def delete_mining_history(session_id):
    """删除指定挖掘历史（同时删除结果文件）"""
    try:
        if not session_id:
            return jsonify({'success': False, 'error': '缺少session_id'}), 400

        sessions = load_completed_mining_sessions()
        changed = False

        if isinstance(sessions, dict):
            if session_id in sessions:
                sessions.pop(session_id, None)
                changed = True
            arr = sessions.get('mining_sessions')
            if isinstance(arr, list):
                new_arr = [x for x in arr if not (isinstance(x, dict) and x.get('session_id') == session_id)]
                if len(new_arr) != len(arr):
                    sessions['mining_sessions'] = new_arr
                    changed = True

        if changed and isinstance(sessions, dict):
            _save_sessions_file(sessions)

        history_dir = _get_history_dir()
        result_file = history_dir / f"mining_results_{session_id}.json"
        if result_file.exists():
            try:
                result_file.unlink()
            except Exception as e:
                print(f"删除结果文件失败: {e}")

        if session_id in mining_sessions:
            mining_sessions.pop(session_id, None)

        return jsonify({'success': True, 'deleted': True})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/history/clear', methods=['POST'])
def clear_mining_history():
    """清空全部挖掘历史（同时删除结果文件）"""
    try:
        sessions = load_completed_mining_sessions()
        deleted_ids = set()

        if isinstance(sessions, dict):
            for k, v in list(sessions.items()):
                if k == 'mining_sessions':
                    continue
                if isinstance(v, dict):
                    sid = v.get('session_id') or k
                    if sid:
                        deleted_ids.add(str(sid))
            arr = sessions.get('mining_sessions')
            if isinstance(arr, list):
                for item in arr:
                    if isinstance(item, dict) and item.get('session_id'):
                        deleted_ids.add(str(item.get('session_id')))
            sessions = {}
            _save_sessions_file(sessions)

        history_dir = _get_history_dir()
        deleted_files = 0
        for sid in deleted_ids:
            fp = history_dir / f"mining_results_{sid}.json"
            if fp.exists():
                try:
                    fp.unlink()
                    deleted_files += 1
                except Exception as e:
                    print(f"删除结果文件失败: {e}")

        mining_sessions.clear()
        mining_progress.clear()

        return jsonify({'success': True, 'deleted_count': len(deleted_ids), 'deleted_files': deleted_files})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

def _find_session_in_file_storage(sessions, session_id):
    """从 mining_sessions.json 中解析会话：支持顶层 uuid 键或 mining_sessions 数组旧结构。"""
    if not isinstance(sessions, dict) or not session_id:
        return None
    if session_id in sessions and session_id != 'mining_sessions':
        sd = sessions[session_id]
        return sd if isinstance(sd, dict) else None
    arr = sessions.get('mining_sessions')
    if isinstance(arr, list):
        for item in arr:
            if isinstance(item, dict) and item.get('session_id') == session_id:
                return item
    return None

@bp.route('/result/<session_id>', methods=['GET'])
def get_mining_result(session_id):
    """获取挖掘结果"""
    try:
        # 先从内存中查找
        if session_id in mining_sessions:
            session = mining_sessions[session_id]
            if session.get('status') in ('completed', 'stopped'):
                return jsonify({
                    'success': True,
                    'session_id': session_id,
                    'results': session.get('results', {}),
                    'config': session.get('config', {}),
                    'completed_time': session.get('completed_time')
                })
        
        # 从文件中加载
        sessions = load_completed_mining_sessions()
        session_data = _find_session_in_file_storage(sessions, session_id)
        if session_data:
            return jsonify({
                'success': True,
                'session_id': session_id,
                'results': session_data.get('results', {}),
                'config': session_data.get('config', {}),
                'completed_time': session_data.get('completed_time')
            })
        
        return jsonify({'success': False, 'error': '会话不存在'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def save_mining_result_to_file(session_id, session_data):
    """保存挖掘结果到文件"""
    try:
        # 确保目录存在
        history_dir = Path(__file__).parent.parent.parent / "factorlib" / "basic_kline" / "mining_history"
        history_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存到mining_sessions.json
        sessions_file = history_dir / "mining_sessions.json"
        
        # 加载现有会话
        try:
            if sessions_file.exists():
                with open(sessions_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        sessions = json.loads(content)
                    else:
                        sessions = {}
            else:
                sessions = {}
        except:
            sessions = {}
        
        # 添加新会话
        sessions[session_id] = session_data
        
        # 保存到文件
        with open(sessions_file, 'w', encoding='utf-8') as f:
            json.dump(sessions, f, ensure_ascii=False, indent=2)
        
        # 保存详细结果到单独文件
        result_file = history_dir / f"mining_results_{session_id}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        
        print(f"挖掘结果已保存: {session_id}")
        
    except Exception as e:
        print(f"保存挖掘结果失败: {e}")

def _clean_evaluation_data(evaluation_result):
    """清理评估数据，只保留必要信息"""
    if not isinstance(evaluation_result, dict):
        return evaluation_result
    
    # 只保留这些字段
    cleaned = {}
    for key in ['factor_name', 'ic_pearson', 'ic_spearman', 'sharpe_ratio', 'win_rate', 'long_short_return']:
        if key in evaluation_result:
            cleaned[key] = evaluation_result[key]
    
    return cleaned

def _clean_optimization_data(optimization_result):
    """清理优化数据，只保留必要信息"""
    if not isinstance(optimization_result, dict):
        return optimization_result
    
    cleaned = {}
    for key, value in optimization_result.items():
        if key == 'selected_factors' and isinstance(value, list):
            # 只保留因子名称列表
            cleaned[key] = value
        elif key in ['method', 'score', 'total_factors']:
            # 保留这些重要字段
            cleaned[key] = value
        # 其他字段可能包含大量数据，暂时不保留
    
    return cleaned

@bp.route('/save_selected_factors', methods=['POST'])
def save_selected_factors():
    """保存选中的因子到存储系统"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        factor_ids = data.get('factor_ids', [])
        
        if not session_id:
            return jsonify({'success': False, 'message': '缺少session_id'})
        
        if not factor_ids:
            return jsonify({'success': False, 'message': '没有选择要保存的因子'})
        
        # 从挖掘结果中获取因子定义
        session_data = None
        if session_id in mining_sessions:
            session_data = mining_sessions[session_id]
        if not session_data:
            session_data = _find_session_in_file_storage(load_completed_mining_sessions(), session_id)
        if not session_data:
            return jsonify({'success': False, 'message': '挖掘会话不存在'})
        
        results = session_data.get('results', {})
        factors = results.get('factors', {})
        mode = results.get('mode', session_data.get('config', {}).get('mode', 'standard'))

        from factor_miner.core.factor_storage import get_global_storage
        storage = get_global_storage()

        algo_name = 'standard_mining'
        if mode == 'cross_sectional':
            algo_name = 'gp_cross_sectional'
        elif mode == 'cross_sectional_rl':
            algo_name = 'rl_cross_sectional'

        saved_count = 0
        saved_factor_ids = []
        timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        for idx, factor_id in enumerate(factor_ids, start=1):
            factor_info = factors.get(factor_id)
            if not factor_info:
                continue

            new_factor_id = f"mined_{mode}_{timestamp_tag}_{idx}"
            expression = factor_info.get('expression', '')
            performance_metrics = dict(factor_info)
            performance_metrics['expression'] = expression
            performance_metrics['source_session_id'] = session_id
            performance_metrics['source_factor_id'] = factor_id

            ok = storage.save_minactor_factor(
                factor_id=new_factor_id,
                name=f"Mined_{mode}_{idx}",
                algorithm_name=algo_name,
                description=f"来源:{mode} 表达式:{expression[:200]}",
                category='mined_factor',
                performance_metrics=performance_metrics,
            )
            if ok:
                saved_count += 1
                saved_factor_ids.append(new_factor_id)
        
        return jsonify({
            'success': True, 
            'saved_count': saved_count,
            'saved_factor_ids': saved_factor_ids,
            'message': f'成功保存 {saved_count} 个因子到“挖掘因子”分类'
        })
        
    except Exception as e:
        print(f"❌ 保存选中因子失败: {e}")
        return jsonify({'success': False, 'message': f'保存失败: {str(e)}'})


@bp.route('/diff/<session_id>', methods=['GET'])
def get_mining_diff(session_id):
    """获取挖掘结果与已有因子库的对比报告"""
    try:
        sessions = load_completed_mining_sessions()
        session_data = _find_session_in_file_storage(sessions, session_id)

        if not session_data:
            if session_id in mining_sessions:
                session_data = mining_sessions[session_id]
            else:
                return jsonify({'success': False, 'error': '会话不存在'})

        results = session_data.get('results', {}) if isinstance(session_data, dict) else {}
        mined_factors = results.get('factors', {})

        if not mined_factors:
            return jsonify({
                'success': True,
                'diff_report': {
                    'summary': {'total_mined': 0, 'new': 0, 'identical': 0, 'different': 0, 'missing_artifact': 0},
                    'items': []
                }
            })

        try:
            from factor_miner.core.factor_storage import get_global_storage
            storage = get_global_storage()
            existing_factors = storage.list_factors() if hasattr(storage, 'list_factors') else {}
        except Exception:
            existing_factors = {}

        existing_ids = set()
        if isinstance(existing_factors, dict):
            existing_ids = set(existing_factors.keys())
        elif isinstance(existing_factors, list):
            existing_ids = {f.get('factor_id', f.get('id', '')) for f in existing_factors if isinstance(f, dict)}

        items = []
        summary = {'total_mined': len(mined_factors), 'new': 0, 'identical': 0, 'different': 0, 'missing_artifact': 0}

        for factor_id, factor_info in mined_factors.items():
            if not isinstance(factor_info, dict):
                continue
            item = {
                'factor_id': factor_id,
                'status': 'new',
                'existing': None,
                'new': {
                    'model_meta': {'signature': factor_info.get('expression', factor_info.get('name', ''))},
                }
            }
            if factor_id in existing_ids:
                item['status'] = 'identical'
                summary['identical'] += 1
            else:
                summary['new'] += 1
            items.append(item)

        return jsonify({
            'success': True,
            'diff_report': {
                'summary': summary,
                'items': items
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


# ==================== 截面因子挖掘 API ====================

@bp.route('/cross_sectional/start', methods=['POST'])
def start_cross_sectional_mining():
    """启动截面因子挖掘（GP遗传编程）"""
    try:
        data = request.get_json()
        print(f"收到截面挖掘请求: {data}")

        symbols = data.get('symbols', [])
        timeframe = data.get('timeframe', '1d')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        data_source = data.get('data_source', 'binance')

        if not symbols or len(symbols) < 3:
            return jsonify({
                'success': False,
                'error': f'截面挖掘至少需要3个交易对，当前选择了{len(symbols) if symbols else 0}个'
            }), 400

        session_id = str(uuid.uuid4())

        gp_config = {
            'population_size': int(data.get('population_size', 200)),
            'max_generations': int(data.get('max_generations', 30)),
            'max_depth': int(data.get('max_depth', 5)),
            'crossover_rate': float(data.get('crossover_rate', 0.7)),
            'mutation_rate': float(data.get('mutation_rate', 0.2)),
            'min_ic': float(data.get('min_ic', 0.02)),
            'min_ir': float(data.get('min_ir', 0.1)),
            'min_coverage': float(data.get('min_coverage', 0.2)),
            'max_factors': int(data.get('max_factors', 15)),
            'max_correlation': float(data.get('max_correlation', 0.7)),
        }

        mining_sessions[session_id] = {
            'status': 'running',
            'progress': 0,
            'current_step': 'initializing',
            'message': '正在初始化截面挖掘任务...',
            'start_time': datetime.now().isoformat(),
            'config': {
                'mode': 'cross_sectional',
                'symbols': symbols,
                'timeframe': timeframe,
                'start_date': start_date,
                'end_date': end_date,
                'data_source': data_source,
                'gp_config': gp_config,
                'session_id': session_id
            },
            'generation_log': [],
            '_gp_miner_ref': None,
        }

        thread = threading.Thread(
            target=_run_cross_sectional_mining_background,
            args=(session_id, symbols, timeframe, start_date, end_date, data_source, gp_config)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': '截面挖掘任务已启动'
        })

    except Exception as e:
        print(f"启动截面挖掘失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'启动截面挖掘失败: {str(e)}'
        }), 500


def _run_cross_sectional_mining_background(session_id, symbols, timeframe,
                                            start_date, end_date, data_source, gp_config):
    """后台运行截面挖掘任务"""
    print(f"开始后台截面挖掘任务: {session_id}")
    try:
        mining_sessions[session_id]['progress'] = 5
        mining_sessions[session_id]['current_step'] = 'data_loading'
        mining_sessions[session_id]['message'] = f'正在加载 {len(symbols)} 个交易对的数据...'

        from factor_miner.core.data_loader import DataLoader
        loader = DataLoader()

        data_dict = {}
        failed_symbols = []

        def _fetch_symbol_data(symbol):
            try:
                market_data = loader.get_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    data_source=data_source,
                    interval=timeframe
                )
                if market_data is not None and not market_data.empty and len(market_data) >= 50:
                    return symbol, market_data, None
                return symbol, None, "数据不足或为空"
            except Exception as e:
                return symbol, None, str(e)

        max_workers = max(4, min(16, len(symbols)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch_symbol_data, symbol): symbol for symbol in symbols}
            for i, future in enumerate(as_completed(futures)):
                symbol, market_data, error = future.result()
                if market_data is not None:
                    data_dict[symbol] = market_data
                    print(f"✅ {symbol} 数据加载成功: {len(market_data)} 条")
                else:
                    failed_symbols.append(symbol)
                    print(f"⚠️ {symbol} 数据加载失败: {error}")

                load_pct = 5 + int((i + 1) / len(symbols) * 20)
                mining_sessions[session_id]['progress'] = load_pct
                mining_sessions[session_id]['message'] = f'数据加载中... ({i+1}/{len(symbols)})'

        if len(data_dict) < 3:
            raise ValueError(
                f'有效数据币种不足（{len(data_dict)}个），'
                f'截面挖掘至少需要3个币种。失败: {failed_symbols}'
            )

        mining_sessions[session_id]['progress'] = 25
        mining_sessions[session_id]['current_step'] = 'gp_mining'
        mining_sessions[session_id]['message'] = f'开始GP进化搜索（{len(data_dict)}个币种）...'

        from factor_miner.core.gp_miner import GPMiner

        miner = GPMiner(gp_config)
        mining_sessions[session_id]['_gp_miner_ref'] = miner

        def progress_callback(pct, message, detail):
            adjusted_pct = 25 + int(pct * 0.65)
            mining_sessions[session_id]['progress'] = min(adjusted_pct, 95)
            mining_sessions[session_id]['message'] = message
            if detail and detail.get('generation'):
                mining_sessions[session_id]['generation_log'].append(detail)

        result = miner.mine(data_dict, progress_callback=progress_callback)

        if not result['success']:
            raise ValueError(result.get('error', 'GP挖掘失败'))

        mining_sessions[session_id]['progress'] = 95
        mining_sessions[session_id]['current_step'] = 'saving'
        mining_sessions[session_id]['message'] = '正在保存挖掘结果...'

        factors_for_storage = {}
        for factor_info in result['factors']:
            fid = factor_info['factor_id']
            factors_for_storage[fid] = {
                'name': factor_info['name'],
                'expression': factor_info['expression'],
                'ic_mean': factor_info['ic_mean'],
                'icir': factor_info['icir'],
                'rank_ic_mean': factor_info['rank_ic_mean'],
                'rank_icir': factor_info['rank_icir'],
                'long_short_return': factor_info['long_short_return'],
                'n_symbols': factor_info['n_symbols'],
                'n_periods': factor_info['n_periods'],
                'total_periods': factor_info.get('total_periods', 0),
                'coverage_rate': factor_info.get('coverage_rate', 0.0),
                'fitness': factor_info['fitness'],
                'depth': factor_info['depth'],
                'size': factor_info['size'],
            }

        # 不再自动保存到因子库：仅保留在本次挖掘结果中，
        # 由用户在前端勾选并点击“保存”后再落库。

        final_result = {
            'mode': 'cross_sectional',
            'factors': factors_for_storage,
            'total_factors': len(result['factors']),
            'n_symbols': result['n_symbols'],
            'total_evaluated': result.get('total_evaluated', 0),
            'generation_stats': result.get('generation_stats', []),
            'gp_config': result.get('config', gp_config),
        }

        was_stopped = result.get('stopped', False)
        final_status = 'stopped' if was_stopped else 'completed'
        status_msg = (
            f'截面挖掘已停止（完成{result.get("actual_generations", "?")}代），发现 {len(result["factors"])} 个因子'
            if was_stopped
            else f'截面挖掘完成！发现 {len(result["factors"])} 个有效因子'
        )

        save_mining_result_to_file(session_id, {
            'session_id': session_id,
            'config': mining_sessions[session_id]['config'],
            'results': final_result,
            'status': final_status,
            'completed_time': datetime.now().isoformat()
        })

        mining_sessions[session_id]['status'] = final_status
        mining_sessions[session_id]['progress'] = 100
        mining_sessions[session_id]['current_step'] = 'completed'
        mining_sessions[session_id]['message'] = status_msg
        mining_sessions[session_id]['completed_time'] = datetime.now().isoformat()
        mining_sessions[session_id]['results'] = final_result

        print(f"截面挖掘任务完成: {session_id}, 发现 {len(result['factors'])} 个因子, 停止={was_stopped}")

    except Exception as e:
        print(f"截面挖掘任务失败: {e}")
        import traceback
        traceback.print_exc()

        mining_sessions[session_id]['status'] = 'failed'
        mining_sessions[session_id]['progress'] = 0
        mining_sessions[session_id]['current_step'] = 'failed'
        mining_sessions[session_id]['message'] = f'截面挖掘失败: {str(e)}'
        mining_sessions[session_id]['error'] = str(e)


@bp.route('/cross_sectional/generation_log/<session_id>', methods=['GET'])
def get_generation_log(session_id):
    """获取GP进化日志"""
    if session_id not in mining_sessions:
        return jsonify({'success': False, 'error': '会话不存在'}), 404

    session = mining_sessions[session_id]
    log = session.get('generation_log', [])
    return jsonify({
        'success': True,
        'log': log[-50:],
        'total_entries': len(log)
    })


@bp.route('/cross_sectional/stop/<session_id>', methods=['POST'])
def stop_cross_sectional_mining(session_id):
    """停止截面挖掘任务"""
    if session_id not in mining_sessions:
        return jsonify({'success': False, 'error': '会话不存在'}), 404

    session = mining_sessions[session_id]
    if session['status'] != 'running':
        return jsonify({'success': False, 'error': '任务不在运行中'})

    session['status'] = 'stopping'
    session['message'] = '正在停止挖掘任务...'

    if session.get('config', {}).get('mode') == 'cross_sectional_rl':
        rl_miner_ref = session.get('_rl_miner_ref')
        if rl_miner_ref and hasattr(rl_miner_ref, 'stop'):
            rl_miner_ref.stop()
    elif session.get('config', {}).get('mode') == 'cross_sectional':
        gp_miner_ref = session.get('_gp_miner_ref')
        if gp_miner_ref and hasattr(gp_miner_ref, 'request_stop'):
            gp_miner_ref.request_stop()

    return jsonify({'success': True, 'message': '停止信号已发送'})


# ==================== RL截面挖掘 ====================

@bp.route('/cross_sectional_rl/start', methods=['POST'])
def start_rl_cross_sectional_mining():
    """启动RL截面因子挖掘"""
    try:
        from factor_miner.core.rl_miner import TORCH_AVAILABLE
        if not TORCH_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'PyTorch未安装，RL挖掘不可用。请安装: pip install torch'
            }), 400

        data = request.get_json()
        print(f"收到RL截面挖掘请求: {data}")

        symbols = data.get('symbols', [])
        timeframe = data.get('timeframe', '1d')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        data_source = data.get('data_source', 'binance')

        if not symbols or len(symbols) < 3:
            return jsonify({
                'success': False,
                'error': f'截面挖掘至少需要3个交易对，当前选择了{len(symbols) if symbols else 0}个'
            }), 400

        session_id = str(uuid.uuid4())

        rl_config = {
            'device': data.get('device', 'auto'),
            'batch_size': int(data.get('batch_size', 4096)),
            'train_steps': int(data.get('train_steps', 500)),
            'max_formula_len': int(data.get('max_formula_len', 16)),
            'lr': float(data.get('lr', 1e-3)),
            'use_lord': bool(data.get('use_lord', True)),
            'lord_decay_rate': float(data.get('lord_decay_rate', 1e-3)),
            'entropy_coef': float(data.get('entropy_coef', 0.01)),
            'd_model': int(data.get('d_model', 64)),
            'nhead': int(data.get('nhead', 4)),
            'num_layers': int(data.get('num_layers', 2)),
            'num_loops': int(data.get('num_loops', 3)),
            'max_factors': int(data.get('max_factors', 15)),
            'max_correlation': float(data.get('max_correlation', 0.7)),
            'min_coverage': float(data.get('min_coverage', 0.2)),
            'trade_size': float(data.get('trade_size', 10000.0)),
            'base_fee': float(data.get('base_fee', 0.001)),
        }

        mining_sessions[session_id] = {
            'status': 'running',
            'progress': 0,
            'current_step': 'initializing',
            'message': '正在初始化RL截面挖掘任务...',
            'start_time': datetime.now().isoformat(),
            'config': {
                'mode': 'cross_sectional_rl',
                'symbols': symbols,
                'timeframe': timeframe,
                'start_date': start_date,
                'end_date': end_date,
                'data_source': data_source,
                'rl_config': rl_config,
                'session_id': session_id,
            },
            'training_log': [],
            '_rl_miner_ref': None,
        }

        thread = threading.Thread(
            target=_run_rl_cross_sectional_mining_background,
            args=(session_id, symbols, timeframe, start_date, end_date, data_source, rl_config)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'RL截面挖掘任务已启动'
        })

    except Exception as e:
        print(f"启动RL截面挖掘失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'启动RL截面挖掘失败: {str(e)}'
        }), 500


def _run_rl_cross_sectional_mining_background(session_id, symbols, timeframe,
                                               start_date, end_date, data_source, rl_config):
    """后台运行RL截面挖掘任务"""
    print(f"开始后台RL截面挖掘任务: {session_id}")
    try:
        mining_sessions[session_id]['progress'] = 5
        mining_sessions[session_id]['current_step'] = 'data_loading'
        mining_sessions[session_id]['message'] = f'正在加载 {len(symbols)} 个交易对的数据...'

        from factor_miner.core.data_loader import DataLoader
        loader = DataLoader()

        data_dict = {}
        failed_symbols = []

        def _fetch_symbol_data(symbol):
            try:
                market_data = loader.get_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    data_source=data_source,
                    interval=timeframe
                )
                if market_data is not None and not market_data.empty and len(market_data) >= 50:
                    return symbol, market_data, None
                return symbol, None, "数据不足或为空"
            except Exception as e:
                return symbol, None, str(e)

        max_workers = max(4, min(16, len(symbols)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch_symbol_data, symbol): symbol for symbol in symbols}
            for i, future in enumerate(as_completed(futures)):
                symbol, market_data, error = future.result()
                if market_data is not None:
                    data_dict[symbol] = market_data
                    print(f"✅ {symbol} 数据加载成功: {len(market_data)} 条")
                else:
                    failed_symbols.append(symbol)
                    print(f"⚠️ {symbol} 数据加载失败: {error}")

                load_pct = 5 + int((i + 1) / len(symbols) * 15)
                mining_sessions[session_id]['progress'] = load_pct
                mining_sessions[session_id]['message'] = f'数据加载中... ({i+1}/{len(symbols)})'

        if len(data_dict) < 3:
            raise ValueError(
                f'有效数据币种不足（{len(data_dict)}个），'
                f'截面挖掘至少需要3个币种。失败: {failed_symbols}'
            )

        mining_sessions[session_id]['progress'] = 20
        mining_sessions[session_id]['current_step'] = 'rl_training'
        mining_sessions[session_id]['message'] = f'开始RL训练（{len(data_dict)}个币种）...'

        from factor_miner.core.rl_miner import RLMiner

        miner = RLMiner(rl_config)
        mining_sessions[session_id]['_rl_miner_ref'] = miner

        def progress_callback(pct, message, detail=None):
            if pct < 0:
                mining_sessions[session_id]['message'] = message
                return
            adjusted_pct = 20 + int(pct * 0.72)
            mining_sessions[session_id]['progress'] = min(adjusted_pct, 95)
            mining_sessions[session_id]['message'] = message
            if detail:
                mining_sessions[session_id]['training_log'].append(detail)

        result = miner.mine(data_dict, progress_callback=progress_callback)

        if not result['success']:
            raise ValueError(result.get('error', 'RL挖掘失败'))

        mining_sessions[session_id]['progress'] = 95
        mining_sessions[session_id]['current_step'] = 'saving'
        mining_sessions[session_id]['message'] = '正在保存挖掘结果...'

        factors_for_storage = {}
        for factor_info in result['factors']:
            fid = factor_info['factor_id']
            factors_for_storage[fid] = {
                'name': factor_info['name'],
                'expression': factor_info['expression'],
                'score': factor_info['score'],
                'avg_return': factor_info['avg_return'],
                'ic_mean': factor_info.get('ic_mean', None),
                'icir': factor_info.get('icir', None),
                'rank_ic_mean': factor_info.get('rank_ic_mean', None),
                'rank_icir': factor_info.get('rank_icir', None),
                'long_short_return': factor_info.get('long_short_return', None),
                'n_symbols': factor_info.get('n_symbols', result.get('n_symbols', 0)),
                'n_periods': factor_info.get('n_periods', result.get('n_periods', 0)),
                'total_periods': factor_info.get('total_periods', result.get('n_periods', 0)),
                'coverage_rate': factor_info.get('coverage_rate', 0.0),
            }

        # 不再自动保存到因子库：仅保留在本次挖掘结果中，
        # 由用户在前端勾选并点击“保存”后再落库。

        final_result = {
            'mode': 'cross_sectional_rl',
            'factors': factors_for_storage,
            'total_factors': len(result['factors']),
            'n_symbols': result['n_symbols'],
            'n_periods': result['n_periods'],
            'device': result.get('device', 'unknown'),
            'total_evaluated': result.get('total_evaluated', 0),
            'training_history': result.get('training_history', []),
            'best_score': result.get('best_score', 0),
            'best_formula': result.get('best_formula', ''),
            'rl_config': result.get('config', rl_config),
        }

        save_mining_result_to_file(session_id, {
            'session_id': session_id,
            'config': mining_sessions[session_id]['config'],
            'results': final_result,
            'status': 'completed',
            'completed_time': datetime.now().isoformat()
        })

        mining_sessions[session_id]['status'] = 'completed'
        mining_sessions[session_id]['progress'] = 100
        mining_sessions[session_id]['current_step'] = 'completed'
        mining_sessions[session_id]['message'] = f'RL截面挖掘完成！发现 {len(result["factors"])} 个有效因子'
        mining_sessions[session_id]['completed_time'] = datetime.now().isoformat()
        mining_sessions[session_id]['results'] = final_result

        print(f"RL截面挖掘任务完成: {session_id}, 发现 {len(result['factors'])} 个因子")

    except Exception as e:
        print(f"RL截面挖掘任务失败: {e}")
        import traceback
        traceback.print_exc()

        mining_sessions[session_id]['status'] = 'failed'
        mining_sessions[session_id]['progress'] = 0
        mining_sessions[session_id]['current_step'] = 'failed'
        mining_sessions[session_id]['message'] = f'RL截面挖掘失败: {str(e)}'
        mining_sessions[session_id]['error'] = str(e)


@bp.route('/cross_sectional_rl/training_log/<session_id>', methods=['GET'])
def get_rl_training_log(session_id):
    """获取RL训练日志"""
    if session_id not in mining_sessions:
        return jsonify({'success': False, 'error': '会话不存在'}), 404

    session = mining_sessions[session_id]
    log = session.get('training_log', [])
    return jsonify({
        'success': True,
        'log': log[-100:],
        'total_entries': len(log)
    })


@bp.route('/cross_sectional_rl/check_torch', methods=['GET'])
def check_torch_available():
    """检查PyTorch是否可用"""
    try:
        from factor_miner.core.rl_miner import TORCH_AVAILABLE
        if not TORCH_AVAILABLE:
            return jsonify({
                'success': True,
                'torch_available': False,
                'message': 'PyTorch未安装'
            })

        import torch
        cuda_available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_available else 'CPU'
        return jsonify({
            'success': True,
            'torch_available': True,
            'cuda_available': cuda_available,
            'device_name': device_name,
            'torch_version': torch.__version__,
        })
    except Exception as e:
        return jsonify({
            'success': True,
            'torch_available': False,
            'message': str(e)
        })
