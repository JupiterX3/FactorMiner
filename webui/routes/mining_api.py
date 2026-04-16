"""
因子挖掘API路由
"""

import sys
import os
import json
import uuid
import threading
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
            'symbols': data['symbols'],
            'timeframes': data['timeframes'],
            'selected_algorithms': data['selected_algorithms'],
            'start_date': data.get('start_date'),
            'end_date': data.get('end_date'),
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
        # 更新进度
        mining_sessions[session_id]['progress'] = 10
        mining_sessions[session_id]['current_step'] = 'data_loading'
        mining_sessions[session_id]['message'] = '正在加载市场数据...'
        
        # 获取挖掘API
        mining_api = get_mining_api()
        
        # 加载数据
        print(f"加载数据: {data['symbols'][0]}, {data['timeframes'][0]}")
        market_data = mining_api.load_data(
            symbol=data['symbols'][0],
            timeframe=data['timeframes'][0],
            start_date=data.get('start_date'),
            end_date=data.get('end_date')
        )
        
        if market_data is None or len(market_data) == 0:
            raise ValueError("数据加载失败或数据为空")
        
        print(f"数据加载成功，共 {len(market_data)} 条记录")
        
        # 更新进度
        mining_sessions[session_id]['progress'] = 30
        mining_sessions[session_id]['current_step'] = 'factor_building'
        mining_sessions[session_id]['message'] = '正在构建因子...'
        
        # 构建因子
        from factor_miner.core.factor_builder import FactorBuilder
        factor_builder = FactorBuilder()
        
        result = factor_builder.build_all_factors(
            data=market_data,
            selected_algorithms=data['selected_algorithms'],
            save_to_storage=True
        )
        
        if not result['success']:
            raise ValueError("因子构建失败")
        
        print(f"因子构建成功，共生成 {result['total_factors']} 个因子")
        
        # 更新进度
        mining_sessions[session_id]['progress'] = 60
        mining_sessions[session_id]['current_step'] = 'evaluation'
        mining_sessions[session_id]['message'] = '正在评估因子...'
        
        # 评估因子
        from factor_miner.core.factor_evaluator import FactorEvaluator
        evaluator = FactorEvaluator()
        
        evaluation_results = {}
        for factor_name, factor_series in result['factors'].items():
            try:
                eval_result = evaluator.evaluate_factor(
                    factor_series, 
                    market_data['close'].pct_change().shift(-1)
                )
                evaluation_results[factor_name] = eval_result
            except Exception as e:
                print(f"评估因子 {factor_name} 失败: {e}")
                continue
        
        print(f"因子评估完成，共评估 {len(evaluation_results)} 个因子")
        
        # 更新进度
        mining_sessions[session_id]['progress'] = 80
        mining_sessions[session_id]['current_step'] = 'optimization'
        mining_sessions[session_id]['message'] = '正在优化因子...'
        
        # 因子优化
        from factor_miner.core.factor_optimizer import FactorOptimizer
        optimizer = FactorOptimizer()
        
        optimization_result = optimizer.optimize_factors(
            result['factors_df'],
            market_data['close'].pct_change().shift(-1)
        )
        
        print(f"因子优化完成，选择了 {len(optimization_result.get('selected_factors', []))} 个因子")
        
        # 保存结果
        mining_sessions[session_id]['progress'] = 90
        mining_sessions[session_id]['current_step'] = 'saving'
        mining_sessions[session_id]['message'] = '正在保存结果...'
        
        # 构建最终结果
        final_result = {
            'factors': result['factors'],
            'factors_df': result['factors_df'].to_dict('records'),
            'total_factors': result['total_factors'],
            'algorithms_used': result['algorithms_used'],
            'evaluation': evaluation_results,
            'optimization': optimization_result
        }
        
        # 保存到文件
        from factor_miner.core.factor_storage import get_global_storage
        storage = get_global_storage()
        storage.save_mining_history(session_id, {
            'session_id': session_id,
            'config': mining_config,
            'results': final_result,
            'status': 'completed',
            'completed_time': datetime.now().isoformat()
        })
        
        # 更新会话状态
        mining_sessions[session_id]['status'] = 'completed'
        mining_sessions[session_id]['progress'] = 100
        mining_sessions[session_id]['current_step'] = 'completed'
        mining_sessions[session_id]['message'] = '挖掘任务完成'
        mining_sessions[session_id]['completed_time'] = datetime.now().isoformat()
        
        print(f"挖掘任务完成: {session_id}")
        
    except Exception as e:
        print(f"挖掘任务失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 更新会话状态为失败
        mining_sessions[session_id]['status'] = 'failed'
        mining_sessions[session_id]['progress'] = 0
        mining_sessions[session_id]['current_step'] = 'failed'
        mining_sessions[session_id]['message'] = f'挖掘失败: {str(e)}'
        mining_sessions[session_id]['error'] = str(e)

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
            
            if session['status'] in ['completed', 'failed']:
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

        if isinstance(sessions, dict):
            if 'mining_sessions' in sessions and isinstance(sessions['mining_sessions'], list):
                for session_data in sessions['mining_sessions']:
                    history.append({
                        'session_id': session_data.get('session_id', ''),
                        'config': session_data.get('config', {}),
                        'total_factors': session_data.get('results', {}).get('total_factors', 0) if isinstance(session_data.get('results'), dict) else 0,
                        'algorithms_used': session_data.get('results', {}).get('algorithms_used', []) if isinstance(session_data.get('results'), dict) else [],
                        'completed_time': session_data.get('completed_time') or session_data.get('timestamp'),
                        'status': session_data.get('status', 'unknown')
                    })
            else:
                for session_id, session_data in sessions.items():
                    history.append({
                        'session_id': session_id,
                        'config': session_data.get('config', {}),
                        'total_factors': session_data.get('results', {}).get('total_factors', 0),
                        'algorithms_used': session_data.get('results', {}).get('algorithms_used', []),
                        'completed_time': session_data.get('completed_time'),
                        'status': session_data.get('status', 'unknown')
                    })
        elif isinstance(sessions, list):
            for session_data in sessions:
                history.append({
                    'session_id': session_data.get('session_id', '') if isinstance(session_data, dict) else '',
                    'config': session_data.get('config', {}) if isinstance(session_data, dict) else {},
                    'total_factors': session_data.get('results', {}).get('total_factors', 0) if isinstance(session_data, dict) and isinstance(session_data.get('results'), dict) else 0,
                    'algorithms_used': session_data.get('results', {}).get('algorithms_used', []) if isinstance(session_data, dict) and isinstance(session_data.get('results'), dict) else [],
                    'completed_time': session_data.get('completed_time') or session_data.get('timestamp') if isinstance(session_data, dict) else '',
                    'status': session_data.get('status', 'unknown') if isinstance(session_data, dict) else 'unknown'
                })

        # 按完成时间倒序排列
        history.sort(key=lambda x: x.get('completed_time', ''), reverse=True)

        return jsonify({'success': True, 'history': history})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

def load_completed_mining_sessions():
    """加载已完成的挖掘会话"""
    try:
        # 优先从mining_sessions.json读取
        sessions_file = Path(__file__).parent.parent.parent / "factorlib" / "minactors" / "mining_history" / "mining_sessions.json"
        
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

@bp.route('/result/<session_id>', methods=['GET'])
def get_mining_result(session_id):
    """获取挖掘结果"""
    try:
        # 先从内存中查找
        if session_id in mining_sessions:
            session = mining_sessions[session_id]
            if session['status'] == 'completed':
                return jsonify({
                    'success': True,
                    'session_id': session_id,
                    'results': session.get('results', {}),
                    'config': session.get('config', {}),
                    'completed_time': session.get('completed_time')
                })
        
        # 从文件中加载
        sessions = load_completed_mining_sessions()
        if session_id in sessions:
            session_data = sessions[session_id]
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
        history_dir = Path(__file__).parent.parent.parent / "factorlib" / "minactors" / "mining_history"
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
        session_data = load_completed_mining_sessions().get(session_id)
        if not session_data:
            return jsonify({'success': False, 'message': '挖掘会话不存在'})
        
        results = session_data.get('results', {})
        factors = results.get('factors', {})
        
        saved_count = 0
        for factor_id in factor_ids:
            if factor_id in factors:
                # 因子已经通过factor_builder保存到存储系统
                # 这里只需要确认保存成功
                saved_count += 1
        
        return jsonify({
            'success': True, 
            'saved_count': saved_count,
            'message': f'成功保存 {saved_count} 个因子'
        })
        
    except Exception as e:
        print(f"❌ 保存选中因子失败: {e}")
        return jsonify({'success': False, 'message': f'保存失败: {str(e)}'})


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
        for i, symbol in enumerate(symbols):
            try:
                market_data = loader.get_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    data_source=data_source,
                    interval=timeframe
                )
                if market_data is not None and not market_data.empty and len(market_data) >= 50:
                    data_dict[symbol] = market_data
                    print(f"✅ {symbol} 数据加载成功: {len(market_data)} 条")
                else:
                    failed_symbols.append(symbol)
                    print(f"⚠️ {symbol} 数据不足或为空")
            except Exception as e:
                failed_symbols.append(symbol)
                print(f"❌ {symbol} 数据加载失败: {e}")

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
                'fitness': factor_info['fitness'],
                'depth': factor_info['depth'],
                'size': factor_info['size'],
            }

        from factor_miner.core.factor_storage import get_global_storage
        storage = get_global_storage()

        for factor_info in result['factors']:
            try:
                storage.save_minactor_factor(
                    factor_id=factor_info['factor_id'],
                    name=factor_info['name'],
                    algorithm_name='gp_cross_sectional',
                    description=f"GP截面因子: {factor_info['expression'][:200]}",
                    category='gp_cs',
                    performance_metrics={
                        'ic_mean': factor_info['ic_mean'],
                        'icir': factor_info['icir'],
                        'rank_ic_mean': factor_info['rank_ic_mean'],
                        'rank_icir': factor_info['rank_icir'],
                        'long_short_return': factor_info['long_short_return'],
                        'n_symbols': factor_info['n_symbols'],
                        'n_periods': factor_info['n_periods'],
                    }
                )
            except Exception as e:
                print(f"保存因子 {factor_info['factor_id']} 失败: {e}")

        final_result = {
            'mode': 'cross_sectional',
            'factors': factors_for_storage,
            'total_factors': len(result['factors']),
            'n_symbols': result['n_symbols'],
            'total_evaluated': result.get('total_evaluated', 0),
            'generation_stats': result.get('generation_stats', []),
            'gp_config': result.get('config', gp_config),
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
        mining_sessions[session_id]['message'] = f'截面挖掘完成！发现 {len(result["factors"])} 个有效因子'
        mining_sessions[session_id]['completed_time'] = datetime.now().isoformat()
        mining_sessions[session_id]['results'] = final_result

        print(f"截面挖掘任务完成: {session_id}, 发现 {len(result['factors'])} 个因子")

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
            'batch_size': int(data.get('batch_size', 512)),
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
        for i, symbol in enumerate(symbols):
            try:
                market_data = loader.get_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    data_source=data_source,
                    interval=timeframe
                )
                if market_data is not None and not market_data.empty and len(market_data) >= 50:
                    data_dict[symbol] = market_data
                    print(f"✅ {symbol} 数据加载成功: {len(market_data)} 条")
                else:
                    failed_symbols.append(symbol)
                    print(f"⚠️ {symbol} 数据不足或为空")
            except Exception as e:
                failed_symbols.append(symbol)
                print(f"❌ {symbol} 数据加载失败: {e}")

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
            }

        from factor_miner.core.factor_storage import get_global_storage
        storage = get_global_storage()

        for factor_info in result['factors']:
            try:
                storage.save_minactor_factor(
                    factor_id=factor_info['factor_id'],
                    name=factor_info['name'],
                    algorithm_name='rl_cross_sectional',
                    description=f"RL截面因子: {factor_info['expression'][:200]}",
                    category='rl_cs',
                    performance_metrics={
                        'score': factor_info['score'],
                        'avg_return': factor_info['avg_return'],
                    }
                )
            except Exception as e:
                print(f"保存因子 {factor_info['factor_id']} 失败: {e}")

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
