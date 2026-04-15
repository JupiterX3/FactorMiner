"""
数据管理API路由
"""

from flask import Blueprint, request, jsonify, current_app, session
import os
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import time
import json
import ccxt  # 添加 CCXT 库
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import sys
try:
    import pyarrow.feather as pa_feather
except Exception:
    pa_feather = None

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

bp = Blueprint('data_api', __name__)

import logging
logger = logging.getLogger(__name__)

DOWNLOADS = {}
DOWNLOADS_LOCK = threading.Lock()
DOWNLOADS_CLEANUP_MINUTES = 30
DOWNLOADS_CLEANUP_INTERVAL_SECONDS = 300
LAST_DOWNLOADS_CLEANUP_TS = 0.0

MARKETS_CACHE = {}
MARKETS_CACHE_TTL_SECONDS = 300
MARKETS_CACHE_LOCK = threading.Lock()

def validate_data_path(file_path):
    """验证文件路径是否在允许的数据目录内，防止路径遍历攻击"""
    if not file_path:
        return False, "文件路径不能为空"
    
    try:
        configured_data_dir = current_app.config.get('DATA_DIR', 'data')
        if 'binance' in str(configured_data_dir) and ('futures' in str(configured_data_dir) or 'spot' in str(configured_data_dir)):
            base_data_dir = Path(configured_data_dir).parent.parent
        else:
            base_data_dir = Path(configured_data_dir)
        
        base_data_dir = base_data_dir.resolve()
        requested_path = Path(file_path).resolve()
        
        if not str(requested_path).startswith(str(base_data_dir)):
            logger.warning(f"非法路径访问尝试: {file_path}")
            return False, "非法路径：文件不在数据目录内"
        
        if not requested_path.exists():
            return False, "文件不存在"
        
        if not str(requested_path).endswith('.feather'):
            return False, "非法文件类型"
        
        return True, requested_path
    except Exception as e:
        logger.error(f"路径验证异常: {e}")
        return False, f"路径验证失败: {str(e)}"

def cleanup_old_downloads():
    """清理过期的下载任务记录"""
    global LAST_DOWNLOADS_CLEANUP_TS
    now_ts = time.time()
    if now_ts - LAST_DOWNLOADS_CLEANUP_TS < DOWNLOADS_CLEANUP_INTERVAL_SECONDS:
        return

    with DOWNLOADS_LOCK:
        LAST_DOWNLOADS_CLEANUP_TS = now_ts
        now = datetime.now()
        expired_keys = []
        for key, task in DOWNLOADS.items():
            if task.get('status') in ['completed', 'failed']:
                end_time_str = task.get('end_time')
                if end_time_str:
                    try:
                        end_time = datetime.fromisoformat(end_time_str)
                        if (now - end_time).total_seconds() > DOWNLOADS_CLEANUP_MINUTES * 60:
                            expired_keys.append(key)
                    except Exception:
                        pass
        for key in expired_keys:
            del DOWNLOADS[key]
        if expired_keys:
            logger.info(f"已清理 {len(expired_keys)} 个过期下载任务记录")

@bp.route('/exchanges', methods=['GET'])
def get_exchanges():
    """获取支持的交易所列表"""
    exchanges = [
        {
            'id': 'binance',
            'name': 'Binance',
            'type': 'cryptocurrency',
            'description': '全球最大的加密货币交易所'
        },
        {
            'id': 'okx',
            'name': 'OKX',
            'type': 'cryptocurrency',
            'description': '领先的加密货币交易平台'
        },
        {
            'id': 'bybit',
            'name': 'Bybit',
            'type': 'cryptocurrency',
            'description': '专业的加密货币衍生品交易所'
        }
    ]
    return jsonify({'success': True, 'data': exchanges})

@bp.route('/stablecoins', methods=['GET'])
def get_stablecoins():
    """获取稳定币列表"""
    from config.settings import STABLECOINS
    return jsonify({'success': True, 'data': list(STABLECOINS)})

def get_exchange_instance(exchange_id, is_futures=False):
    """获取交易所实例"""
    from factor_miner.core.data_downloader import DataDownloader
    downloader = DataDownloader()
    downloader.trade_type = 'futures' if is_futures else 'spot'
    return downloader.get_exchange_instance(config_id=None, exchange_id=exchange_id)

def format_symbol_for_download(symbol, trade_type):
    """统一格式化前端交易对到 CCXT 交易对格式"""
    # 仅支持 futures(永续) 与 spot
    if trade_type == 'futures':
        parts = symbol.split('_')
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}:USDT"
        return f"{symbol.replace('_', '/')}:USDT"
    return symbol.replace('_', '/')

def load_markets_cached(exchange_id, is_futures=False):
    """加载并缓存市场数据，避免每次请求都 load_markets"""
    cache_key = (exchange_id, 'futures' if is_futures else 'spot')
    now_ts = time.time()

    with MARKETS_CACHE_LOCK:
        cached = MARKETS_CACHE.get(cache_key)
        if cached and (now_ts - cached['timestamp'] < MARKETS_CACHE_TTL_SECONDS):
            return cached['markets']

    exchange = get_exchange_instance(exchange_id, is_futures=is_futures)
    if not exchange:
        return {}

    markets = exchange.load_markets()
    with MARKETS_CACHE_LOCK:
        MARKETS_CACHE[cache_key] = {
            'timestamp': now_ts,
            'markets': markets
        }
    return markets

def format_symbol(market, market_type='spot', exchange_id='binance'):
    """格式化交易对信息"""
    base = market['base']
    quote = market['quote']
    
    contract_type = market.get('info', {}).get('contractType', '')
    delivery_date = market.get('info', {}).get('deliveryDate', '')
    settle = market.get('settle', '')
    
    if exchange_id == 'binance':
        if contract_type == 'PERPETUAL':
            symbol = f"{base}_{quote}"
            ccxt_symbol = f"{base}/{quote}:{settle}" if settle else f"{base}/{quote}"
        elif delivery_date:
            symbol = f"{base}_{quote}_{delivery_date}"
            ccxt_symbol = f"{base}/{quote}:{settle}" if settle else f"{base}/{quote}"
        else:
            symbol = f"{base}_{quote}"
            ccxt_symbol = f"{base}/{quote}"
    elif exchange_id == 'okx':
        symbol = f"{base}-{quote}"
        ccxt_symbol = market.get('symbol', f"{base}/{quote}")
    elif exchange_id == 'bybit':
        symbol = f"{base}{quote}"
        ccxt_symbol = market.get('symbol', f"{base}/{quote}")
    else:
        symbol = f"{base}_{quote}"
        ccxt_symbol = f"{base}/{quote}"
    
    result = {
        'symbol': symbol,
        'ccxt_symbol': ccxt_symbol,
        'name': f"{base}/{quote}",
        'type': market_type,
        'base': base,
        'quote': quote,
        'active': market.get('active', True)
    }
    
    if contract_type:
        result['contract_type'] = contract_type
    if delivery_date:
        result['delivery_date'] = delivery_date
    
    return result

def _filter_markets(markets_data, market_type, exchange, filter_func):
    """通用交易对过滤函数
    
    Args:
        markets_data: ccxt 返回的市场数据字典
        market_type: 市场类型 ('spot' 或 'futures')
        exchange: 交易所名称
        filter_func: 过滤条件函数，返回 True 表示保留
    
    Returns:
        过滤后的交易对列表
    """
    seen = set()
    result = []
    
    for symbol, market in markets_data.items():
        try:
            if filter_func(market) and market.get('active', True):
                formatted = format_symbol(market, market_type, exchange)
                if formatted['symbol'] not in seen:
                    seen.add(formatted['symbol'])
                    result.append(formatted)
        except Exception:
            continue
    
    return result


@bp.route('/symbols/<exchange>', methods=['GET'])
def get_symbols(exchange):
    """获取指定交易所的交易对列表"""
    try:
        spot_markets = []
        perpetual_markets = []
        
        logger.debug(f"获取 {exchange} 交易对列表")
        
        spot_markets_data = {}
        futures_markets_data = {}
        
        try:
            spot_markets_data = load_markets_cached(exchange, is_futures=False)
        except Exception as e:
            logger.warning(f"获取现货市场失败: {e}")
        
        try:
            futures_markets_data = load_markets_cached(exchange, is_futures=True)
        except Exception as e:
            logger.warning(f"获取期货市场失败: {e}")
        
        spot_markets = _filter_markets(
            spot_markets_data, 'spot', exchange,
            lambda m: m.get('quote') == 'USDT'
        )
        
        perpetual_markets = _filter_markets(
            futures_markets_data, 'futures', exchange,
            lambda m: m.get('settle') == 'USDT' and 
                      m.get('info', {}).get('contractType') == 'PERPETUAL'
        )
        
        spot_markets.sort(key=lambda x: x['symbol'])
        perpetual_markets.sort(key=lambda x: x['symbol'])
        
        logger.info(f"现货: {len(spot_markets)} 个 | 期货(永续): {len(perpetual_markets)} 个")
        
        return jsonify({
            'success': True,
            'data': {
                'spot': spot_markets,
                'futures': perpetual_markets
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'获取交易对失败: {str(e)}'
        }), 500

@bp.route('/timeframes', methods=['GET'])
def get_timeframes():
    """获取支持的时间框架"""
    timeframes = [
        {'value': '1m', 'name': '1分钟', 'description': '1分钟K线数据'},
        {'value': '3m', 'name': '3分钟', 'description': '3分钟K线数据'},
        {'value': '5m', 'name': '5分钟', 'description': '5分钟K线数据'},
        {'value': '15m', 'name': '15分钟', 'description': '15分钟K线数据'},
        {'value': '1h', 'name': '1小时', 'description': '1小时K线数据'},
        {'value': '2h', 'name': '2小时', 'description': '2小时K线数据'},
        {'value': '4h', 'name': '4小时', 'description': '4小时K线数据'},
        {'value': '6h', 'name': '6小时', 'description': '6小时K线数据'},
        {'value': '8h', 'name': '8小时', 'description': '8小时K线数据'},
        {'value': '12h', 'name': '12小时', 'description': '12小时K线数据'},
        {'value': '1d', 'name': '1天', 'description': '1天K线数据'}
    ]
    return jsonify({'success': True, 'data': timeframes})

def _to_datetime_series(series):
    """将任意时间列稳健转换为 datetime"""
    try:
        if pd.api.types.is_datetime64_any_dtype(series):
            return pd.to_datetime(series, errors='coerce')
        if pd.api.types.is_numeric_dtype(series):
            s = series.dropna()
            if len(s) == 0:
                return pd.to_datetime(series, errors='coerce', unit='s')
            sample = s.iloc[0]
            unit = 'ms' if sample > 10_000_000_000 else 's'
            return pd.to_datetime(series, errors='coerce', unit=unit)
        return pd.to_datetime(series, errors='coerce')
    except Exception:
        return pd.to_datetime(series, errors='coerce')

def _parse_filename(filename):
    """解析数据文件名，提取 symbol、timeframe、data_type

    支持的格式:
        BTC_USDT_USDT-1h-futures  -> symbol=BTC_USDT, timeframe=1h, data_type=futures
        BTC_USDT_USDT-1h-spot     -> symbol=BTC_USDT, timeframe=1h, data_type=spot
        BTC_USDT_USDT-1h          -> symbol=BTC_USDT, timeframe=1h, data_type=unknown
    """
    data_type = 'unknown'
    base_name = filename

    if filename.endswith('-futures'):
        data_type = 'futures'
        base_name = filename[:-8]
    elif filename.endswith('-spot'):
        data_type = 'spot'
        base_name = filename[:-5]

    last_hyphen = base_name.rfind('-')
    if last_hyphen != -1:
        symbol_part = base_name[:last_hyphen]
        timeframe_part = base_name[last_hyphen + 1:]
        symbol_parts = symbol_part.split('_')
        if len(symbol_parts) >= 2:
            return f"{symbol_parts[0]}_{symbol_parts[1]}", timeframe_part, data_type

    return filename, 'unknown', data_type


def _extract_feather_metadata(file_path):
    """读取 feather 文件元信息"""
    start_str = ""
    end_str = ""
    columns = []
    data_points = 0

    time_name_candidates = ['open_time', 'opentime', 'start_time', 'close_time', 'closetime', 'end_time', 'time', 'timestamp', 'datetime', 'date']

    if pa_feather is not None:
        table = pa_feather.read_table(file_path)
        columns = list(table.column_names)
        data_points = table.num_rows

        cols_lower = {c.lower(): c for c in columns}
        time_col = None
        for c in time_name_candidates:
            if c in cols_lower:
                time_col = cols_lower[c]
                break

        if time_col:
            ts_table = pa_feather.read_table(file_path, columns=[time_col])
            ts_df = ts_table.to_pandas()
            ts_series = None
            # 某些 feather 文件会把单列时间转成 index（而非普通列）
            if time_col in ts_df.columns:
                ts_series = ts_df[time_col]
            elif ts_df.index is not None:
                index_name = str(ts_df.index.name or '').lower()
                if index_name == str(time_col).lower() or len(ts_df.columns) == 0:
                    ts_series = ts_df.index.to_series(index=ts_df.index)

            if ts_series is not None:
                dt = _to_datetime_series(ts_series)
                if dt.notna().any():
                    start_str = dt.min().strftime('%Y-%m-%d')
                    end_str = dt.max().strftime('%Y-%m-%d')
        return data_points, start_str, end_str, columns

    df = pd.read_feather(file_path)
    data_points = len(df)
    columns = list(df.columns)
    cols_lower = {c.lower(): c for c in columns}
    time_col = None
    for c in time_name_candidates:
        if c in cols_lower:
            time_col = cols_lower[c]
            break
    if time_col:
        dt = _to_datetime_series(df[time_col])
        if dt.notna().any():
            start_str = dt.min().strftime('%Y-%m-%d')
            end_str = dt.max().strftime('%Y-%m-%d')
    return data_points, start_str, end_str, columns

@bp.route('/local-data', methods=['GET'])
def get_local_data():
    """获取本地存储的数据信息"""
    try:
        # 获取查询参数
        exchange = request.args.get('exchange', 'binance')
        trade_type = request.args.get('trade_type', '')  # 空字符串表示所有类型
        
        # 构建数据目录路径
        configured_data_dir = current_app.config.get('DATA_DIR', 'data')
        logger.debug(f"配置的DATA_DIR: {configured_data_dir}")
        
        # 如果配置的路径已经指向具体目录，则使用其父目录
        if 'binance' in str(configured_data_dir) and ('futures' in str(configured_data_dir) or 'spot' in str(configured_data_dir)):
            base_data_dir = Path(configured_data_dir).parent.parent
        else:
            base_data_dir = Path(configured_data_dir)
        
        local_data = []
        
        # 如果指定了特定类型，只扫描该类型目录
        if trade_type:
            search_dirs = [base_data_dir / exchange / trade_type]
        else:
            # 仅扫描现货与期货目录
            search_dirs = [
                base_data_dir / exchange / 'futures',
                base_data_dir / exchange / 'spot'
            ]
        
        logger.debug(f"基础数据目录: {base_data_dir}")
        logger.debug(f"扫描目录: {[str(d) for d in search_dirs]}")
        
        for data_dir in search_dirs:
            if not data_dir.exists():
                logger.debug(f"目录不存在: {data_dir}")
                continue
                
            logger.debug(f"扫描目录: {data_dir}")
            logger.debug(f"目录内容: {list(data_dir.glob('*.feather'))}")
            
            for file_path in data_dir.glob('*.feather'):
                try:
                    filename = file_path.stem
                    symbol, timeframe_part, data_type = _parse_filename(filename)

                    if timeframe_part == 'unknown':
                        continue

                    data_points, start_str, end_str, _ = _extract_feather_metadata(file_path)
                    
                    data_info = {
                        'exchange': exchange,
                        'symbol': symbol,
                        'timeframe': timeframe_part,
                        'data_type': data_type,
                        'file_path': str(file_path),
                        'file_size': f"{file_path.stat().st_size / 1024 / 1024:.2f} MB",
                        'data_points': data_points,
                        'date_range': {
                            'start': start_str,
                            'end': end_str
                        },
                        'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                    }
                    local_data.append(data_info)
                except Exception as e:
                    logger.warning(f"读取文件失败 {file_path}: {e}")
                    continue
        
        logger.debug(f"返回数据条数: {len(local_data)}")
        return jsonify({'success': True, 'data': local_data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


_local_data_cache = {}
_local_data_cache_lock = threading.Lock()
_LOCAL_DATA_CACHE_TTL = 300

@bp.route('/local-data-cached', methods=['GET'])
def get_local_data_cached():
    global _local_data_cache
    exchange = request.args.get('exchange', 'binance')
    trade_type = request.args.get('trade_type', '')
    force_refresh = request.args.get('force', '0') == '1'
    
    cache_key = f"{exchange}_{trade_type}"
    
    with _local_data_cache_lock:
        if not force_refresh and cache_key in _local_data_cache:
            cached = _local_data_cache[cache_key]
            if time.time() - cached['timestamp'] < _LOCAL_DATA_CACHE_TTL:
                return jsonify({'success': True, 'data': cached['data'], 'cached': True})
    
    try:
        configured_data_dir = current_app.config.get('DATA_DIR', 'data')
        if 'binance' in str(configured_data_dir) and ('futures' in str(configured_data_dir) or 'spot' in str(configured_data_dir)):
            base_data_dir = Path(configured_data_dir).parent.parent
        else:
            base_data_dir = Path(configured_data_dir)
        
        local_data = []
        
        if trade_type:
            search_dirs = [base_data_dir / exchange / trade_type]
        else:
            search_dirs = [
                base_data_dir / exchange / 'futures',
                base_data_dir / exchange / 'spot'
            ]
        
        for data_dir in search_dirs:
            if not data_dir.exists():
                continue
            for file_path in data_dir.glob('*.feather'):
                try:
                    filename = file_path.stem
                    symbol, timeframe_part, data_type = _parse_filename(filename)
                    if timeframe_part == 'unknown':
                        continue
                    data_points, start_str, end_str, _ = _extract_feather_metadata(file_path)
                    data_info = {
                        'exchange': exchange,
                        'symbol': symbol,
                        'timeframe': timeframe_part,
                        'data_type': data_type,
                        'file_path': str(file_path),
                        'file_size': f"{file_path.stat().st_size / 1024 / 1024:.2f} MB",
                        'data_points': data_points,
                        'date_range': {
                            'start': start_str,
                            'end': end_str
                        },
                        'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                    }
                    local_data.append(data_info)
                except Exception as e:
                    logger.warning(f"读取文件失败 {file_path}: {e}")
                    continue
        
        with _local_data_cache_lock:
            _local_data_cache[cache_key] = {
                'data': local_data,
                'timestamp': time.time()
            }
        
        return jsonify({'success': True, 'data': local_data, 'cached': False})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@bp.route('/view-data', methods=['POST'])
def view_data():
    """查看数据文件内容"""
    logger.info("view_data API 开始执行")
    try:
        data = request.get_json()
        logger.debug(f"接收到的请求数据: {data}")
        
        file_path = data.get('file_path')
        limit = data.get('limit')
        offset = data.get('offset', 0)
        logger.debug(f"文件路径: {file_path}, limit: {limit}, offset: {offset}")
        
        is_valid, result = validate_data_path(file_path)
        if not is_valid:
            logger.warning(f"路径验证失败: {result}")
            return jsonify({'success': False, 'error': result})
        
        file_path = str(result)
        logger.info(f"文件路径验证通过: {file_path}")
        
        df = pd.read_feather(file_path)
        logger.info(f"文件读取成功，数据形状: {df.shape}")
        logger.debug(f"列名: {list(df.columns)}")
        
        filename = Path(file_path).stem
        symbol, timeframe, _ = _parse_filename(filename)
        
        # 一次遍历完成列匹配，避免多次循环
        if 'date' not in df.columns and df.index.name:
            df = df.reset_index()

        time_col = None
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['time', 'date', 'timestamp', 'datetime', 'open_time', 'opentime']:
                time_col = col
                break
        if time_col is None:
            for col in df.columns:
                if any(k in col.lower() for k in ['time', 'date', 'timestamp', 'datetime']):
                    time_col = col
                    break

        candidates = {
            'open': ['open', 'opn', 'op', 'o'],
            'high': ['high', 'hi', 'h'],
            'low': ['low', 'lo', 'l'],
            'close': ['close', 'cl', 'c'],
            'volume': ['volume', 'vol', 'v']
        }
        ohlcv_cols = {}
        scores = {}

        def _match_score(col_name, token_list):
            lower = col_name.lower()
            for idx, token in enumerate(token_list):
                if lower == token:
                    return idx
            for idx, token in enumerate(token_list):
                if lower.startswith(token):
                    return idx + 10
            for idx, token in enumerate(token_list):
                if token in lower:
                    return idx + 20
            return None

        for col in df.columns:
            for field, token_list in candidates.items():
                score = _match_score(col, token_list)
                if score is None:
                    continue
                if field not in scores or score < scores[field]:
                    scores[field] = score
                    ohlcv_cols[field] = col

        if time_col and time_col in df.columns:
            ts_series = _to_datetime_series(df[time_col])
            if hasattr(ts_series.dt, 'tz') and ts_series.dt.tz is not None:
                ts_series = ts_series.dt.tz_convert('UTC')
            else:
                ts_series = ts_series.dt.tz_localize('UTC')
            timestamp_values = [ts.isoformat() if pd.notna(ts) else None for ts in ts_series]
        else:
            timestamp_values = [None] * len(df)

        result_df = pd.DataFrame({'timestamp': timestamp_values})
        for field in ['open', 'high', 'low', 'close', 'volume']:
            col_name = ohlcv_cols.get(field)
            if col_name and col_name in df.columns:
                result_df[field] = pd.to_numeric(df[col_name], errors='coerce')
            else:
                result_df[field] = pd.NA

        result_df = result_df.where(pd.notna(result_df), None)
        result_df = result_df.sort_values(
            by='timestamp',
            key=lambda s: s.fillna('9999-12-31T23:59:59+00:00')
        )
        total_count = len(result_df)
        if limit is not None:
            try:
                limit = int(limit)
                offset = int(offset)
                result_df = result_df.iloc[offset:offset + limit]
            except (ValueError, TypeError):
                pass
        ohlcv_data = result_df.to_dict('records')
        
        # 准备返回数据
        logger.debug("准备返回 view_data 数据")
        result_data = {
            'symbol': symbol,
            'timeframe': timeframe,
            'file_path': file_path,
            'file_size': f"{Path(file_path).stat().st_size / 1024 / 1024:.2f} MB",
            'last_modified': datetime.fromtimestamp(Path(file_path).stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'ohlcv_data': ohlcv_data,
            'total_count': total_count,
            'columns_found': list(df.columns),
            'ohlcv_columns_mapped': ohlcv_cols
        }
        
        logger.debug(
            f"view_data 返回摘要: symbol={result_data['symbol']}, timeframe={result_data['timeframe']}, "
            f"rows={len(result_data['ohlcv_data'])}, mapped={result_data['ohlcv_columns_mapped']}"
        )
        logger.info("view_data 执行成功")
        return jsonify({'success': True, 'data': result_data})
        
    except Exception as e:
        import traceback
        logger.error(f"查看数据失败: {e}")
        logger.debug(traceback.format_exc())
        return jsonify({'success': False, 'error': f'查看数据失败: {str(e)}'})

@bp.route('/download', methods=['POST'])
def start_download():
    """开始下载数据"""
    try:
        data = request.get_json()
        exchange = data.get('exchange')
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        trade_type = data.get('trade_type', 'spot')  # 默认为现货
        
        # 生成下载ID
        download_id = f"{exchange}_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 初始化下载状态到全局存储
        with DOWNLOADS_LOCK:
            DOWNLOADS[download_id] = {
                'id': download_id,
            'exchange': exchange,
            'symbol': symbol,
            'timeframe': timeframe,
            'start_date': start_date,
            'end_date': end_date,
                'trade_type': trade_type,
                'status': 'starting',
                'progress': 0,
                'message': '正在初始化下载...',
                'start_time': datetime.now().isoformat()
            }
        
        # 启动后台下载任务
        def download_task():
            try:
                from factor_miner.core.batch_downloader import SmartBatchDownloader
                downloader = SmartBatchDownloader()
                downloader.trade_type = trade_type
                
                with DOWNLOADS_LOCK:
                    if download_id in DOWNLOADS:
                        DOWNLOADS[download_id]['status'] = 'downloading'
                        DOWNLOADS[download_id]['message'] = '正在初始化下载...'
                
                formatted_symbol = format_symbol_for_download(symbol, trade_type)
                
                logger.debug(f"交易对格式转换: {symbol} -> {formatted_symbol} (trade_type: {trade_type})")
                
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                total_days = (end_dt - start_dt).days
                
                with DOWNLOADS_LOCK:
                    if download_id in DOWNLOADS:
                        DOWNLOADS[download_id]['message'] = f'开始分批下载，总天数: {total_days} 天'
                
                result = downloader.download_ohlcv_batch(
                    config_id=None,  # 不使用配置文件，直接创建实例
                    symbol=formatted_symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    trade_type=trade_type,
                    progress_callback=lambda progress, message: update_download_progress(download_id, progress, message)
                )
                
                if result.get('success'):
                    with DOWNLOADS_LOCK:
                        if download_id in DOWNLOADS:
                            DOWNLOADS[download_id]['status'] = 'completed'
                            DOWNLOADS[download_id]['progress'] = 100
                            DOWNLOADS[download_id]['message'] = f'下载完成！共 {result.get("total_records", 0)} 条数据'
                            DOWNLOADS[download_id]['file_path'] = result.get('file_path', '')
                            DOWNLOADS[download_id]['end_time'] = datetime.now().isoformat()
                else:
                    with DOWNLOADS_LOCK:
                        if download_id in DOWNLOADS:
                            DOWNLOADS[download_id]['status'] = 'failed'
                            DOWNLOADS[download_id]['message'] = f'下载失败: {result.get("error", "未知错误")}'
                            DOWNLOADS[download_id]['end_time'] = datetime.now().isoformat()
                    
            except Exception as e:
                with DOWNLOADS_LOCK:
                    if download_id in DOWNLOADS:
                        DOWNLOADS[download_id]['status'] = 'failed'
                        DOWNLOADS[download_id]['message'] = f'下载异常: {str(e)}'
                        DOWNLOADS[download_id]['end_time'] = datetime.now().isoformat()
                logger.error(f"下载任务异常: {e}")
        
        # 启动后台线程
        thread = threading.Thread(target=download_task)
        thread.daemon = True
        thread.start()
        
        with DOWNLOADS_LOCK:
            download_snapshot = dict(DOWNLOADS[download_id])
        return jsonify({'success': True, 'data': download_snapshot})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/downloads', methods=['GET'])
def get_all_downloads():
    """获取所有下载任务的状态"""
    try:
        cleanup_old_downloads()
        with DOWNLOADS_LOCK:
            downloads = [dict(item) for item in DOWNLOADS.values()]
        return jsonify({'success': True, 'data': downloads})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/batch-download', methods=['POST'])
def start_batch_download():
    """开始批量下载数据"""
    try:
        data = request.get_json()
        exchange = data.get('exchange')
        symbols = data.get('symbols', [])  # 交易对列表
        timeframes = data.get('timeframes', [])  # 时间框架列表
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        trade_type = data.get('trade_type', 'spot')  # 默认为现货
        
        if not symbols or not timeframes:
            return jsonify({'success': False, 'error': '请选择交易对和时间框架'})
        
        # 计算总任务数
        total_tasks = len(symbols) * len(timeframes)
        logger.info(f"开始批量下载: {len(symbols)} 个交易对 × {len(timeframes)} 个时间框架 = {total_tasks} 个任务")
        
        # 创建批量下载任务ID
        batch_id = f"batch_{exchange}_{trade_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 初始化批量下载状态
        with DOWNLOADS_LOCK:
            DOWNLOADS[batch_id] = {
                'id': batch_id,
                'type': 'batch',
                'exchange': exchange,
                'symbols': symbols,
                'timeframes': timeframes,
                'start_date': start_date,
                'end_date': end_date,
                'trade_type': trade_type,
                'status': 'starting',
            'progress': 0,
                'message': f'正在初始化批量下载，共 {total_tasks} 个任务...',
                'start_time': datetime.now().isoformat(),
                'total_tasks': total_tasks,
                'completed_tasks': 0,
                'failed_tasks': 0,
                'task_results': []
            }
        
        # 启动后台批量下载任务
        def batch_download_task():
            try:
                import time
                from factor_miner.core.batch_downloader import SmartBatchDownloader
                downloader = SmartBatchDownloader()
                downloader.trade_type = trade_type
                
                _ = downloader.get_cached_exchange()
                logger.debug("交易所实例已缓存，市场数据已加载")
                
                completed_count = 0
                failed_count = 0
                
                # 遍历所有交易对和时间框架组合
                for symbol in symbols:
                    for timeframe in timeframes:
                        task_id = f"{symbol}_{timeframe}"
                        
                        try:
                            # 更新当前任务状态
                            with DOWNLOADS_LOCK:
                                if batch_id in DOWNLOADS:
                                    current_task = completed_count + failed_count + 1
                                    progress = int((current_task - 1) / total_tasks * 100)
                                    DOWNLOADS[batch_id]['progress'] = progress
                                    DOWNLOADS[batch_id]['message'] = f'正在下载 {symbol} {timeframe} ({current_task}/{total_tasks})'
                                    logger.debug(f"批量下载进度更新: {batch_id} -> {progress}% - {symbol} {timeframe}")
                            
                            formatted_symbol = format_symbol_for_download(symbol, trade_type)
                            
                            result = downloader.download_ohlcv_batch(
                                config_id=None,
                                symbol=formatted_symbol,
                                timeframe=timeframe,
                                start_date=start_date,
                                end_date=end_date,
                                trade_type=trade_type,
                                progress_callback=None  # 批量下载时不使用进度回调
                            )
                            
                            # 记录任务结果
                            task_result = {
                                'task_id': task_id,
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'success': result.get('success', False),
                                'message': result.get('message') or result.get('error', '未知状态'),
                                'records': result.get('total_records', 0),
                                'file_path': result.get('file_path', '')
                            }
                            
                            if result.get('success'):
                                completed_count += 1
                                task_result['status'] = 'completed'
                                logger.debug(f"任务完成: {symbol} {timeframe}")
                            else:
                                failed_count += 1
                                task_result['status'] = 'failed'
                                logger.warning(f"任务失败: {symbol} {timeframe} - {result.get('error', '未知错误')}")
                            
                            # 更新批量下载状态
                            with DOWNLOADS_LOCK:
                                if batch_id in DOWNLOADS:
                                    DOWNLOADS[batch_id]['task_results'].append(task_result)
                                    DOWNLOADS[batch_id]['completed_tasks'] = completed_count
                                    DOWNLOADS[batch_id]['failed_tasks'] = failed_count
                                    progress = int((completed_count + failed_count) / total_tasks * 100)
                                    DOWNLOADS[batch_id]['progress'] = progress
                                    logger.debug(f"批量下载任务完成: {batch_id} -> 进度 {progress}% - 完成 {completed_count}, 失败 {failed_count}")
                            
                            time.sleep(0.5)
                            
                        except Exception as e:
                            failed_count += 1
                            logger.error(f"任务异常: {symbol} {timeframe} - {str(e)}")
                            
                            # 记录失败的任务
                            task_result = {
                                'task_id': task_id,
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'success': False,
                                'message': f'任务异常: {str(e)}',
                                'records': 0,
                                'file_path': '',
                                'status': 'failed'
                            }
                            
                            with DOWNLOADS_LOCK:
                                if batch_id in DOWNLOADS:
                                    DOWNLOADS[batch_id]['task_results'].append(task_result)
                                    DOWNLOADS[batch_id]['failed_tasks'] = failed_count
                                    progress = int((completed_count + failed_count) / total_tasks * 100)
                                    DOWNLOADS[batch_id]['progress'] = progress
                                    logger.debug(f"批量下载任务异常: {batch_id} -> 进度 {progress}% - 完成 {completed_count}, 失败 {failed_count}")
                
                # 批量下载完成
                with DOWNLOADS_LOCK:
                    if batch_id in DOWNLOADS:
                        DOWNLOADS[batch_id]['status'] = 'completed'
                        DOWNLOADS[batch_id]['progress'] = 100
                        DOWNLOADS[batch_id]['message'] = f'批量下载完成！成功: {completed_count}, 失败: {failed_count}'
                        DOWNLOADS[batch_id]['end_time'] = datetime.now().isoformat()
                        logger.info(f"批量下载最终完成: {batch_id} -> completed - 成功 {completed_count}, 失败 {failed_count}")
                
                logger.info(f"批量下载完成: 成功 {completed_count}, 失败 {failed_count}")
                
            except Exception as e:
                with DOWNLOADS_LOCK:
                    if batch_id in DOWNLOADS:
                        DOWNLOADS[batch_id]['status'] = 'failed'
                        DOWNLOADS[batch_id]['message'] = f'批量下载异常: {str(e)}'
                        DOWNLOADS[batch_id]['end_time'] = datetime.now().isoformat()
                logger.error(f"批量下载任务异常: {e}")
        
        # 启动后台线程
        thread = threading.Thread(target=batch_download_task)
        thread.daemon = True
        thread.start()
        
        with DOWNLOADS_LOCK:
            batch_snapshot = dict(DOWNLOADS[batch_id])
        return jsonify({'success': True, 'data': batch_snapshot})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def update_download_progress(download_id, progress, message):
    """更新下载进度（线程安全）"""
    with DOWNLOADS_LOCK:
        if download_id in DOWNLOADS:
            DOWNLOADS[download_id]['progress'] = progress
            DOWNLOADS[download_id]['message'] = message

@bp.route('/download-status/<download_id>', methods=['GET'])
def get_download_status(download_id):
    """获取下载状态"""
    try:
        cleanup_old_downloads()
        with DOWNLOADS_LOCK:
            if download_id not in DOWNLOADS:
                return jsonify({
                    'success': False,
                    'error': '下载任务不存在'
                }), 404
            download_info = dict(DOWNLOADS[download_id])
        
        # 计算下载速度（如果正在下载）
        if download_info['status'] == 'downloading' and 'start_time' in download_info:
            start_time = datetime.fromisoformat(download_info['start_time'])
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > 0:
                speed = f"{download_info['progress'] / elapsed * 100:.1f} %/s"
            else:
                speed = "计算中..."
        else:
            speed = "N/A"
        
        status_info = {
            'id': download_id,
            'progress': download_info['progress'],
            'status': download_info['status'],
            'message': download_info['message'],
            'exchange': download_info.get('exchange', ''),
            'symbol': download_info.get('symbol', ''),
            'timeframe': download_info.get('timeframe', ''),
            'start_date': download_info.get('start_date', ''),
            'end_date': download_info.get('end_date', ''),
            'trade_type': download_info.get('trade_type', ''),
            'file_path': download_info.get('file_path', ''),
            'download_speed': speed
        }
        
        return jsonify({'success': True, 'data': status_info})
        
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': f'获取下载状态失败: {str(e)}'
        }), 500

@bp.route('/download-suggestions', methods=['POST'])
def get_download_suggestions():
    """获取下载建议"""
    try:
        data = request.get_json()
        exchange = data.get('exchange')
        trade_type = data.get('trade_type')
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        
        # 获取本地数据
        local_data = []
        
        # 构建数据目录路径
        search_dirs = []
        
        if trade_type == 'futures':
            # 期货类型：检查 DATA_DIR 是否已经是 futures 目录
            base_dir = Path(current_app.config.get('DATA_DIR', 'data'))
            logger.debug(f"DATA_DIR 配置: {base_dir}")
            
            # 如果 DATA_DIR 已经是 futures 目录，直接使用
            if base_dir.name == 'futures':
                search_dirs.append(base_dir)
                logger.debug(f"期货类型：DATA_DIR 已是 futures 目录，使用 {base_dir}")
            else:
                # 否则构建标准路径
                futures_dir = base_dir / exchange / 'futures'
                search_dirs.append(futures_dir)
                logger.debug(f"期货类型：构建标准路径 {futures_dir}")
        else:
            # 现货
            data_dir = Path(current_app.config.get('DATA_DIR', 'data')) / exchange / trade_type
            search_dirs.append(data_dir)
        
        # 在所有相关目录中查找数据
        import re
        # 构建精确的正则匹配模式
        # 文件名格式通常为: SYMBOL_TIMEFRAME-type.feather 或 SYMBOL-TIMEFRAME-type.feather
        pattern = re.compile(rf"^{re.escape(symbol)}[-_]{re.escape(timeframe)}[-_].*\.feather$", re.IGNORECASE)
        
        for search_dir in search_dirs:
            if search_dir.exists():
                logger.debug(f"在目录 {search_dir} 中查找数据文件...")
                logger.debug(f"使用正则匹配模式: {pattern.pattern}")
                
                # 遍历所有feather文件，使用正则精确匹配
                for file_path in search_dir.glob("*.feather"):
                    filename = file_path.name
                    if not pattern.match(filename):
                        continue
                    try:
                        logger.debug(f"找到数据文件: {file_path}")
                        df = pd.read_feather(file_path)
                        # 获取时间范围 - 支持多种时间列名
                        time_col = None
                        for col in df.columns:
                            if col.lower() in ['date', 'time', 'datetime', 'timestamp']:
                                time_col = col
                                break
                        
                        if time_col:
                            start_date = pd.to_datetime(df[time_col].min()).strftime('%Y-%m-%d')
                            end_date = pd.to_datetime(df[time_col].max()).strftime('%Y-%m-%d')
                            local_data.append({
                                'data_type': trade_type,
                                'start_date': start_date,
                                'end_date': end_date,
                                'data_points': len(df),
                                'file_size': f"{file_path.stat().st_size / 1024 / 1024:.2f} MB",
                                'file_path': str(file_path)
                            })
                            logger.debug(f"成功读取数据文件: {file_path}, 数据点数: {len(df)}")
                    except Exception as e:
                        logger.warning(f"读取文件 {file_path} 失败: {str(e)}")
            else:
                logger.debug(f"目录不存在: {search_dir}")
        
        logger.info(f"总共找到 {len(local_data)} 个数据文件")
        
        # 生成下载建议
        recommended_downloads = []
        
        # 如果没有本地数据，建议下载最近一个月的数据
        if not local_data:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            recommended_downloads.append({
                'data_type': trade_type,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'reason': '建议从最近一个月开始下载数据'
            })
        else:
            # 检查数据是否需要更新到最新
            latest_data = max(local_data, key=lambda x: x['end_date'])
            latest_date = datetime.strptime(latest_data['end_date'], '%Y-%m-%d')
            if (datetime.now() - latest_date).days > 1:
                recommended_downloads.append({
                    'data_type': trade_type,
                    'start_date': latest_data['end_date'],
                    'end_date': datetime.now().strftime('%Y-%m-%d'),
                    'reason': '更新数据至最新'
                })
            
            # 检查数据是否有空缺
            sorted_data = sorted(local_data, key=lambda x: x['start_date'])
            for i in range(len(sorted_data) - 1):
                current_end = datetime.strptime(sorted_data[i]['end_date'], '%Y-%m-%d')
                next_start = datetime.strptime(sorted_data[i + 1]['start_date'], '%Y-%m-%d')
                if (next_start - current_end).days > 1:
                    recommended_downloads.append({
                        'data_type': trade_type,
                        'start_date': sorted_data[i]['end_date'],
                        'end_date': sorted_data[i + 1]['start_date'],
                        'reason': '补充数据空缺'
                    })
        
        return jsonify({
            'success': True,
            'data': {
                'exchange': exchange,
                'symbol': symbol,
                'timeframe': timeframe,
                'trade_type': trade_type,
                'existing_data': local_data,
                'recommended_downloads': recommended_downloads
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'获取下载建议失败: {str(e)}'
        }), 500

@bp.route('/data-health', methods=['POST'])
def check_data_health():
    """检查数据健康度"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        
        if not file_path or not os.path.exists(file_path):
            return jsonify({'success': False, 'error': '文件不存在'})
        
        # 导入健康度检查器
        from factor_miner.core.data_health_checker import health_checker
        
        # 读取数据文件
        df = pd.read_feather(file_path)
        
        # 检查数据健康度
        health_report = health_checker.check_data_health(df, timeframe, symbol)
        
        return jsonify({
            'success': True,
            'data': health_report
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/auto-fill-gaps', methods=['POST'])
def auto_fill_gaps():
    """自动补全数据断层"""
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        trade_type = data.get('trade_type', 'futures')
        data_dir = data.get('data_dir')
        
        if not symbol or not timeframe:
            return jsonify({'success': False, 'error': '缺少必要参数'})
        
        # 导入断层补全器
        from factor_miner.core.data_gap_filler import gap_filler
        
        # 执行自动补全
        result = gap_filler.auto_fill_gaps(symbol, timeframe, trade_type, data_dir)
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/scan-gaps', methods=['POST'])
def scan_data_gaps():
    """扫描数据断层"""
    try:
        data = request.get_json()
        data_dir = data.get('data_dir', 'data/binance/futures')
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        
        # 导入断层补全器
        from factor_miner.core.data_gap_filler import gap_filler
        
        # 扫描断层
        gaps = gap_filler.scan_for_gaps(data_dir, symbol, timeframe)
        
        return jsonify({
            'success': True,
            'data': {
                'gaps': gaps,
                'total_gaps': len(gaps),
                'data_dir': data_dir
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/delete-data', methods=['POST'])
def delete_data():
    """删除本地数据"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        
        is_valid, result = validate_data_path(file_path)
        if not is_valid:
            return jsonify({'success': False, 'error': result})
        
        os.remove(result)
        logger.info(f"成功删除数据文件: {file_path}")
        return jsonify({'success': True, 'message': '数据删除成功'})
    except Exception as e:
        logger.error(f"删除数据失败: {e}")
        return jsonify({'success': False, 'error': str(e)}) 
