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
from typing import Dict
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

SYMBOL_CACHE_DIR = Path(__file__).parent.parent.parent / 'data' / 'symbol_cache'
SYMBOL_CACHE_LOCK = threading.Lock()

def _ensure_cache_dir():
    SYMBOL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _read_symbol_cache(cache_type, key):
    _ensure_cache_dir()
    safe_key = key.replace('/', '_').replace('\\', '_')
    cache_path = SYMBOL_CACHE_DIR / f"{cache_type}_{safe_key}.json"
    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"读取缓存文件失败 {cache_path}: {e}")
    return None

def _write_symbol_cache(cache_type, key, data):
    _ensure_cache_dir()
    safe_key = key.replace('/', '_').replace('\\', '_')
    cache_path = SYMBOL_CACHE_DIR / f"{cache_type}_{safe_key}.json"
    try:
        with SYMBOL_CACHE_LOCK:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
        logger.info(f"缓存文件已更新: {cache_path.name}")
    except Exception as e:
        logger.warning(f"写入缓存文件失败 {cache_path}: {e}")

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
    """获取指定交易所的交易对列表（缓存优先，force=1 强制刷新）"""
    try:
        force_refresh = request.args.get('force', '0') == '1'

        if not force_refresh:
            cached = _read_symbol_cache('exchange_symbols', exchange)
            if cached and cached.get('data'):
                logger.info(f"从缓存文件加载 {exchange} 交易对列表")
                return jsonify({
                    'success': True,
                    'data': cached['data'],
                    'cached': True,
                    'cached_at': cached.get('updated_at', '')
                })

        spot_markets = []
        perpetual_markets = []

        logger.info(f"强制刷新 {exchange} 交易对列表")

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

        result_data = {
            'spot': spot_markets,
            'futures': perpetual_markets
        }

        _write_symbol_cache('exchange_symbols', exchange, {
            'data': result_data,
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        return jsonify({
            'success': True,
            'data': result_data,
            'cached': False
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

def _scan_local_data(exchange, trade_type):
    """扫描本地feather文件，返回数据信息列表"""
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

    return local_data

@bp.route('/local-data', methods=['GET'])
def get_local_data():
    """获取本地存储的数据信息（缓存优先，force=1 强制刷新）"""
    try:
        exchange = request.args.get('exchange', 'binance')
        trade_type = request.args.get('trade_type', '')
        force_refresh = request.args.get('force', '0') == '1'

        cache_key = f"{exchange}_{trade_type or 'all'}"

        if not force_refresh:
            cached = _read_symbol_cache('local_data', cache_key)
            if cached and cached.get('data'):
                logger.info(f"从缓存文件加载本地数据: {cache_key}")
                return jsonify({
                    'success': True,
                    'data': cached['data'],
                    'cached': True,
                    'cached_at': cached.get('updated_at', '')
                })

        logger.info(f"强制刷新本地数据: {cache_key}")
        local_data = _scan_local_data(exchange, trade_type)

        _write_symbol_cache('local_data', cache_key, {
            'data': local_data,
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        return jsonify({
            'success': True,
            'data': local_data,
            'cached': False
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


_local_data_cache = {}
_local_data_cache_lock = threading.Lock()
_LOCAL_DATA_CACHE_TTL = 300

@bp.route('/local-data-cached', methods=['GET'])
def get_local_data_cached():
    """获取本地存储的数据信息（缓存优先，force=1 强制刷新）
    保留此路由以兼容旧前端调用，内部逻辑与 local-data 一致
    """
    exchange = request.args.get('exchange', 'binance')
    trade_type = request.args.get('trade_type', '')
    force_refresh = request.args.get('force', '0') == '1'

    cache_key = f"{exchange}_{trade_type or 'all'}"

    if not force_refresh:
        cached = _read_symbol_cache('local_data', cache_key)
        if cached and cached.get('data'):
            return jsonify({
                'success': True,
                'data': cached['data'],
                'cached': True,
                'cached_at': cached.get('updated_at', '')
            })

    try:
        logger.info(f"强制刷新本地数据(cached): {cache_key}")
        local_data = _scan_local_data(exchange, trade_type)

        _write_symbol_cache('local_data', cache_key, {
            'data': local_data,
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        return jsonify({
            'success': True,
            'data': local_data,
            'cached': False
        })
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


@bp.route('/retry/<download_id>', methods=['POST'])
def retry_batch_download(download_id):
    """重试批量下载中失败的子任务。"""
    try:
        with DOWNLOADS_LOCK:
            task = DOWNLOADS.get(download_id)
            if task is None:
                return jsonify({'success': False, 'error': '任务不存在'}), 404
            if task.get('type') != 'batch':
                return jsonify({'success': False, 'error': '仅支持重试批量下载任务'})
            if task.get('status') not in ('completed', 'failed'):
                return jsonify({'success': False, 'error': '当前任务状态不支持重试'})

            task_results = task.get('task_results', [])
            failed_items = [r for r in task_results if r.get('status') == 'failed']
            if not failed_items:
                return jsonify({'success': False, 'error': '没有失败的子任务可重试'})

            retry_symbols = [r['symbol'] for r in failed_items]
            retry_timeframes = list(set(r['timeframe'] for r in failed_items))
            exchange = task.get('exchange', 'binance')
            trade_type = task.get('trade_type', 'futures')
            start_date = task.get('start_date', '')
            end_date = task.get('end_date', '')

            task['status'] = 'downloading'
            task['progress'] = 0
            task['message'] = f'正在重试 {len(failed_items)} 个失败子任务...'
            task['failed_tasks'] = 0
            task['completed_tasks'] = task.get('completed_tasks', 0)
            task['task_results'] = [r for r in task_results if r.get('status') != 'failed']
            task['total_tasks'] = len(task['task_results']) + len(failed_items)

        total_tasks = len(failed_items)

        def retry_task():
            try:
                import time
                from factor_miner.core.batch_downloader import SmartBatchDownloader
                downloader = SmartBatchDownloader()
                downloader.trade_type = trade_type
                _ = downloader.get_cached_exchange()

                completed_count = 0
                failed_count = 0

                for item in failed_items:
                    symbol = item['symbol']
                    timeframe = item['timeframe']
                    try:
                        with DOWNLOADS_LOCK:
                            if download_id in DOWNLOADS:
                                current = completed_count + failed_count + 1
                                progress = int((task['completed_tasks'] + current - 1) / task['total_tasks'] * 100)
                                DOWNLOADS[download_id]['progress'] = progress
                                DOWNLOADS[download_id]['message'] = f'重试下载 {symbol} {timeframe} ({current}/{total_tasks})'

                        formatted_symbol = format_symbol_for_download(symbol, trade_type)
                        result = downloader.download_ohlcv_batch(
                            config_id=None,
                            symbol=formatted_symbol,
                            timeframe=timeframe,
                            start_date=start_date,
                            end_date=end_date,
                            trade_type=trade_type,
                            progress_callback=None,
                        )

                        task_result = {
                            'task_id': f"{symbol}_{timeframe}",
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'success': result.get('success', False),
                            'message': result.get('message') or result.get('error', '未知状态'),
                            'records': result.get('total_records', 0),
                            'file_path': result.get('file_path', ''),
                        }

                        if result.get('success'):
                            completed_count += 1
                            task_result['status'] = 'completed'
                        else:
                            failed_count += 1
                            task_result['status'] = 'failed'

                        with DOWNLOADS_LOCK:
                            if download_id in DOWNLOADS:
                                DOWNLOADS[download_id]['task_results'].append(task_result)
                                DOWNLOADS[download_id]['completed_tasks'] += (1 if task_result['status'] == 'completed' else 0)
                                DOWNLOADS[download_id]['failed_tasks'] += (1 if task_result['status'] == 'failed' else 0)
                                progress = int((DOWNLOADS[download_id]['completed_tasks'] + DOWNLOADS[download_id]['failed_tasks']) / DOWNLOADS[download_id]['total_tasks'] * 100)
                                DOWNLOADS[download_id]['progress'] = progress

                        time.sleep(0.5)

                    except Exception as e:
                        failed_count += 1
                        task_result = {
                            'task_id': f"{symbol}_{timeframe}",
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'success': False,
                            'message': f'任务异常: {str(e)}',
                            'records': 0,
                            'file_path': '',
                            'status': 'failed',
                        }
                        with DOWNLOADS_LOCK:
                            if download_id in DOWNLOADS:
                                DOWNLOADS[download_id]['task_results'].append(task_result)
                                DOWNLOADS[download_id]['failed_tasks'] += 1
                                progress = int((DOWNLOADS[download_id]['completed_tasks'] + DOWNLOADS[download_id]['failed_tasks']) / DOWNLOADS[download_id]['total_tasks'] * 100)
                                DOWNLOADS[download_id]['progress'] = progress

                with DOWNLOADS_LOCK:
                    if download_id in DOWNLOADS:
                        final_failed = DOWNLOADS[download_id]['failed_tasks']
                        final_completed = DOWNLOADS[download_id]['completed_tasks']
                        DOWNLOADS[download_id]['status'] = 'completed' if final_failed == 0 else 'completed'
                        DOWNLOADS[download_id]['progress'] = 100
                        DOWNLOADS[download_id]['message'] = f'重试完成！成功: {final_completed}, 失败: {final_failed}'
                        DOWNLOADS[download_id]['end_time'] = datetime.now().isoformat()

            except Exception as e:
                with DOWNLOADS_LOCK:
                    if download_id in DOWNLOADS:
                        DOWNLOADS[download_id]['status'] = 'failed'
                        DOWNLOADS[download_id]['message'] = f'重试异常: {str(e)}'
                        DOWNLOADS[download_id]['end_time'] = datetime.now().isoformat()

        thread = threading.Thread(target=retry_task)
        thread.daemon = True
        thread.start()

        return jsonify({'success': True, 'data': {'retried_tasks': len(failed_items)}})
    except Exception as e:
        logger.exception('retry batch download 异常')
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

        all_local_data = _scan_local_data(exchange, trade_type)

        for item in all_local_data:
            if item['symbol'] == symbol and (not timeframe or item['timeframe'] == timeframe):
                local_data.append({
                    'data_type': item.get('data_type', trade_type),
                    'start_date': item['date_range']['start'],
                    'end_date': item['date_range']['end'],
                    'data_points': item['data_points'],
                    'file_size': item['file_size'],
                    'file_path': item['file_path']
                })

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


# =====================================================================
# 额外数据下载（OI / LSR / Funding / Mark-Index）—— v4
#
# 只支持币安 U 本位合约（trade_type=futures）。每个 symbol 下载时按所选
# data_types 分别启动子任务，整体进度取各子任务平均值。结果落盘到：
#   data/binance/futures_metrics/      （OI + LSR + taker_buy_ratio_metrics）
#   data/binance/futures_funding/       （资金费率）
#   data/binance/futures_markprice/     （Mark K 线）
#   data/binance/futures_indexprice/    （Index K 线）
#
# 对应因子目录：factorlib/derivatives/, factorlib/funding/。
# =====================================================================

EXTRA_DOWNLOADS = {}
EXTRA_DOWNLOADS_LOCK = threading.Lock()

_EXTRA_SUPPORTED_TYPES = (
    'metrics',
    'funding',
    'mark',
    'index',
    'liquidations',
    'macro',
    'sentiment',
)
_EXTRA_TYPE_LABELS = {
    'metrics': '持仓量/多空比/主动买入占比',
    'funding': '资金费率',
    'mark': 'Mark Price K 线',
    'index': 'Index Price K 线',
    'liquidations': '大额清算（Liquidations）',
    'macro': '宏观与货币因子（DXY/SPX/IXIC/Gold/10Y）',
    'sentiment': '情绪与波动率因子（VIX/稳定币购买力）',
}


def _extra_storage_root():
    return Path(__file__).parent.parent.parent / 'data' / 'binance'


def _extra_update(task_id: str, **fields):
    with EXTRA_DOWNLOADS_LOCK:
        task = EXTRA_DOWNLOADS.get(task_id)
        if task is None:
            return
        task.update(fields)


def _extra_update_subtype(task_id: str, data_type: str, **fields):
    """更新子数据类型的状态并重算整体 progress。"""
    with EXTRA_DOWNLOADS_LOCK:
        task = EXTRA_DOWNLOADS.get(task_id)
        if task is None:
            return
        subtasks = task.setdefault('subtasks', {})
        sub = subtasks.setdefault(data_type, {})
        sub.update(fields)
        # 整体 progress = 各子任务 progress 平均
        if subtasks:
            progs = [int(s.get('progress', 0) or 0) for s in subtasks.values()]
            task['progress'] = int(sum(progs) / len(progs))


def _extra_normalize_symbol_to_ccxt(symbol: str) -> str:
    """
    前端传过来的可能是 BTC_USDT / BTC_USDT_USDT / BTCUSDT / BTC/USDT。
    统一规整成 ccxt 统一符号 BTC/USDT:USDT 供 ExtraDataDownloader 使用。
    """
    if not symbol:
        return symbol
    s = symbol.strip().upper()
    if '/' in s:
        return s if ':' in s else (s + ':USDT' if s.endswith('/USDT') else s)
    if '_' in s:
        parts = [p for p in s.split('_') if p]
        base = parts[0] if parts else s
        quote = parts[1] if len(parts) >= 2 else 'USDT'
        settle = parts[2] if len(parts) >= 3 else quote
        return f"{base}/{quote}:{settle}"
    if s.endswith('USDT'):
        base = s[:-4]
        return f"{base}/USDT:USDT"
    return s


def _run_single_extra_task(
    task_id: str,
    ccxt_symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    data_types: list,
):
    """单 symbol 多 data_type 的下载主循环（在后台线程里跑）。"""
    from factor_miner.core.extra_data_downloader import get_extra_downloader

    downloader = get_extra_downloader()
    _extra_update(task_id, status='downloading', message='开始下载额外数据...')

    results = {}
    for dt in data_types:
        label = _EXTRA_TYPE_LABELS.get(dt, dt)
        _extra_update_subtype(
            task_id, dt, status='downloading', progress=0,
            message=f'开始下载 {label} ...',
        )

        def _cb(progress, message, _dt=dt):
            _extra_update_subtype(task_id, _dt, progress=int(progress), message=message)

        try:
            if dt == 'metrics':
                res = downloader.download_metrics(
                    symbol=ccxt_symbol, timeframe=timeframe or '5m',
                    start_date=start_date, end_date=end_date,
                    progress_callback=_cb,
                )
            elif dt == 'funding':
                res = downloader.download_funding_rate(
                    symbol=ccxt_symbol,
                    start_date=start_date, end_date=end_date,
                    progress_callback=_cb,
                )
            elif dt == 'mark':
                res = downloader.download_mark_index_klines(
                    symbol=ccxt_symbol, timeframe=timeframe or '1h',
                    start_date=start_date, end_date=end_date, kind='mark',
                    progress_callback=_cb,
                )
            elif dt == 'index':
                res = downloader.download_mark_index_klines(
                    symbol=ccxt_symbol, timeframe=timeframe or '1h',
                    start_date=start_date, end_date=end_date, kind='index',
                    progress_callback=_cb,
                )
            elif dt == 'liquidations':
                res = downloader.download_liquidations(
                    symbol=ccxt_symbol, timeframe=timeframe or '1h',
                    start_date=start_date, end_date=end_date,
                    progress_callback=_cb,
                )
            elif dt == 'macro':
                res = downloader.download_macro_factors(
                    symbol=ccxt_symbol, timeframe=timeframe or '1h',
                    start_date=start_date, end_date=end_date,
                    progress_callback=_cb,
                )
            elif dt == 'sentiment':
                res = downloader.download_sentiment_factors(
                    symbol=ccxt_symbol, timeframe=timeframe or '1h',
                    start_date=start_date, end_date=end_date,
                    progress_callback=_cb,
                )
            else:
                res = {'success': False, 'error': f'不支持的数据类型: {dt}'}
        except Exception as e:
            logger.exception(f'[extra-download] {dt} 任务失败')
            res = {'success': False, 'error': str(e)}

        results[dt] = res
        if res.get('success'):
            _extra_update_subtype(
                task_id, dt, status='completed', progress=100,
                message=f'{label} 完成，共 {res.get("data_points", 0)} 条',
                file_path=res.get('file_path'),
                data_points=res.get('data_points', 0),
            )
        else:
            _extra_update_subtype(
                task_id, dt, status='failed', progress=100,
                message=f'{label} 失败: {res.get("error", "未知错误")}',
            )

    all_ok = all(r.get('success') for r in results.values())
    _extra_update(
        task_id,
        status='completed' if all_ok else 'partial_failed',
        progress=100,
        message='全部完成' if all_ok else '部分子任务失败，请查看子任务错误信息',
        end_time=datetime.now().isoformat(),
        results=results,
    )


@bp.route('/extra/download', methods=['POST'])
def start_extra_download():
    """单个 symbol 的额外数据下载（可多选 data_types）。macro/sentiment 为全局数据，无需 symbol。"""
    try:
        payload = request.get_json() or {}
        symbol = payload.get('symbol') or ''
        timeframe = payload.get('timeframe') or '1h'
        start_date = payload.get('start_date')
        end_date = payload.get('end_date')
        data_types = payload.get('data_types') or []

        data_types = [dt for dt in data_types if dt in _EXTRA_SUPPORTED_TYPES]
        if not data_types:
            return jsonify({'success': False, 'error': '未选择任何数据类型'})

        global_types = {'macro', 'sentiment'}
        per_symbol_types = [dt for dt in data_types if dt not in global_types]
        has_global = bool(set(data_types) & global_types)

        if per_symbol_types and not symbol:
            return jsonify({'success': False, 'error': f'缺少 symbol（{per_symbol_types} 需要 symbol）'})

        ccxt_symbol = _extra_normalize_symbol_to_ccxt(symbol) if symbol else ''
        task_id = f"extra_{symbol or 'global'}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        with EXTRA_DOWNLOADS_LOCK:
            EXTRA_DOWNLOADS[task_id] = {
                'id': task_id,
                'kind': 'single',
                'symbol': symbol or 'GLOBAL',
                'ccxt_symbol': ccxt_symbol,
                'timeframe': timeframe,
                'start_date': start_date,
                'end_date': end_date,
                'data_types': list(data_types),
                'status': 'starting',
                'progress': 0,
                'message': '正在初始化下载...',
                'start_time': datetime.now().isoformat(),
                'subtasks': {dt: {'status': 'pending', 'progress': 0} for dt in data_types},
            }

        thread = threading.Thread(
            target=_run_single_extra_task,
            args=(task_id, ccxt_symbol, timeframe, start_date, end_date, list(data_types)),
            daemon=True,
        )
        thread.start()

        with EXTRA_DOWNLOADS_LOCK:
            snapshot = dict(EXTRA_DOWNLOADS[task_id])
        return jsonify({'success': True, 'data': snapshot})
    except Exception as e:
        logger.exception('extra/download 异常')
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/extra/batch-download', methods=['POST'])
def start_extra_batch_download():
    """多 symbol 的额外数据下载。每个 symbol 生成一个独立 task_id。"""
    try:
        payload = request.get_json() or {}
        symbols = payload.get('symbols') or []
        timeframe = payload.get('timeframe') or '1h'
        start_date = payload.get('start_date')
        end_date = payload.get('end_date')
        data_types = payload.get('data_types') or []

        if not symbols:
            return jsonify({'success': False, 'error': '未选择任何 symbol'})
        if not start_date or not end_date:
            return jsonify({'success': False, 'error': '缺少 start_date/end_date'})
        data_types = [dt for dt in data_types if dt in _EXTRA_SUPPORTED_TYPES]
        if not data_types:
            return jsonify({'success': False, 'error': '未选择任何数据类型'})

        global_types = {'macro', 'sentiment'}
        per_symbol_types = [dt for dt in data_types if dt not in global_types]
        has_global = bool(set(data_types) & global_types)

        batch_id = f"extra_batch_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        task_ids = []

        if has_global:
            global_task_id = f"{batch_id}_global"
            with EXTRA_DOWNLOADS_LOCK:
                EXTRA_DOWNLOADS[global_task_id] = {
                    'id': global_task_id,
                    'kind': 'batch-child',
                    'batch_id': batch_id,
                    'symbol': 'GLOBAL',
                    'ccxt_symbol': '',
                    'timeframe': timeframe,
                    'start_date': start_date,
                    'end_date': end_date,
                    'data_types': sorted(set(data_types) & global_types),
                    'status': 'starting',
                    'progress': 0,
                    'message': '等待执行全局数据...',
                    'start_time': datetime.now().isoformat(),
                    'subtasks': {dt: {'status': 'pending', 'progress': 0} for dt in sorted(set(data_types) & global_types)},
                }
            task_ids.append(global_task_id)
            thread = threading.Thread(
                target=_run_single_extra_task,
                args=(global_task_id, '', timeframe, start_date, end_date, sorted(set(data_types) & global_types)),
                daemon=True,
            )
            thread.start()

        for i, sym in enumerate(symbols):
            ccxt_symbol = _extra_normalize_symbol_to_ccxt(sym)
            sym_types = per_symbol_types
            if not sym_types:
                continue
            task_id = f"{batch_id}_{i}_{sym}"
            with EXTRA_DOWNLOADS_LOCK:
                EXTRA_DOWNLOADS[task_id] = {
                    'id': task_id,
                    'kind': 'batch-child',
                    'batch_id': batch_id,
                    'symbol': sym,
                    'ccxt_symbol': ccxt_symbol,
                    'timeframe': timeframe,
                    'start_date': start_date,
                    'end_date': end_date,
                    'data_types': list(sym_types),
                    'status': 'starting',
                    'progress': 0,
                    'message': '等待执行...',
                    'start_time': datetime.now().isoformat(),
                    'subtasks': {dt: {'status': 'pending', 'progress': 0} for dt in sym_types},
                }
            task_ids.append(task_id)

            thread = threading.Thread(
                target=_run_single_extra_task,
                args=(task_id, ccxt_symbol, timeframe, start_date, end_date, list(sym_types)),
                daemon=True,
            )
            thread.start()

        return jsonify({
            'success': True,
            'data': {
                'batch_id': batch_id,
                'task_ids': task_ids,
                'total_tasks': len(task_ids),
            },
        })
    except Exception as e:
        logger.exception('extra/batch-download 异常')
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/extra/download-status/<task_id>', methods=['GET'])
def get_extra_download_status(task_id):
    """获取单个额外数据下载任务的状态。"""
    try:
        with EXTRA_DOWNLOADS_LOCK:
            task = EXTRA_DOWNLOADS.get(task_id)
            if task is None:
                return jsonify({'success': False, 'error': '任务不存在'}), 404
            snapshot = dict(task)
        return jsonify({'success': True, 'data': snapshot})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/extra/downloads', methods=['GET'])
def list_extra_downloads():
    """列出所有额外数据下载任务（可按 batch_id 过滤）。"""
    try:
        batch_id = request.args.get('batch_id')
        with EXTRA_DOWNLOADS_LOCK:
            items = [dict(v) for v in EXTRA_DOWNLOADS.values()]
        if batch_id:
            items = [it for it in items if it.get('batch_id') == batch_id]
        items.sort(key=lambda x: x.get('start_time', ''), reverse=True)
        return jsonify({'success': True, 'data': items})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/extra/retry/<task_id>', methods=['POST'])
def retry_extra_download(task_id):
    """重试指定任务中失败的子任务。"""
    try:
        with EXTRA_DOWNLOADS_LOCK:
            task = EXTRA_DOWNLOADS.get(task_id)
            if task is None:
                return jsonify({'success': False, 'error': '任务不存在'}), 404
            if task.get('status') not in ('partial_failed', 'failed'):
                return jsonify({'success': False, 'error': '当前任务状态不支持重试'})
            ccxt_symbol = task.get('ccxt_symbol', '')
            timeframe = task.get('timeframe', '1h')
            start_date = task.get('start_date', '')
            end_date = task.get('end_date', '')
            subtasks = task.get('subtasks', {})
            failed_types = [dt for dt, s in subtasks.items() if s.get('status') == 'failed']
            if not failed_types:
                return jsonify({'success': False, 'error': '没有失败的子任务可重试'})
            for dt in failed_types:
                task['subtasks'][dt] = {'status': 'pending', 'progress': 0}
            task['status'] = 'downloading'
            task['progress'] = 0
            task['message'] = f'正在重试 {len(failed_types)} 个失败子任务...'

        thread = threading.Thread(
            target=_run_single_extra_task,
            args=(task_id, ccxt_symbol, timeframe, start_date, end_date, failed_types),
            daemon=True,
        )
        thread.start()

        return jsonify({'success': True, 'data': {'retried_types': failed_types}})
    except Exception as e:
        logger.exception('extra/retry 异常')
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/extra/inventory', methods=['GET'])
def list_extra_inventory():
    """
    列出本地已下载的额外数据文件清单，按 symbol 聚合。

    查询参数 trade_type（目前仅 futures 支持这些数据）。
    """
    try:
        root = _extra_storage_root()
        sub_dirs = {
            'metrics': root / 'futures_metrics',
            'funding': root / 'futures_funding',
            'mark': root / 'futures_markprice',
            'index': root / 'futures_indexprice',
            'liquidations': root / 'futures_liquidations',
            'macro': root / 'futures_macro',
            'sentiment': root / 'futures_sentiment',
        }
        global_types = {'macro', 'sentiment'}
        global_items = []
        inventory: Dict = {}
        for kind, d in sub_dirs.items():
            if not d.exists():
                continue
            for f in d.glob('*.feather'):
                stem = f.stem
                try:
                    size_kb = round(f.stat().st_size / 1024, 1)
                except Exception:
                    size_kb = None

                if kind in global_types:
                    parts = stem.split('-')
                    tf = parts[0] if len(parts) >= 2 else None
                    global_items.append({
                        'data_type': kind,
                        'label': _EXTRA_TYPE_LABELS.get(kind, kind),
                        'timeframe': tf,
                        'file': str(f),
                        'size_kb': size_kb,
                    })
                    continue

                parts = stem.split('-')
                if kind == 'funding':
                    safe_sym = '-'.join(parts[:-1]) if len(parts) >= 2 else parts[0]
                    tf = None
                else:
                    if len(parts) >= 3:
                        tf = parts[-2]
                        safe_sym = '-'.join(parts[:-2])
                    else:
                        tf = None
                        safe_sym = stem
                entry = inventory.setdefault(safe_sym, {
                    'symbol': safe_sym,
                    'items': [],
                })
                entry['items'].append({
                    'data_type': kind,
                    'label': _EXTRA_TYPE_LABELS.get(kind, kind),
                    'timeframe': tf,
                    'file': str(f),
                    'size_kb': size_kb,
                })

        items = list(inventory.values())
        items.sort(key=lambda x: x['symbol'])
        return jsonify({
            'success': True,
            'data': {
                'types': [
                    {'id': k, 'label': v, 'dir': str(sub_dirs[k])}
                    for k, v in _EXTRA_TYPE_LABELS.items()
                ],
                'symbols': items,
                'global': global_items,
            },
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/extra/supported-types', methods=['GET'])
def get_extra_supported_types():
    """前端下拉/复选框初始化用。"""
    return jsonify({
        'success': True,
        'data': [
            {'id': k, 'label': _EXTRA_TYPE_LABELS[k]}
            for k in _EXTRA_SUPPORTED_TYPES
        ],
    })


UNIVERSE_STABLECOINS = {
    "USDT", "USDC", "USDE", "BUSD", "DAI", "TUSD", "FDUSD",
    "USDP", "GUSD", "USDD", "LUSD", "FRAX", "USD1",
}

UNIVERSE_BLACKLIST = {
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "FDUSD",
    "WBTC", "WETH", "STETH", "RETH", "BFUSD", "PAXG",
}

UNIVERSE_SYMBOL_ALIASES = {
    "RENDER": "RNDR",
    "THE": "TON",
    "THE-OPEN-NETWORK": "TON",
    "BINANCE-PEG-BSC-USD": "BUSD",
    "BINANCECOIN": "BNB",
    "BINANCE-USD": "BUSD",
    "POLYGON-ECOSYSTEM-TOKEN": "POL",
    "MATIC": "POL",
    "POLYGON": "POL",
    "FANTOM": "FTM",
    "CHAINLINK": "LINK",
    "UNISWAP": "UNI",
    "AVALANCHE": "AVAX",
    "COSMOS": "ATOM",
    "POLKADOT": "DOT",
    "NEAR-PROTOCOL": "NEAR",
    "FLOW": "FLOW",
    "ALGORAND": "ALGO",
    "FILECOIN": "FIL",
    "QUANT": "QNT",
    "ARBITRUM": "ARB",
    "OPTIMISM": "OP",
    "STACKS": "STX",
    "CELESTIA": "TIA",
    "SEI": "SEI",
    "SUI": "SUI",
    "APTOS": "APT",
    "STARKNET": "STRK",
    "MANTLE": "MNT",
    "WORLDCOIN": "WLD",
    "JUPITER": "JUP",
    "PENDLE": "PENDLE",
    "BONK": "BONK",
    "DOGECOIN": "DOGE",
    "SHIBA-INU": "SHIB",
    "BITCOIN-CASH": "BCH",
    "BITCOIN-SV": "BSV",
    "LITECOIN": "LTC",
    "ETHEREUM-CLASSIC": "ETC",
    "MONERO": "XMR",
    "ZCASH": "ZEC",
    "DASH": "DASH",
    "DECENTRALAND": "MANA",
    "SANDBOX": "SAND",
    "AXIE-INFINITY": "AXS",
    "AAVE": "AAVE",
    "MAKER": "MKR",
    "COMPOUND": "COMP",
    "CURVE": "CRV",
    "SYNTHETIX": "SNX",
    "1INCH": "1INCH",
    "ENJINCOIN": "ENJ",
    "BASIC-ATTENTION-TOKEN": "BAT",
    "HOLORCHAIN": "HOT",
    "IOTA": "IOTA",
    "ZILLIQA": "ZIL",
    "ICON": "ICX",
    "VECHAIN": "VET",
    "THETA": "THETA",
    "HEDERA": "HBAR",
    "KAVA": "KAVA",
    "TEZOS": "XTZ",
    "ELROND": "EGLD",
    "HARMONY": "ONE",
    "KUSAMA": "KSM",
    "OCEAN-PROTOCOL": "OCEAN",
    "REN": "REN",
    "BALANCER": "BAL",
    "UNSTOPPABLE-DOMAINS": "UD",
    "PEPE": "PEPE",
    "FLOKI": "FLOKI",
    "BRETT": "BRETT",
    "TURBO": "TURBO",
    "MEME": "MEME",
    "ORDI": "ORDI",
    "SATS": "SATS",
    "RATS": "RATS",
    "BLUR": "BLUR",
    "LOOKS": "LOOKS",
    "ENS": "ENS",
    "GAS": "GAS",
    "LEVER": "LEVER",
    "HOOK": "HOOK",
    "MAGIC": "MAGIC",
    "GMX": "GMX",
    "DYDX": "DYDX",
    "SNX": "SNX",
    "PERP": "PERP",
    "GNO": "GNO",
    "RPL": "RPL",
    "LDO": "LDO",
    "FXS": "FXS",
    "RDN": "RDN",
    "WOO": "WOO",
    "AGLD": "AGLD",
    "PRIME": "PRIME",
    "HIGH": "HIGH",
    "NMR": "NMR",
    "SUPER": "SUPER",
    "CYBER": "CYBER",
    "ARKHAM": "ARKM",
    "WLD": "WLD",
    "PIXEL": "PIXEL",
    "PORTAL": "PORTAL",
    "AEVO": "AEVO",
    "ENA": "ENA",
    "ETHFI": "ETHFI",
    "W": "W",
    "OMNI": "OMNI",
    "REZ": "REZ",
    "SAGA": "SAGA",
    "TAO": "TAO",
    "ONDO": "ONDO",
    "ALT": "ALT",
    "MANTA": "MANTA",
    "DYM": "DYM",
    "XAI": "XAI",
    "JTO": "JTO",
}

_UNIVERSE_CACHE = {}
_UNIVERSE_CACHE_LOCK = threading.Lock()
_UNIVERSE_CACHE_TTL = 3600


@bp.route('/universe-filter', methods=['POST'])
def universe_filter():
    """
    币种池筛选：参考 CoinGecko 市值 + Binance 上市时间/成交额 过滤，
    返回筛选后的币种列表，可直接适配截面评估的币种列表导入。

    请求体 JSON:
      - min_market_cap_rank: int, 市值排名上限（默认 200）
      - min_avg_volume_usd: float, 30天日均成交额下限（默认 1_000_000）
      - min_age_days: int, 上市最少天数（默认 90）
      - exclude_stablecoins: bool, 排除稳定币（默认 True）
      - exclude_blacklist: bool, 排除黑名单（默认 True）
      - trade_type: str, 'futures' | 'spot'（默认 'futures'）
      - exchange: str, 交易所（默认 'binance'）
    """
    try:
        data = request.get_json() or {}
        min_rank = int(data.get('min_market_cap_rank', 200))
        min_vol = float(data.get('min_avg_volume_usd', 1_000_000))
        min_age = int(data.get('min_age_days', 90))
        exclude_stable = data.get('exclude_stablecoins', True)
        exclude_black = data.get('exclude_blacklist', True)
        trade_type = str(data.get('trade_type', 'futures')).lower()
        exchange_id = str(data.get('exchange', 'binance')).lower()

        if isinstance(exclude_stable, str):
            exclude_stable = exclude_stable.strip().lower() in ('1', 'true', 'yes')
        else:
            exclude_stable = bool(exclude_stable)
        if isinstance(exclude_black, str):
            exclude_black = exclude_black.strip().lower() in ('1', 'true', 'yes')
        else:
            exclude_black = bool(exclude_black)

        import requests as _requests

        cache_key = (min_rank, trade_type, exchange_id)
        now_ts = time.time()
        with _UNIVERSE_CACHE_LOCK:
            cached = _UNIVERSE_CACHE.get(cache_key)
            if cached and (now_ts - cached['ts'] < _UNIVERSE_CACHE_TTL):
                coin_df = cached['df']
            else:
                coin_df = None

        if coin_df is None:
            cg_url = "https://api.coingecko.com/api/v3/coins/markets"
            try:
                from config.webui_config import PROXY_CONFIG as _proxy_cfg
                _proxies = {}
                if _proxy_cfg.get('http'):
                    _proxies['http'] = _proxy_cfg['http']
                if _proxy_cfg.get('https'):
                    _proxies['https'] = _proxy_cfg['https']
            except Exception:
                _proxies = {}

            _PER_PAGE = 100
            _total_pages = max(1, (min_rank + _PER_PAGE - 1) // _PER_PAGE)
            _total_pages = min(_total_pages, 5)

            all_coins = []
            _pages_ok = 0
            for _page in range(1, _total_pages + 1):
                cg_params = {
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": _PER_PAGE,
                    "page": _page,
                }
                try:
                    resp = _requests.get(cg_url, params=cg_params, proxies=_proxies or None, timeout=30)
                    if resp.status_code == 429:
                        logger.warning(f"CoinGecko API 第{_page}页被限频(429)，等待后重试")
                        time.sleep(60)
                        resp = _requests.get(cg_url, params=cg_params, proxies=_proxies or None, timeout=30)
                    resp.raise_for_status()
                    page_data = resp.json()
                    if not page_data:
                        break
                    for coin in page_data:
                        all_coins.append({
                            "id": coin.get("id", ""),
                            "symbol": (coin.get("symbol") or "").upper(),
                            "market_cap": coin.get("market_cap", 0),
                            "market_cap_rank": coin.get("market_cap_rank", 9999),
                        })
                    _pages_ok += 1
                    if len(page_data) < _PER_PAGE:
                        break
                    if _page < _total_pages:
                        time.sleep(1.5)
                except Exception as e:
                    logger.warning(f"CoinGecko API 第{_page}页请求失败: {e}")
                    if _page == 1:
                        return jsonify({
                            'success': False,
                            'error': f'CoinGecko API 请求失败: {str(e)}'
                        }), 502
                    break

            logger.info(f"CoinGecko 分页获取: 请求{_total_pages}页, 成功{_pages_ok}页, 共{len(all_coins)}条")
            coin_df = pd.DataFrame(all_coins)
            with _UNIVERSE_CACHE_LOCK:
                _UNIVERSE_CACHE[cache_key] = {'df': coin_df, 'ts': now_ts}

        is_futures = trade_type == 'futures'
        try:
            markets = load_markets_cached(exchange_id, is_futures=is_futures)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'加载交易所市场数据失败: {str(e)}'
            }), 502

        exchange_inst = get_exchange_instance(exchange_id, is_futures=is_futures)
        if not exchange_inst:
            return jsonify({
                'success': False,
                'error': '无法初始化交易所实例'
            }), 502

        usdt_bases = set()
        for mkt_id, mkt_info in markets.items():
            if mkt_info.get('quote') == 'USDT':
                usdt_bases.add(mkt_info.get('base', '').upper())

        def _resolve_symbol(sym_base, coin_id=""):
            _FUTURES_PREFIXES = ["1000", "10000", "100000"]

            def _try_ccxt(base):
                if is_futures:
                    c = f"{base}/USDT:USDT"
                else:
                    c = f"{base}/USDT"
                return c if c in markets else None

            ccxt_sym = _try_ccxt(sym_base)
            if ccxt_sym:
                return sym_base, ccxt_sym

            for prefix in _FUTURES_PREFIXES:
                ccxt_sym = _try_ccxt(f"{prefix}{sym_base}")
                if ccxt_sym:
                    return f"{prefix}{sym_base}", ccxt_sym

            alias = UNIVERSE_SYMBOL_ALIASES.get(sym_base)
            if alias:
                ccxt_sym = _try_ccxt(alias)
                if ccxt_sym:
                    return alias, ccxt_sym
                for prefix in _FUTURES_PREFIXES:
                    ccxt_sym = _try_ccxt(f"{prefix}{alias}")
                    if ccxt_sym:
                        return f"{prefix}{alias}", ccxt_sym

            if coin_id:
                alias2 = UNIVERSE_SYMBOL_ALIASES.get(coin_id.upper())
                if alias2:
                    ccxt_sym = _try_ccxt(alias2)
                    if ccxt_sym:
                        return alias2, ccxt_sym
                    for prefix in _FUTURES_PREFIXES:
                        ccxt_sym = _try_ccxt(f"{prefix}{alias2}")
                        if ccxt_sym:
                            return f"{prefix}{alias2}", ccxt_sym

            for base in usdt_bases:
                if base == sym_base:
                    ccxt_sym = _try_ccxt(base)
                    if ccxt_sym:
                        return base, ccxt_sym

            sym_lower = sym_base.lower()
            coin_id_lower = coin_id.lower() if coin_id else ""
            candidates = []
            for base in usdt_bases:
                base_lower = base.lower()
                if base_lower == sym_lower or base_lower == coin_id_lower:
                    ccxt_sym = _try_ccxt(base)
                    if ccxt_sym:
                        return base, ccxt_sym
                if (sym_lower in base_lower or base_lower in sym_lower) and len(sym_lower) >= 3:
                    candidates.append(base)
                if coin_id_lower and (coin_id_lower in base_lower or base_lower in coin_id_lower) and len(coin_id_lower) >= 3:
                    if base not in candidates:
                        candidates.append(base)

            if len(candidates) == 1:
                base = candidates[0]
                ccxt_sym = _try_ccxt(base)
                if ccxt_sym:
                    return base, ccxt_sym

            return None, None

        results = []
        total_scanned = 0
        skipped_stable = 0
        skipped_black = 0
        skipped_no_market = 0
        skipped_volume = 0
        skipped_age = 0
        no_market_details = []

        for _, row in coin_df.iterrows():
            if row.get('market_cap_rank', 9999) > min_rank:
                continue

            symbol_base = str(row.get("symbol", "")).upper()
            coin_id = str(row.get("id", ""))
            total_scanned += 1

            if exclude_stable and symbol_base in UNIVERSE_STABLECOINS:
                skipped_stable += 1
                continue

            if exclude_black and symbol_base in UNIVERSE_BLACKLIST:
                skipped_black += 1
                continue

            resolved_base, ccxt_sym = _resolve_symbol(symbol_base, coin_id)

            if resolved_base is None:
                skipped_no_market += 1
                no_market_details.append({
                    "symbol": symbol_base,
                    "coin_id": coin_id,
                    "rank": int(row.get('market_cap_rank', 0)),
                })
                continue

            local_sym = f"{resolved_base}_USDT"

            avg_vol = None
            try:
                ohlcv = exchange_inst.fetch_ohlcv(ccxt_sym, timeframe='1d', limit=30)
                if ohlcv and len(ohlcv) > 0:
                    vol_df = pd.DataFrame(
                        ohlcv,
                        columns=["timestamp", "open", "high", "low", "close", "volume"]
                    )
                    vol_df["dollar_volume"] = vol_df["close"] * vol_df["volume"]
                    avg_vol = vol_df["dollar_volume"].mean()
            except Exception:
                avg_vol = None

            if avg_vol is None or avg_vol < min_vol:
                skipped_volume += 1
                continue

            age_days = None
            first_date = None
            try:
                ohlcv_full = exchange_inst.fetch_ohlcv(ccxt_sym, timeframe='1d', limit=1000)
                if ohlcv_full and len(ohlcv_full) > 0:
                    first_ts = ohlcv_full[0][0]
                    first_date = datetime.utcfromtimestamp(first_ts / 1000)
                    age_days = (datetime.utcnow() - first_date).days
            except Exception:
                age_days = None

            if age_days is None or age_days < min_age:
                skipped_age += 1
                continue

            results.append({
                "symbol": local_sym,
                "ccxt_symbol": ccxt_sym,
                "base": symbol_base,
                "market_cap_rank": int(row.get('market_cap_rank', 0)),
                "avg_30d_volume": round(avg_vol, 2) if avg_vol else None,
                "age_days": age_days,
                "first_trade_date": first_date.strftime('%Y-%m-%d') if first_date else None,
            })

        results.sort(key=lambda x: x.get('market_cap_rank', 9999))

        return jsonify({
            'success': True,
            'data': {
                'symbols': results,
                'total_scanned': total_scanned,
                'total_fetched_from_cg': len(coin_df),
                'skipped_stable': skipped_stable,
                'skipped_black': skipped_black,
                'skipped_no_market': skipped_no_market,
                'no_market_details': no_market_details,
                'skipped_volume': skipped_volume,
                'skipped_age': skipped_age,
                'passed': len(results),
            },
            'params': {
                'min_market_cap_rank': min_rank,
                'min_avg_volume_usd': min_vol,
                'min_age_days': min_age,
                'exclude_stablecoins': exclude_stable,
                'exclude_blacklist': exclude_black,
                'trade_type': trade_type,
                'exchange': exchange_id,
            }
        })

    except Exception as e:
        logger.error(f"币种池筛选失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'币种池筛选失败: {str(e)}'
        }), 500
