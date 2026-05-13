"""
实时数据获取器：从交易所 API 直接拉取最新数据用于截面实时扫描。

支持的数据类型：
  - OHLCV K线（ccxt fetch_ohlcv）
  - Metrics（OI / LSR / Taker 比率，Binance fapi/data 端点）
  - Funding Rate（ccxt fetch_funding_rate_history）
  - Mark / Index 价格（ccxt fetch_mark_ohlcv / fetch_index_ohlcv）

设计原则：
  - 轻量级：只拉取最近 N 根 bar，不涉及归档
  - 线程安全：每个调用独立创建 exchange 实例
  - 容错：单个币种/数据源失败不影响其他
  - 复用连接：fetch_realtime_data 内部复用同一个 exchange 实例
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import ccxt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TIMEFRAME_TO_MINUTES = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360,
    "12h": 720, "1d": 1440,
}


def _create_exchange(exchange_id: str = 'binance') -> Optional[ccxt.Exchange]:
    try:
        import os
        http_proxy = os.getenv('HTTP_PROXY')
        https_proxy = os.getenv('HTTPS_PROXY')
        if not http_proxy:
            http_proxy = 'http://127.0.0.1:7897'
        if not https_proxy:
            https_proxy = 'http://127.0.0.1:7897'
        proxies = {}
        if http_proxy:
            proxies['http'] = http_proxy
        if https_proxy:
            proxies['https'] = https_proxy

        options = {
            'enableRateLimit': True,
            'timeout': 30000,
            'proxies': proxies if proxies else None,
            'defaultType': 'swap',
            'options': {
                'defaultSubType': 'linear',
            },
        }
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class(options)
        exchange.load_markets()
        return exchange
    except Exception as e:
        logger.error(f"创建交易所实例失败: {e}")
        return None


def _to_ccxt_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    if "/" in s:
        return s
    if "_" in s:
        parts = [p for p in s.split("_") if p]
        if len(parts) >= 2:
            base = parts[0]
            quote = parts[1]
            settle = parts[2] if len(parts) >= 3 else quote
            return f"{base}/{quote}:{settle}"
        s = "".join(parts)
    if s.endswith("USDT"):
        base = s[:-4]
        return f"{base}/USDT:USDT"
    return s


def _to_fapi_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    if "_" in s:
        parts = [p for p in s.split("_") if p]
        if len(parts) >= 2:
            return f"{parts[0]}{parts[1]}"
        s = "".join(parts)
    s = s.replace("/", "").replace(":", "")
    if s.endswith("USDT"):
        return s
    base = s.split("USDT")[0]
    return f"{base}USDT"


def _fetch_ohlcv_with_exchange(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str = '15m',
    limit: int = 200,
) -> Optional[pd.DataFrame]:
    ccxt_sym = _to_ccxt_symbol(symbol)
    try:
        ohlcv = exchange.fetch_ohlcv(ccxt_sym, timeframe, limit=limit)
        if not ohlcv:
            return None
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('date').drop(columns=['timestamp'])
        df.index = df.index.tz_localize('UTC')
        return df
    except Exception as e:
        logger.warning(f"获取 {symbol} OHLCV 失败: {e}")
        return None


def _fetch_metrics_with_exchange(
    exchange: ccxt.Exchange,
    symbol: str,
    period: str = '5m',
    limit: int = 200,
) -> Optional[pd.DataFrame]:
    fapi_sym = _to_fapi_symbol(symbol)
    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(minutes=limit * 5 + 30)

    specs = [
        ("fapiDataGetOpenInterestHist", {"sumOpenInterest": "open_interest"}, "OI"),
        ("fapiDataGetGlobalLongShortAccountRatio", {"longShortRatio": "lsr_global_account"}, "LSR-global"),
        ("fapiDataGetTopLongShortAccountRatio", {"longShortRatio": "lsr_top_account"}, "LSR-top-acct"),
        ("fapiDataGetTopLongShortPositionRatio", {"longShortRatio": "lsr_top_position"}, "LSR-top-pos"),
        ("fapiDataGetTakerlongshortRatio", {"buySellRatio": "taker_buy_ratio_metrics"}, "Taker-ratio"),
    ]

    dfs = []
    for endpoint, value_keys, tag in specs:
        method = None
        for name in (endpoint, endpoint[0].lower() + endpoint[1:]):
            if hasattr(exchange, name):
                method = getattr(exchange, name)
                break
        if method is None:
            continue

        rows = []
        try:
            res = method({
                "symbol": fapi_sym,
                "period": period,
                "startTime": int(start_dt.timestamp() * 1000),
                "endTime": int(end_dt.timestamp() * 1000),
                "limit": min(limit, 500),
            })
            if res:
                for item in res:
                    ts = item.get("timestamp")
                    if ts is None:
                        continue
                    row = {"date": pd.to_datetime(int(ts), unit="ms")}
                    for src_k, dst_k in value_keys.items():
                        v = item.get(src_k)
                        try:
                            row[dst_k] = float(v) if v is not None else None
                        except Exception:
                            row[dst_k] = None
                    rows.append(row)
        except Exception as e:
            logger.warning(f"[{tag}] {symbol} 获取失败: {e}")

        if rows:
            dfs.append(pd.DataFrame(rows).drop_duplicates(subset=["date"]).sort_values("date"))
        time.sleep(0.1)

    if not dfs:
        return None

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="date", how="outer")
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def _fetch_funding_with_exchange(
    exchange: ccxt.Exchange,
    symbol: str,
    limit: int = 100,
) -> Optional[pd.DataFrame]:
    ccxt_sym = _to_ccxt_symbol(symbol)
    try:
        since = int((datetime.utcnow() - timedelta(days=limit * 8 / 24 + 1)).timestamp() * 1000)
        records = exchange.fetch_funding_rate_history(ccxt_sym, since=since, limit=limit)
        if not records:
            return None

        rows = []
        for r in records:
            ts = r.get('timestamp') or r.get('fundingDatetime')
            if ts is None:
                continue
            try:
                if isinstance(ts, (int, float)):
                    dt = pd.to_datetime(int(ts), unit='ms')
                else:
                    dt = pd.to_datetime(ts)
            except Exception:
                continue
            rate = r.get('fundingRate')
            if rate is not None:
                try:
                    rate = float(rate)
                except Exception:
                    rate = None
            rows.append({"date": dt, "funding_rate": rate})

        if not rows:
            return None
        df = pd.DataFrame(rows).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        logger.warning(f"获取 {symbol} funding 失败: {e}")
        return None


def _fetch_mark_index_with_exchange(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str = '15m',
    limit: int = 200,
) -> Optional[pd.DataFrame]:
    ccxt_sym = _to_ccxt_symbol(symbol)
    result_dfs = []

    for kind, close_col in [("mark", "mark_close"), ("index", "index_close")]:
        try:
            method_name = f"fetch_{kind}_ohlcv"
            method = getattr(exchange, method_name, None)
            if method is None:
                continue
            ohlcv = method(ccxt_sym, timeframe, limit=limit)
            if not ohlcv:
                continue
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', close_col, 'volume'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[['date', close_col]].copy()
            result_dfs.append(df)
        except Exception as e:
            logger.warning(f"获取 {symbol} {kind} 价格失败: {e}")
        time.sleep(0.1)

    if not result_dfs:
        return None

    merged = result_dfs[0]
    for df in result_dfs[1:]:
        merged = merged.merge(df, on="date", how="outer")
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def fetch_realtime_data(
    symbol: str,
    timeframe: str = '15m',
    limit: int = 200,
    include: Optional[List[str]] = None,
    exchange_id: str = 'binance',
    exchange: Optional[ccxt.Exchange] = None,
) -> Optional[pd.DataFrame]:
    """
    一次性获取某个币种的全部实时数据（OHLCV + extras），合并为一个 DataFrame。
    内部复用同一个 exchange 实例，减少连接开销。

    Args:
        symbol: 交易对
        timeframe: K线周期
        limit: 获取bar数量
        include: 额外数据类别，支持 'metrics', 'funding', 'basis'；传空列表 [] 表示不拉取任何 extras
        exchange_id: 交易所ID
        exchange: 可选的共享 ccxt 实例（线程间共享时由调用方管理生命周期）

    Returns:
        DataFrame with DatetimeIndex, 包含 OHLCV 和额外列
    """
    if include is None:
        include = ['metrics', 'funding', 'basis']

    _own_exchange = exchange is None
    if _own_exchange:
        exchange = _create_exchange(exchange_id)
    if exchange is None:
        return None

    try:
        ohlcv_df = _fetch_ohlcv_with_exchange(exchange, symbol, timeframe, limit)
        if ohlcv_df is None or ohlcv_df.empty:
            return None

        if 'metrics' in include:
            metrics_df = _fetch_metrics_with_exchange(exchange, symbol, '5m', limit)
            if metrics_df is not None and not metrics_df.empty:
                ohlcv_df = _left_join_on_date(ohlcv_df, metrics_df)

        if 'funding' in include:
            funding_df = _fetch_funding_with_exchange(exchange, symbol, limit)
            if funding_df is not None and not funding_df.empty:
                ohlcv_df = _left_join_on_date(ohlcv_df, funding_df)

        want_basis = 'basis' in include
        if want_basis or 'mark' in include or 'index' in include:
            mi_df = _fetch_mark_index_with_exchange(exchange, symbol, timeframe, limit)
            if mi_df is not None and not mi_df.empty:
                ohlcv_df = _left_join_on_date(ohlcv_df, mi_df)
            if want_basis and 'index_close' in ohlcv_df.columns:
                if 'mark_close' in ohlcv_df.columns:
                    ohlcv_df['basis'] = (ohlcv_df['mark_close'] - ohlcv_df['index_close']) / ohlcv_df['index_close']
                elif 'close' in ohlcv_df.columns:
                    ohlcv_df['basis'] = (ohlcv_df['close'] - ohlcv_df['index_close']) / ohlcv_df['index_close']

        return ohlcv_df
    finally:
        if _own_exchange and hasattr(exchange, 'close'):
            exchange.close()


def _left_join_on_date(main_df: pd.DataFrame, extra_df: pd.DataFrame) -> pd.DataFrame:
    if extra_df is None or extra_df.empty:
        return main_df
    if main_df is None or main_df.empty:
        return main_df

    def _normalize_dt_index(idx):
        dt_idx = pd.to_datetime(idx, errors='coerce')
        try:
            if getattr(dt_idx, 'tz', None) is not None:
                dt_idx = dt_idx.tz_convert('UTC').tz_localize(None)
        except Exception:
            try:
                dt_idx = dt_idx.tz_localize(None)
            except Exception:
                pass
        return dt_idx

    extra = extra_df.copy()
    if 'date' in extra.columns:
        extra['date'] = pd.to_datetime(extra['date'], errors='coerce')
        extra = extra.set_index('date')

    original_main_index = main_df.index
    main = main_df.copy()
    main.index = _normalize_dt_index(main.index)
    extra.index = _normalize_dt_index(extra.index)

    extra = extra[extra.index.notna()]
    extra = extra[~extra.index.duplicated(keep='last')].sort_index()

    aligned = extra.reindex(main.index, method='ffill')
    for col in aligned.columns:
        main[col] = aligned[col]

    main.index = original_main_index
    return main
