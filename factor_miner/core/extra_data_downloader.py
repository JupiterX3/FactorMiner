"""
额外数据下载器：持仓量(OI) / 多空比(LSR) / 主动买入(Taker) / 资金费率(Funding) / 标记指数价(Mark/Index → Basis)。

设计原则：
- **归档优先**：优先从 `data.binance.vision` 拉取每日/每月历史 ZIP（覆盖从 2020 年起的全量历史，
  仅延迟一天），远端 API 仅用于补齐归档未覆盖的最新 1–2 天。
- 每类数据独立 feather 落盘，互不污染主 OHLCV。
- Metrics 固定落盘 5m（归档本身粒度），需要更粗 tf 时由 `DataLoader.load_with_extras` 做 resample。
- 向下兼容：旧数据不存在额外列不会影响现有链路。

产物目录约定：
    data/binance/futures_metrics/   {SYMBOL}-5m-metrics.feather    # OI + LSR + taker_ratio (5m)
    data/binance/futures_funding/   {SYMBOL}-funding.feather         # 事件级 funding
    data/binance/futures_markprice/ {SYMBOL}-{tf}-mark.feather       # Mark K 线
    data/binance/futures_indexprice/{SYMBOL}-{tf}-index.feather      # Index K 线

归档源（免费、CDN、无鉴权）：
    https://data.binance.vision/data/futures/um/daily/metrics/{SYM}/{SYM}-metrics-YYYY-MM-DD.zip
    https://data.binance.vision/data/futures/um/monthly/fundingRate/{SYM}/{SYM}-fundingRate-YYYY-MM.zip
    https://data.binance.vision/data/futures/um/daily/markPriceKlines/{SYM}/{tf}/{SYM}-markPriceKlines-{tf}-YYYY-MM-DD.zip
    https://data.binance.vision/data/futures/um/daily/indexPriceKlines/{SYM}/{tf}/{SYM}-indexPriceKlines-{tf}-YYYY-MM-DD.zip

实盘一致性：所有列都能通过如下实时接口取到同名字段，用相同的因子函数复现。
    openInterest              → GET /fapi/v1/openInterest
    lsr_*                     → GET /futures/data/*LongShort*Ratio
    taker_buy_ratio_metrics   → GET /futures/data/takerlongshortRatio
    funding_rate              → GET /fapi/v1/fundingRate
    mark_close / index_close  → GET /fapi/v1/markPriceKlines?params={price: mark|index}
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import ccxt
import pandas as pd
import requests

from .data_downloader import DataDownloader

logger = logging.getLogger(__name__)


_METRICS_DIR = Path("data/binance/futures_metrics")
_FUNDING_DIR = Path("data/binance/futures_funding")
_MARK_DIR = Path("data/binance/futures_markprice")
_INDEX_DIR = Path("data/binance/futures_indexprice")
_LIQUIDATIONS_DIR = Path("data/binance/futures_liquidations")
_MACRO_DIR = Path("data/binance/futures_macro")
_SENTIMENT_DIR = Path("data/binance/futures_sentiment")

# 归档（Binance Vision）相关常量
_VISION_BASE = "https://data.binance.vision/data/futures/um"
_VISION_HTTP_TIMEOUT = 30        # 单次下载超时（秒）
_VISION_PARALLEL = 8             # 并发拉取数
_VISION_RETRY = 3                # 单个 URL 重试次数
_VISION_MIN_DATE = datetime(2020, 1, 1)  # 归档开始时间（大致）
_ARCHIVE_LAG_DAYS = 2            # 归档落盘一般 T+1，留 2 天安全区


@dataclass
class ExtraDownloadResult:
    """统一的下载结果结构。"""

    success: bool
    file_path: Optional[str] = None
    data_points: int = 0
    columns: Optional[List[str]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "file_path": self.file_path,
            "data_points": self.data_points,
            "columns": self.columns or [],
            "error": self.error,
        }


class ExtraDataDownloader(DataDownloader):
    """额外数据下载器：归档优先 + 实时 API 增量。"""

    _TIMEFRAME_TO_MINUTES = {
        "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120,
        "4h": 240, "6h": 360, "12h": 720, "1d": 1440,
    }

    # Binance Vision 归档 metrics CSV 的字段映射
    # 归档 5m 粒度，列名来自官方 README；历史版本可能没有表头，因此我们用位置读列。
    _METRICS_COLS_ORDERED = [
        "create_time",
        "symbol",
        "sum_open_interest",
        "sum_open_interest_value",
        "count_toptrader_long_short_ratio",
        "sum_toptrader_long_short_ratio",
        "count_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
    ]
    _METRICS_RENAME = {
        "sum_open_interest": "open_interest",
        "count_long_short_ratio": "lsr_global_account",
        "count_toptrader_long_short_ratio": "lsr_top_account",
        "sum_toptrader_long_short_ratio": "lsr_top_position",
        "sum_taker_long_short_vol_ratio": "taker_buy_ratio_metrics",
    }

    # Binance K 线归档标准 12 列（markPrice / indexPrice / klines 共用）
    _KLINE_COLS_ORDERED = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_volume", "taker_buy_quote_volume", "ignore",
    ]

    def __init__(self):
        super().__init__()
        self.trade_type = "futures"
        self._http_session = requests.Session()
        self._http_session.headers.update({"User-Agent": "FactorMiner/1.0"})

    # ------------------------------------------------------------------
    # 公共工具
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_symbol(symbol: str) -> str:
        return symbol.replace("/", "_").replace(":", "_")

    @classmethod
    def _ensure_dirs(cls) -> None:
        for d in (
            _METRICS_DIR,
            _FUNDING_DIR,
            _MARK_DIR,
            _INDEX_DIR,
            _LIQUIDATIONS_DIR,
            _MACRO_DIR,
            _SENTIMENT_DIR,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def _get_futures_exchange(self) -> Optional[ccxt.Exchange]:
        """拿一个期货口径的 ccxt 实例。"""
        try:
            exchange = self.get_exchange_instance(config_id=None, exchange_id="binance")
            if exchange is not None:
                try:
                    exchange.load_markets()
                    return exchange
                except Exception as e:
                    logger.warning(f"DataDownloader 方式初始化失败，尝试直连 binance: {e}")
        except Exception:
            pass
        try:
            # 兜底：直接初始化 ccxt.binance futures，绕过项目层配置差异
            ex = ccxt.binance({
                "enableRateLimit": True,
                "timeout": 30000,
                "options": {
                    "defaultType": "future",
                    "defaultSubType": "linear",
                },
            })
            ex.load_markets()
            return ex
        except Exception as e:
            logger.error(f"初始化期货交易所失败: {e}")
            return None

    @classmethod
    def _merge_and_save(
        cls, df_new: pd.DataFrame, save_path: Path, key: str = "date"
    ) -> ExtraDownloadResult:
        """把新数据 upsert 到历史 feather。"""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if df_new is None or df_new.empty:
            if save_path.exists():
                df_existing = pd.read_feather(save_path)
                return ExtraDownloadResult(
                    success=True,
                    file_path=str(save_path),
                    data_points=len(df_existing),
                    columns=list(df_existing.columns),
                )
            return ExtraDownloadResult(success=False, error="下载结果为空")

        if key in df_new.columns:
            df_new[key] = pd.to_datetime(df_new[key], errors="coerce")
            if getattr(df_new[key].dt, "tz", None) is not None:
                df_new[key] = df_new[key].dt.tz_localize(None)

        if save_path.exists():
            try:
                df_old = pd.read_feather(save_path)
                if key in df_old.columns:
                    df_old[key] = pd.to_datetime(df_old[key], errors="coerce")
                    if getattr(df_old[key].dt, "tz", None) is not None:
                        df_old[key] = df_old[key].dt.tz_localize(None)
                merged = pd.concat([df_old, df_new], ignore_index=True)
                merged = (
                    merged.drop_duplicates(subset=[key], keep="last")
                    .sort_values(key)
                    .reset_index(drop=True)
                )
                df_save = merged
            except Exception as e:
                logger.warning(f"合并历史文件失败({e})，将用新数据覆盖")
                df_save = df_new
        else:
            df_save = df_new

        tmp = save_path.with_suffix(".tmp")
        df_save.to_feather(tmp)
        tmp.replace(save_path)
        return ExtraDownloadResult(
            success=True,
            file_path=str(save_path),
            data_points=len(df_save),
            columns=list(df_save.columns),
        )

    # ------------------------------------------------------------------
    # 归档工具（data.binance.vision）
    # ------------------------------------------------------------------
    def _http_get_zip(self, url: str) -> Optional[bytes]:
        """
        下载单个 ZIP 字节流。命中 404 视为"当日无归档"（币种未上线/节假日），返回 None；
        其他异常会重试 `_VISION_RETRY` 次。
        """
        for attempt in range(_VISION_RETRY):
            try:
                resp = self._http_session.get(url, timeout=_VISION_HTTP_TIMEOUT)
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                return resp.content
            except requests.RequestException as e:
                if attempt == _VISION_RETRY - 1:
                    logger.warning(f"下载归档失败 {url}: {e}")
                    return None
                time.sleep(0.5 * (attempt + 1))
        return None

    @staticmethod
    def _read_csv_from_zip(blob: bytes, column_names: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """
        解压 ZIP 并把第一个 CSV 读成 DataFrame。归档早期文件无表头，晚期文件有表头，
        统一策略：
          - 若第一行包含字母（认为是表头）→ 用表头；
          - 否则按传入 `column_names` 读；
        未知情况返回 None。
        """
        try:
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_names:
                    return None
                with zf.open(csv_names[0]) as f:
                    raw = f.read()
            text = raw.decode("utf-8", errors="replace")
            first_line = text.split("\n", 1)[0].strip().lower()
            has_header = any(c.isalpha() for c in first_line)
            if has_header:
                df = pd.read_csv(io.StringIO(text))
            else:
                if not column_names:
                    return None
                df = pd.read_csv(io.StringIO(text), header=None, names=column_names)
            return df
        except Exception as e:
            logger.warning(f"解析归档 CSV 失败: {e}")
            return None

    def _fetch_archive_parallel(
        self,
        url_items: List[Dict],
        reader,
        progress_callback=None,
        tag: str = "",
    ) -> List[pd.DataFrame]:
        """
        并发拉取归档 ZIP 并用 `reader(blob, meta) -> DataFrame | None` 解析。

        Args:
            url_items: [{"url": str, "meta": dict}, ...]
            reader: 回调，接受 `(blob: bytes, meta: dict)` 返回 DataFrame 或 None
        """
        frames: List[pd.DataFrame] = []
        total = len(url_items)
        done = 0
        with ThreadPoolExecutor(max_workers=_VISION_PARALLEL) as ex:
            futs = {
                ex.submit(self._http_get_zip, item["url"]): item
                for item in url_items
            }
            for fut in as_completed(futs):
                item = futs[fut]
                blob = fut.result()
                done += 1
                if blob is not None:
                    df = reader(blob, item.get("meta", {}))
                    if df is not None and not df.empty:
                        frames.append(df)
                if progress_callback and total > 0:
                    progress_callback(
                        min(99, int(done * 100 / total)),
                        f"[{tag}] 归档 {done}/{total}，已解析 {len(frames)} 份",
                    )
        return frames

    @staticmethod
    def _daterange(start: datetime, end: datetime):
        """按日切分，闭区间 start、开区间 end。"""
        cur = datetime(start.year, start.month, start.day)
        end = datetime(end.year, end.month, end.day)
        while cur < end:
            yield cur
            cur = cur + timedelta(days=1)

    @staticmethod
    def _monthrange(start: datetime, end: datetime):
        """按月切分（归档 funding 是月粒度）。"""
        cur = datetime(start.year, start.month, 1)
        end_m = datetime(end.year, end.month, 1)
        while cur <= end_m:
            yield cur
            if cur.month == 12:
                cur = datetime(cur.year + 1, 1, 1)
            else:
                cur = datetime(cur.year, cur.month + 1, 1)

    # ------------------------------------------------------------------
    # 1. Metrics：OI + LSR + taker_buy_ratio
    # ------------------------------------------------------------------
    def download_metrics(
        self,
        symbol: str,
        timeframe: str = "5m",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        progress_callback=None,
    ) -> Dict:
        """
        下载衍生品截面指标（OI / LSR / takerLongShortRatio）。

        - 数据源：`/futures/data/openInterestHist` 等 fapi data 端点，免费且最多 500 根/请求、最近 30 天；
        - 粒度：支持 `5m/15m/30m/1h/2h/4h/6h/12h/1d`；
        - 输出列：`open_interest`、`lsr_global_account`、`lsr_top_account`、
                 `lsr_top_position`、`taker_buy_ratio_metrics`。

        Args:
            symbol: 形如 `BTC/USDT:USDT` 或 `BTCUSDT`
            timeframe: 目标 K 线周期
            start_date/end_date: `YYYY-MM-DD`，缺省为最近 30 天
        """
        _ = timeframe  # 兼容旧签名，归档固定 5m
        try:
            self._ensure_dirs()
            fapi_symbol = self._to_fapi_symbol(symbol)
            save_path = _METRICS_DIR / f"{self._safe_symbol(symbol)}-5m-metrics.feather"

            end_dt = (
                datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.utcnow()
            )
            start_dt = (
                datetime.strptime(start_date, "%Y-%m-%d")
                if start_date
                else end_dt - timedelta(days=180)
            )
            if start_dt < _VISION_MIN_DATE:
                start_dt = _VISION_MIN_DATE

            # 归档切线：归档最多到 (今日 - ARCHIVE_LAG) 这一天
            archive_cutoff = datetime.utcnow() - timedelta(days=_ARCHIVE_LAG_DAYS)
            archive_end = min(end_dt, archive_cutoff)
            recent_start = max(start_dt, archive_cutoff)
            url_items = [
                {
                    "url": (
                        f"{_VISION_BASE}/daily/metrics/{fapi_symbol}/"
                        f"{fapi_symbol}-metrics-{d:%Y-%m-%d}.zip"
                    ),
                    "meta": {"date": d},
                }
                for d in self._daterange(start_dt, archive_end)
            ]
            frames: List[pd.DataFrame] = []
            if url_items:
                frames = self._fetch_archive_parallel(
                    url_items,
                    reader=self._parse_metrics_day,
                    progress_callback=progress_callback,
                    tag="metrics-archive",
                )
            df_archive = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

            df_recent = self._fetch_recent_metrics_via_fapi(
                fapi_symbol=fapi_symbol,
                start_dt=recent_start,
                end_dt=end_dt,
                progress_callback=progress_callback,
            )

            parts = [d for d in (df_archive, df_recent) if d is not None and not d.empty]
            if not parts:
                return ExtraDownloadResult(
                    success=False,
                    error="未获取到任何 metrics 数据（归档 + 实时 API 均为空）",
                ).to_dict()
            df_all = pd.concat(parts, ignore_index=True)
            df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
            df_all = (
                df_all.dropna(subset=["date"])
                .drop_duplicates(subset=["date"], keep="last")
                .sort_values("date")
                .reset_index(drop=True)
            )
            df_all = df_all[
                (df_all["date"] >= pd.Timestamp(start_dt))
                & (df_all["date"] < pd.Timestamp(end_dt))
            ]

            return self._merge_and_save(df_all, save_path).to_dict()

        except Exception as e:
            logger.exception("下载 metrics 失败")
            return ExtraDownloadResult(success=False, error=str(e)).to_dict()

    def _parse_metrics_day(self, blob: bytes, _meta: Dict) -> Optional[pd.DataFrame]:
        """解析一天的 metrics ZIP 为统一 schema。"""
        df = self._read_csv_from_zip(blob, column_names=self._METRICS_COLS_ORDERED)
        if df is None or df.empty:
            return None

        if "create_time" in df.columns:
            df["date"] = pd.to_datetime(df["create_time"], errors="coerce", utc=False)
        elif "timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"], errors="coerce", unit="ms")
        else:
            return None

        rename_map = {src: dst for src, dst in self._METRICS_RENAME.items() if src in df.columns}
        df = df.rename(columns=rename_map)

        for col in self._METRICS_RENAME.values():
            if col not in df.columns:
                df[col] = None
            df[col] = pd.to_numeric(df[col], errors="coerce")
        keep_cols = ["date"] + list(self._METRICS_RENAME.values())
        return df[keep_cols]

    def _fetch_recent_metrics_via_fapi(
        self,
        fapi_symbol: str,
        start_dt: datetime,
        end_dt: datetime,
        progress_callback=None,
    ) -> pd.DataFrame:
        """用 fapi/data 端点补齐归档未覆盖的最近一小段（固定 5m 粒度）。"""
        if start_dt >= end_dt:
            return pd.DataFrame()
        exchange = self._get_futures_exchange()
        if exchange is None:
            logger.warning("无法初始化 ccxt，跳过 metrics 实时增量")
            return pd.DataFrame()

        specs = [
            ("fapiDataGetOpenInterestHist",            {"sumOpenInterest": "open_interest"},           "OI"),
            ("fapiDataGetGlobalLongShortAccountRatio", {"longShortRatio":  "lsr_global_account"},      "LSR-global"),
            ("fapiDataGetTopLongShortAccountRatio",    {"longShortRatio":  "lsr_top_account"},         "LSR-top-acct"),
            ("fapiDataGetTopLongShortPositionRatio",   {"longShortRatio":  "lsr_top_position"},        "LSR-top-pos"),
            ("fapiDataGetTakerlongshortRatio",         {"buySellRatio":    "taker_buy_ratio_metrics"}, "Taker-ratio"),
        ]
        dfs = []
        for endpoint, value_keys, tag in specs:
            dfs.append(self._fetch_fapi_data_series(
                exchange=exchange,
                endpoint=endpoint,
                symbol=fapi_symbol,
                period="5m",
                start_dt=start_dt,
                end_dt=end_dt,
                time_key="timestamp",
                value_keys=value_keys,
                progress_callback=None,
                tag=tag,
            ))
        merged = self._merge_multi_on_date(dfs)
        if progress_callback:
            progress_callback(99, f"[metrics-recent] 实时补齐 {len(merged)} 条 5m 样本")
        return merged

    # ------------------------------------------------------------------
    # 2. Funding Rate
    # ------------------------------------------------------------------
    def download_funding_rate(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        progress_callback=None,
    ) -> Dict:
        """
        下载资金费率历史（事件级 8h / 部分币种 4h）。

        输出列：`date`、`funding_rate`。
        实盘 `predicted_funding_rate` 通过 `/fapi/v1/premiumIndex` 实时取，不入历史表。
        """
        try:
            self._ensure_dirs()
            fapi_symbol = self._to_fapi_symbol(symbol)
            save_path = _FUNDING_DIR / f"{self._safe_symbol(symbol)}-funding.feather"

            end_dt = (
                datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.utcnow()
            )
            start_dt = (
                datetime.strptime(start_date, "%Y-%m-%d")
                if start_date
                else end_dt - timedelta(days=365)
            )
            if start_dt < _VISION_MIN_DATE:
                start_dt = _VISION_MIN_DATE

            # 归档是按 monthly 发布，保守认为：当前月 + 上一月可能尚未归档，走 ccxt 补齐。
            # 注意"归档切线"要基于 now，而不是 end_dt；对历史查询不应把最后一个月错误地排除。
            now = datetime.utcnow()
            now_month_start = datetime(now.year, now.month, 1)
            archive_safe_month = (now_month_start - timedelta(days=1)).replace(day=1)
            # 枚举月份到：min(end_dt 所在月 - 1d, archive_safe_month - 1d)
            end_month_cap = min(end_dt - timedelta(days=1), archive_safe_month - timedelta(days=1))
            months = list(self._monthrange(start_dt, end_month_cap)) if end_month_cap >= start_dt else []
            # recent 起点：归档切线与用户 start_dt 的较大者
            recent_start = max(start_dt, archive_safe_month)

            url_items = [
                {
                    "url": (
                        f"{_VISION_BASE}/monthly/fundingRate/{fapi_symbol}/"
                        f"{fapi_symbol}-fundingRate-{m:%Y-%m}.zip"
                    ),
                    "meta": {"month": m},
                }
                for m in months
            ]
            frames: List[pd.DataFrame] = []
            if url_items:
                frames = self._fetch_archive_parallel(
                    url_items,
                    reader=self._parse_funding_month,
                    progress_callback=progress_callback,
                    tag="funding-archive",
                )
            df_archive = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

            # recent_start 已在前面按 archive_safe_month 计算
            df_recent = self._fetch_recent_funding_via_ccxt(
                symbol=symbol,
                start_dt=recent_start,
                end_dt=end_dt,
                progress_callback=progress_callback,
            )

            parts = [d for d in (df_archive, df_recent) if d is not None and not d.empty]
            if not parts:
                return ExtraDownloadResult(
                    success=False,
                    error="未获取到任何 funding_rate 数据（归档 + ccxt 均为空）",
                ).to_dict()
            df_all = pd.concat(parts, ignore_index=True)
            df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
            df_all = (
                df_all.dropna(subset=["date", "funding_rate"])
                .drop_duplicates(subset=["date"], keep="last")
                .sort_values("date")
                .reset_index(drop=True)
            )
            df_all = df_all[
                (df_all["date"] >= pd.Timestamp(start_dt))
                & (df_all["date"] < pd.Timestamp(end_dt))
            ]

            return self._merge_and_save(df_all, save_path).to_dict()

        except Exception as e:
            logger.exception("下载 funding_rate 失败")
            return ExtraDownloadResult(success=False, error=str(e)).to_dict()

    @staticmethod
    def _parse_funding_month(blob: bytes, _meta: Dict) -> Optional[pd.DataFrame]:
        """
        解析一月 funding ZIP。归档常见列：
            calc_time, funding_interval_hours, last_funding_rate
        早期版本列名可能略有不同（funding_rate / fundingRate），做宽容映射。
        """
        df = ExtraDataDownloader._read_csv_from_zip(
            blob, column_names=["calc_time", "funding_interval_hours", "last_funding_rate"]
        )
        if df is None or df.empty:
            return None

        # 时间列
        t_col = None
        for c in ("calc_time", "fundingTime", "calcTime", "timestamp"):
            if c in df.columns:
                t_col = c
                break
        if t_col is None:
            return None

        # 费率列
        r_col = None
        for c in ("last_funding_rate", "fundingRate", "funding_rate", "lastFundingRate"):
            if c in df.columns:
                r_col = c
                break
        if r_col is None:
            return None

        out = pd.DataFrame()
        # calc_time 可能是毫秒 int，也可能是 YYYY-MM-DD HH:MM:SS 字符串
        t_series = df[t_col]
        if pd.api.types.is_numeric_dtype(t_series):
            out["date"] = pd.to_datetime(t_series, unit="ms", errors="coerce")
        else:
            out["date"] = pd.to_datetime(t_series, errors="coerce")
        out["funding_rate"] = pd.to_numeric(df[r_col], errors="coerce")
        return out.dropna(subset=["date", "funding_rate"])

    def _fetch_recent_funding_via_ccxt(
        self,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime,
        progress_callback=None,
    ) -> pd.DataFrame:
        """用 ccxt.fetch_funding_rate_history 补齐归档未覆盖的最近区段。"""
        if start_dt >= end_dt:
            return pd.DataFrame()
        exchange = self._get_futures_exchange()
        if exchange is None:
            logger.warning("无法初始化 ccxt，跳过 funding 实时增量")
            return pd.DataFrame()

        ccxt_symbol = self._to_ccxt_unified_symbol(symbol)
        all_rows: List[Dict] = []
        since = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        while since < end_ms:
            try:
                rows = exchange.fetch_funding_rate_history(
                    ccxt_symbol, since=since, limit=1000
                )
            except Exception as e:
                logger.warning(f"拉取 funding_rate_history 失败: {e}")
                break
            if not rows:
                break
            for r in rows:
                ts = r.get("timestamp")
                fr = r.get("fundingRate")
                if ts is None or fr is None or ts > end_ms:
                    continue
                all_rows.append({
                    "date": pd.to_datetime(ts, unit="ms"),
                    "funding_rate": float(fr),
                })
            last_ts = rows[-1].get("timestamp")
            if last_ts is None or last_ts <= since:
                break
            since = last_ts + 1
            time.sleep(getattr(exchange, "rateLimit", 200) / 1000)

        if progress_callback:
            progress_callback(99, f"[funding-recent] ccxt 补齐 {len(all_rows)} 条")
        return pd.DataFrame(all_rows)

    # ------------------------------------------------------------------
    # 3. Mark / Index K 线（→ basis）
    # ------------------------------------------------------------------
    def download_mark_index_klines(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        kind: str = "mark",
        progress_callback=None,
    ) -> Dict:
        """
        下载 Mark Price 或 Index Price K 线。

        Args:
            kind: 'mark' 或 'index'
            timeframe: 与主 OHLCV 同周期即可，后续 `basis = (close - index_close) / index_close`
        输出列：`date, {kind}_open, {kind}_high, {kind}_low, {kind}_close`
        """
        try:
            self._ensure_dirs()
            if kind not in ("mark", "index"):
                return ExtraDownloadResult(success=False, error="kind 必须是 mark 或 index").to_dict()

            fapi_symbol = self._to_fapi_symbol(symbol)
            out_dir = _MARK_DIR if kind == "mark" else _INDEX_DIR
            save_path = out_dir / f"{self._safe_symbol(symbol)}-{timeframe}-{kind}.feather"
            archive_subdir = "markPriceKlines" if kind == "mark" else "indexPriceKlines"

            end_dt = (
                datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.utcnow()
            )
            start_dt = (
                datetime.strptime(start_date, "%Y-%m-%d")
                if start_date
                else end_dt - timedelta(days=180)
            )
            if start_dt < _VISION_MIN_DATE:
                start_dt = _VISION_MIN_DATE

            archive_cutoff = datetime.utcnow() - timedelta(days=_ARCHIVE_LAG_DAYS)
            archive_end = min(end_dt, archive_cutoff)
            recent_start = max(start_dt, archive_cutoff)
            # 注意：mark/indexPriceKlines 归档的文件名是 `{SYM}-{tf}-{DATE}.zip`，
            # 并**不**包含 `markPriceKlines`/`indexPriceKlines` 前缀（与普通 klines 一致）。
            url_items = [
                {
                    "url": (
                        f"{_VISION_BASE}/daily/{archive_subdir}/{fapi_symbol}/{timeframe}/"
                        f"{fapi_symbol}-{timeframe}-{d:%Y-%m-%d}.zip"
                    ),
                    "meta": {"date": d},
                }
                for d in self._daterange(start_dt, archive_end)
            ]
            frames: List[pd.DataFrame] = []
            if url_items:
                frames = self._fetch_archive_parallel(
                    url_items,
                    reader=lambda blob, _m, _k=kind: self._parse_kline_day(blob, _m, _k),
                    progress_callback=progress_callback,
                    tag=f"{kind}-archive",
                )
            df_archive = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

            df_recent = self._fetch_recent_mark_index_via_ccxt(
                symbol=symbol,
                timeframe=timeframe,
                kind=kind,
                start_dt=recent_start,
                end_dt=end_dt,
                progress_callback=progress_callback,
            )

            parts = [d for d in (df_archive, df_recent) if d is not None and not d.empty]
            if not parts:
                return ExtraDownloadResult(
                    success=False,
                    error=f"未获取到任何 {kind}PriceKlines 数据",
                ).to_dict()
            df_all = pd.concat(parts, ignore_index=True)
            df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
            df_all = (
                df_all.dropna(subset=["date"])
                .drop_duplicates(subset=["date"], keep="last")
                .sort_values("date")
                .reset_index(drop=True)
            )
            df_all = df_all[
                (df_all["date"] >= pd.Timestamp(start_dt))
                & (df_all["date"] < pd.Timestamp(end_dt))
            ]
            return self._merge_and_save(df_all, save_path).to_dict()

        except Exception as e:
            logger.exception(f"下载 {kind}PriceKlines 失败")
            return ExtraDownloadResult(success=False, error=str(e)).to_dict()

    def _parse_kline_day(self, blob: bytes, _meta: Dict, kind: str) -> Optional[pd.DataFrame]:
        """
        解析一天的 markPrice/indexPrice K 线 ZIP。目标列：
            date, {kind}_open, {kind}_high, {kind}_low, {kind}_close
        """
        df = self._read_csv_from_zip(blob, column_names=self._KLINE_COLS_ORDERED)
        if df is None or df.empty:
            return None

        if "open_time" in df.columns:
            t_series = df["open_time"]
        elif "timestamp" in df.columns:
            t_series = df["timestamp"]
        else:
            return None

        if pd.api.types.is_numeric_dtype(t_series):
            # 早期（秒）极少见；微秒也有过；用数量级判断
            sample = float(t_series.dropna().iloc[0]) if not t_series.dropna().empty else 0
            if sample > 1e15:
                date_col = pd.to_datetime(t_series, unit="us", errors="coerce")
            elif sample > 1e12:
                date_col = pd.to_datetime(t_series, unit="ms", errors="coerce")
            else:
                date_col = pd.to_datetime(t_series, unit="s", errors="coerce")
        else:
            date_col = pd.to_datetime(t_series, errors="coerce")

        out = pd.DataFrame({"date": date_col})
        for src in ("open", "high", "low", "close"):
            if src in df.columns:
                out[f"{kind}_{src}"] = pd.to_numeric(df[src], errors="coerce")
            else:
                out[f"{kind}_{src}"] = None
        return out.dropna(subset=["date"])

    def _fetch_recent_mark_index_via_ccxt(
        self,
        symbol: str,
        timeframe: str,
        kind: str,
        start_dt: datetime,
        end_dt: datetime,
        progress_callback=None,
    ) -> pd.DataFrame:
        """用 ccxt 补齐归档未覆盖的最近一小段 {kind}PriceKlines。"""
        if start_dt >= end_dt:
            return pd.DataFrame()
        exchange = self._get_futures_exchange()
        if exchange is None:
            logger.warning(f"无法初始化 ccxt，跳过 {kind} 实时增量")
            return pd.DataFrame()

        ccxt_symbol = self._to_ccxt_unified_symbol(symbol)
        timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
        all_rows: List[List] = []
        since = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        while since < end_ms:
            try:
                rows = exchange.fetch_ohlcv(
                    ccxt_symbol, timeframe, since=since, limit=1000,
                    params={"price": kind},
                )
            except Exception as e:
                logger.warning(f"拉取 {kind}PriceKlines(ccxt) 失败: {e}")
                break
            if not rows:
                break
            all_rows.extend(rows)
            last_ts = rows[-1][0]
            if last_ts <= since:
                since += timeframe_ms
            else:
                since = last_ts + timeframe_ms
            time.sleep(getattr(exchange, "rateLimit", 200) / 1000)

        if not all_rows:
            return pd.DataFrame()
        df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "_v"])
        df = df.drop(columns=["_v"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        out = pd.DataFrame({"date": df["date"]})
        for src in ("open", "high", "low", "close"):
            out[f"{kind}_{src}"] = pd.to_numeric(df[src], errors="coerce")
        if progress_callback:
            progress_callback(99, f"[{kind}-recent] ccxt 补齐 {len(out)} 根")
        return out

    # ------------------------------------------------------------------
    # 私有辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _to_fapi_symbol(symbol: str) -> str:
        """把 `BTC/USDT:USDT` 转为 `BTCUSDT`，用于 fapi/data 端点。"""
        base = symbol.replace("/", "").split(":")[0]
        return base.upper()

    @staticmethod
    def _to_ccxt_unified_symbol(symbol: str) -> str:
        """保持 ccxt 统一符号；兼容传入 `BTCUSDT`（补成 `BTC/USDT:USDT`）。"""
        s = symbol.strip().upper()
        if "/" in s:
            return s
        if s.endswith("USDT"):
            base = s[:-4]
            return f"{base}/USDT:USDT"
        return s

    def _parse_pandas_freq(self, timeframe: str) -> str:
        tf = (timeframe or "1h").lower()
        if tf.endswith("m"):
            return f"{int(tf[:-1])}min"
        if tf.endswith("h"):
            return f"{int(tf[:-1])}h"
        if tf.endswith("d"):
            return f"{int(tf[:-1])}d"
        return "1h"

    # ------------------------------------------------------------------
    # 4. Liquidations（大额清算）
    # ------------------------------------------------------------------
    def download_liquidations(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        progress_callback=None,
    ) -> Dict:
        """下载/聚合清算事件（公开接口不可得时降级为清算压力代理指标）。"""
        try:
            self._ensure_dirs()
            fapi_symbol = self._to_fapi_symbol(symbol)
            save_path = _LIQUIDATIONS_DIR / f"{self._safe_symbol(symbol)}-{timeframe}-liquidations.feather"

            end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.utcnow()
            start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else end_dt - timedelta(days=7)
            if start_dt >= end_dt:
                return ExtraDownloadResult(success=False, error="时间范围无效").to_dict()

            exchange = self._get_futures_exchange()
            if exchange is None:
                return ExtraDownloadResult(success=False, error="无法初始化币安期货接口").to_dict()

            # Binance 历史清算明细属于签名接口，公网上通常不可直接拉全量历史。
            # 这里采用公开 OHLCV 的“清算压力代理”作为兜底：
            # liquidation_proxy = abs(ret) * volume（极端波动+大成交量近似强平压力）
            ccxt_symbol = self._to_ccxt_unified_symbol(symbol)
            since = int(start_dt.timestamp() * 1000)
            end_ms = int(end_dt.timestamp() * 1000)
            tf_minutes = self._TIMEFRAME_TO_MINUTES.get((timeframe or "1h").lower(), 60)
            tf_ms = tf_minutes * 60 * 1000
            rows = []
            cursor = since
            loops = 0
            total = max(1, math.ceil((end_ms - since) / (tf_ms * 1000)))
            while cursor < end_ms:
                try:
                    batch = exchange.fetch_ohlcv(ccxt_symbol, timeframe, since=cursor, limit=1000)
                except Exception as e:
                    logger.warning(f"拉取 OHLCV 失败（liquidation proxy）: {e}")
                    batch = []
                if not batch:
                    break
                rows.extend(batch)
                last_ts = int(batch[-1][0])
                if last_ts <= cursor:
                    break
                cursor = last_ts + tf_ms
                loops += 1
                if progress_callback:
                    progress_callback(min(99, int(loops * 100 / total)), f"[liquidations] 代理序列分片 {loops}/{total}")
                time.sleep(getattr(exchange, "rateLimit", 200) / 1000)

            if not rows:
                return ExtraDownloadResult(success=False, error="未获取到可用于构造清算代理的数据").to_dict()

            o = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
            o["date"] = pd.to_datetime(o["ts"], unit="ms")
            o["close"] = pd.to_numeric(o["close"], errors="coerce")
            o["volume"] = pd.to_numeric(o["volume"], errors="coerce")
            o["ret"] = o["close"].pct_change().fillna(0.0)
            o["liq_proxy"] = (o["ret"].abs() * o["volume"]).fillna(0.0)
            o["long_liquidation_value"] = o["liq_proxy"].where(o["ret"] < 0, 0.0)
            o["short_liquidation_value"] = o["liq_proxy"].where(o["ret"] > 0, 0.0)
            o["liquidation_count"] = (o["liq_proxy"] > 0).astype(int)
            df = o[["date", "long_liquidation_value", "short_liquidation_value", "liquidation_count"]].copy()
            freq = self._parse_pandas_freq(timeframe)
            agg = (
                df.set_index("date")
                .resample(freq)
                .agg({
                    "long_liquidation_value": "sum",
                    "short_liquidation_value": "sum",
                    "liquidation_count": "sum",
                })
                .fillna(0.0)
                .reset_index()
            )
            agg["liquidations"] = agg["long_liquidation_value"] + agg["short_liquidation_value"]
            return self._merge_and_save(agg, save_path).to_dict()
        except Exception as e:
            logger.exception("下载 liquidations 失败")
            return ExtraDownloadResult(success=False, error=str(e)).to_dict()

    # ------------------------------------------------------------------
    # 5. Macro / Sentiment（yfinance + coingecko）
    # ------------------------------------------------------------------
    def _download_yfinance_close(self, ticker_map: Dict[str, str], start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        import yfinance as yf

        tickers = list(ticker_map.values())
        raw = pd.DataFrame()
        max_1h_days = 730
        for interval in ("1h", "1d"):
            _start = start_dt
            if interval == "1h" and (end_dt - start_dt).days > max_1h_days:
                _start = end_dt - timedelta(days=max_1h_days)
            for attempt in range(3):
                try:
                    raw = yf.download(
                        tickers=tickers,
                        start=_start.strftime("%Y-%m-%d"),
                        end=end_dt.strftime("%Y-%m-%d"),
                        interval=interval,
                        auto_adjust=False,
                        progress=False,
                        group_by="ticker",
                        threads=True,
                    )
                except Exception as e:
                    logger.warning(f"yfinance 下载失败 interval={interval} attempt={attempt+1}: {e}")
                    raw = pd.DataFrame()
                if raw is not None and len(raw) > 0:
                    break
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
            if raw is not None and len(raw) > 0:
                break
        if raw is None or len(raw) == 0:
            return pd.DataFrame()

        out = pd.DataFrame(index=raw.index)
        for col_name, ticker in ticker_map.items():
            series = None
            if isinstance(raw.columns, pd.MultiIndex):
                for pattern in [
                    (ticker, "Close"), ("Close", ticker),
                    (ticker, "close"), ("close", ticker),
                ]:
                    if pattern in raw.columns:
                        series = raw[pattern]
                        break
                if series is None:
                    level0_vals = set(raw.columns.get_level_values(0))
                    level1_vals = set(raw.columns.get_level_values(1))
                    if ticker in level0_vals and "Close" in level1_vals:
                        series = raw.xs("Close", level=1, axis=1).get(ticker)
                    elif "Close" in level0_vals and ticker in level1_vals:
                        series = raw.xs(ticker, level=1, axis=1).get("Close")
            else:
                if "Close" in raw.columns and len(tickers) == 1:
                    series = raw["Close"]
                elif ticker in raw.columns:
                    series = raw[ticker]
            if series is not None:
                out[col_name] = pd.to_numeric(series, errors="coerce")
        out = out.reset_index().rename(columns={"Datetime": "date", "Date": "date"})
        if "date" not in out.columns and len(out.columns) > 0:
            out = out.rename(columns={out.columns[0]: "date"})
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        return out.dropna(subset=["date"])

    def _download_stablecoin_supply_ratio(self, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """用 CoinGecko 市值序列构造 stablecoin_supply_ratio = usdt_mcap / btc_mcap。"""
        def _market_caps(coin_id: str):
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            # 天数过大时 hourly 不稳定，先尝试 hourly，失败则降级 daily。
            days = max(1, (end_dt - start_dt).days + 3)
            for interval in ("hourly", "daily"):
                try:
                    params = {"vs_currency": "usd", "days": str(days), "interval": interval}
                    resp = self._http_session.get(url, params=params, timeout=30)
                    resp.raise_for_status()
                    j = resp.json() or {}
                    m = j.get("market_caps") or []
                    df = pd.DataFrame(m, columns=["ts", "mcap"])
                    if not df.empty:
                        return df
                except Exception as e:
                    logger.warning(f"CoinGecko {coin_id} 拉取失败 interval={interval}: {e}")
                    continue
            return pd.DataFrame(columns=["ts", "mcap"])

        usdt = _market_caps("tether")
        btc = _market_caps("bitcoin")
        if usdt.empty or btc.empty:
            return pd.DataFrame()
        usdt["date"] = pd.to_datetime(usdt["ts"], unit="ms")
        btc["date"] = pd.to_datetime(btc["ts"], unit="ms")
        merged = usdt[["date", "mcap"]].merge(
            btc[["date", "mcap"]],
            on="date",
            how="inner",
            suffixes=("_usdt", "_btc"),
        )
        merged["stablecoin_supply_ratio"] = merged["mcap_usdt"] / merged["mcap_btc"].replace(0, pd.NA)
        merged = merged[(merged["date"] >= pd.Timestamp(start_dt)) & (merged["date"] < pd.Timestamp(end_dt))]
        return merged[["date", "stablecoin_supply_ratio"]].dropna()

    def _resample_and_ffill(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        freq = self._parse_pandas_freq(timeframe)
        out = (
            df.set_index("date")
            .sort_index()
            .resample(freq)
            .last()
            .ffill()
            .reset_index()
        )
        return out

    def download_macro_factors(
        self,
        symbol: str = "",
        timeframe: str = "1h",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        progress_callback=None,
    ) -> Dict:
        try:
            self._ensure_dirs()
            save_path = _MACRO_DIR / f"{timeframe}-macro.feather"
            end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.utcnow()
            start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else end_dt - timedelta(days=365)
            ticker_map = {
                "dxy": "DX-Y.NYB",
                "spx": "^GSPC",
                "ixic": "^IXIC",
                "gold": "GC=F",
                "us10y_yield": "^TNX",
            }
            if progress_callback:
                progress_callback(15, "正在下载宏观行情 (yfinance)...")
            df = self._download_yfinance_close(ticker_map, start_dt, end_dt)
            if df.empty:
                return ExtraDownloadResult(success=False, error="宏观数据下载为空").to_dict()
            if "us10y_yield" in df.columns:
                df["us10y_yield"] = pd.to_numeric(df["us10y_yield"], errors="coerce") / 100.0
            if progress_callback:
                progress_callback(70, "正在按时间框架重采样并前向填充...")
            df = self._resample_and_ffill(df, timeframe)
            if progress_callback:
                progress_callback(95, "正在保存宏观因子...")
            return self._merge_and_save(df, save_path).to_dict()
        except Exception as e:
            logger.exception("下载 macro 失败")
            return ExtraDownloadResult(success=False, error=str(e)).to_dict()

    def download_sentiment_factors(
        self,
        symbol: str = "",
        timeframe: str = "1h",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        progress_callback=None,
    ) -> Dict:
        try:
            self._ensure_dirs()
            save_path = _SENTIMENT_DIR / f"{timeframe}-sentiment.feather"
            end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.utcnow()
            start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else end_dt - timedelta(days=365)

            if progress_callback:
                progress_callback(10, "正在下载 VIX...")
            vix_df = self._download_yfinance_close({"vix": "^VIX"}, start_dt, end_dt)
            if progress_callback:
                progress_callback(45, "正在下载稳定币购买力代理（USDT/BTC 市值比）...")
            stable_df = self._download_stablecoin_supply_ratio(start_dt, end_dt)

            parts = []
            if not vix_df.empty:
                parts.append(vix_df[["date", "vix"]])
            if not stable_df.empty:
                parts.append(stable_df[["date", "stablecoin_supply_ratio"]])
            if not parts:
                return ExtraDownloadResult(success=False, error="情绪数据下载为空").to_dict()
            merged = parts[0]
            for p in parts[1:]:
                merged = merged.merge(p, on="date", how="outer")

            if progress_callback:
                progress_callback(80, "正在按时间框架重采样并前向填充...")
            merged = self._resample_and_ffill(merged, timeframe)
            if progress_callback:
                progress_callback(95, "正在保存情绪因子...")
            return self._merge_and_save(merged, save_path).to_dict()
        except Exception as e:
            logger.exception("下载 sentiment 失败")
            return ExtraDownloadResult(success=False, error=str(e)).to_dict()

    @staticmethod
    def _merge_multi_on_date(dfs: List[pd.DataFrame]) -> pd.DataFrame:
        dfs = [df for df in dfs if df is not None and not df.empty]
        if not dfs:
            return pd.DataFrame()
        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.merge(df, on="date", how="outer")
        merged = merged.sort_values("date").reset_index(drop=True)
        return merged

    def _fetch_fapi_data_series(
        self,
        exchange: ccxt.Exchange,
        endpoint: str,
        symbol: str,
        period: str,
        start_dt: datetime,
        end_dt: datetime,
        time_key: str,
        value_keys: Dict[str, str],
        progress_callback=None,
        tag: str = "",
    ) -> pd.DataFrame:
        """
        调用币安 fapi/data 风格端点（分页，每次最多 500 条，每请求覆盖约 ``period * 500``）。

        端点方法名通过 `endpoint` 传入（大小写敏感的 ccxt 反射方法名），
        这样可以同时兼容 `fapiDataGetOpenInterestHist` 与 `fapiData_get_openInterestHist`。
        """
        method = None
        for name in (endpoint, endpoint[0].lower() + endpoint[1:]):
            if hasattr(exchange, name):
                method = getattr(exchange, name)
                break
        if method is None:
            logger.warning(f"[{tag}] ccxt 没有 {endpoint}，跳过")
            return pd.DataFrame(columns=["date", *value_keys.values()])

        period_ms = self._TIMEFRAME_TO_MINUTES[period] * 60 * 1000
        window_ms = 500 * period_ms  # 每次约 500 条
        rows: List[Dict] = []
        cur = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        while cur < end_ms:
            batch_end = min(cur + window_ms, end_ms)
            try:
                res = method(
                    {
                        "symbol": symbol,
                        "period": period,
                        "startTime": cur,
                        "endTime": batch_end,
                        "limit": 500,
                    }
                )
            except Exception as e:
                logger.warning(f"[{tag}] 请求失败: {e}")
                break

            if not res:
                cur = batch_end + 1
                continue

            for item in res:
                ts = item.get(time_key)
                if ts is None:
                    continue
                try:
                    ts_int = int(ts)
                except Exception:
                    continue
                row = {"date": pd.to_datetime(ts_int, unit="ms")}
                for src_k, dst_k in value_keys.items():
                    v = item.get(src_k)
                    try:
                        row[dst_k] = float(v) if v is not None else None
                    except Exception:
                        row[dst_k] = None
                rows.append(row)

            if progress_callback:
                progress_callback(
                    min(100, int((cur - int(start_dt.timestamp() * 1000)) * 100 / max(1, end_ms - int(start_dt.timestamp() * 1000)))),
                    f"[{tag}] 累计 {len(rows)} 条",
                )

            cur = batch_end + 1
            time.sleep(getattr(exchange, "rateLimit", 200) / 1000)

        if not rows:
            return pd.DataFrame(columns=["date", *value_keys.values()])

        df = (
            pd.DataFrame(rows)
            .drop_duplicates(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )
        return df


# 便捷单例
_default_extra_downloader: Optional[ExtraDataDownloader] = None


def get_extra_downloader() -> ExtraDataDownloader:
    global _default_extra_downloader
    if _default_extra_downloader is None:
        _default_extra_downloader = ExtraDataDownloader()
    return _default_extra_downloader
