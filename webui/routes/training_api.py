"""
时序因子训练模型API路由
支持决策树(LightGBM/XGBoost)、逻辑回归等模型
支持TSFM(IBM Granite TTM)时序基础模型微调
支持回归/分类方式预测未来时间步标签
损失函数参考 audit/损失函数说明.md
"""

import sys
import os
import json
import uuid
import threading
import time
import logging
import pickle
import copy
import re
from datetime import datetime
from pathlib import Path
from flask import Blueprint, request, jsonify, current_app

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss
)

try:
    from tsfm_public import TimeSeriesPreprocessor, TinyTimeMixerForPrediction
    TSFM_AVAILABLE = True
except ImportError:
    TSFM_AVAILABLE = False

bp = Blueprint('training_api', __name__, url_prefix='/api/training')

logger = logging.getLogger(__name__)

# 注意：training_sessions 为进程内内存存储，**不兼容多 worker 部署**。
# 若以 gunicorn/uwsgi 多 worker 运行，POST /start 与 GET /status 可能命中
# 不同进程导致轮询 404；需要时可改为 Redis / 共享文件存储。
training_sessions = {}
training_sessions_lock = threading.Lock()

MODELS_DIR = Path(__file__).parent.parent.parent / "factorlib" / "trained_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
VALID_MODEL_ID_PATTERN = re.compile(r'^[A-Za-z0-9_.-]+$')
TRAINING_SESSION_TTL_SECONDS = 30 * 60
TRAINING_MAX_SESSIONS = 50
DEFAULT_MAX_CONCURRENT_TRAININGS = 1
DEFAULT_TRAINING_TIMEOUT_SECONDS = 0


_EXCHANGE_NAMES = frozenset({'binance', 'okx', 'bybit', 'huobi', 'kucoin', 'gate', 'bitget'})
_TRADE_TYPE_NAMES = frozenset({'futures', 'spot', 'swap', 'margin', 'options'})


def _get_allowed_data_root():
    """获取允许的训练数据根目录。

    优先级：
    1. 显式 DATA_ROOT 配置；
    2. DATA_DIR 若其路径 **末两级** 恰好是 `<exchange>/<trade_type>`，回退到上两级；
    3. 否则直接返回 DATA_DIR。
    """
    explicit_root = current_app.config.get('DATA_ROOT')
    if explicit_root:
        return Path(explicit_root).resolve()

    configured_data_dir = Path(current_app.config.get('DATA_DIR', 'data'))
    parts = [p.lower() for p in configured_data_dir.parts[-2:]]
    if len(parts) == 2 and parts[0] in _EXCHANGE_NAMES and parts[1] in _TRADE_TYPE_NAMES:
        return configured_data_dir.parent.parent.resolve()
    return configured_data_dir.resolve()


def _is_path_under(parent_path, child_path):
    try:
        child_path.relative_to(parent_path)
        return True
    except ValueError:
        return False


def _validate_training_data_file_path(file_path):
    if not file_path:
        raise ValueError("缺少文件路径")
    requested_path = Path(file_path).resolve()
    allowed_root = _get_allowed_data_root()
    if not _is_path_under(allowed_root, requested_path):
        raise ValueError("非法路径：文件不在数据目录内")
    if not requested_path.exists() or not requested_path.is_file():
        raise ValueError("数据文件不存在")
    if requested_path.suffix.lower() != '.feather':
        raise ValueError("非法文件类型，仅支持feather")
    return requested_path


def _resolve_model_dir(model_id):
    if not model_id or not VALID_MODEL_ID_PATTERN.match(model_id):
        raise ValueError("非法模型ID")
    model_dir = (MODELS_DIR / model_id).resolve()
    models_root = MODELS_DIR.resolve()
    if not _is_path_under(models_root, model_dir):
        raise ValueError("非法模型路径")
    return model_dir


def _safe_pickle_load(file_path):
    class _RestrictedUnpickler(pickle.Unpickler):
        allowed_prefixes = (
            'builtins',
            'collections',
            'numpy',
            'scipy',
            'sklearn',
            'pandas',
            'joblib',
            'threadpoolctl',
        )

        def find_class(self, module, name):
            if module in self.allowed_prefixes or module.startswith(self.allowed_prefixes):
                return super().find_class(module, name)
            raise pickle.UnpicklingError(f"不允许反序列化模块: {module}.{name}")

    with open(file_path, 'rb') as f:
        return _RestrictedUnpickler(f).load()


def _cleanup_training_sessions_locked():
    now_ts = time.time()
    expired_keys = []
    for sid, session in training_sessions.items():
        status = session.get('status')
        completed_ts = session.get('completed_ts')
        if status in ('completed', 'failed') and completed_ts and now_ts - completed_ts > TRAINING_SESSION_TTL_SECONDS:
            expired_keys.append(sid)
    for sid in expired_keys:
        del training_sessions[sid]

    if len(training_sessions) <= TRAINING_MAX_SESSIONS:
        return

    removable = []
    for sid, session in training_sessions.items():
        if session.get('status') in ('running', 'pending'):
            continue
        removable.append((session.get('completed_ts') or 0.0, sid))
    removable.sort(key=lambda x: x[0])
    overflow = max(0, len(training_sessions) - TRAINING_MAX_SESSIONS)
    for _, sid in removable[:overflow]:
        if sid in training_sessions:
            del training_sessions[sid]


class TrainingCancelled(Exception):
    """用户主动取消训练时抛出。"""


def _assert_training_alive(session_id):
    """检查训练会话是否超时或被取消，用于迭代回调。"""
    with training_sessions_lock:
        session = training_sessions.get(session_id)
        if not session:
            raise TimeoutError("训练会话不存在")
        if session.get('cancel_requested'):
            raise TrainingCancelled("用户已取消训练")
        start_ts = session.get('start_ts')
        raw_timeout = session.get('timeout_seconds')
        if raw_timeout is None:
            timeout_seconds = float(DEFAULT_TRAINING_TIMEOUT_SECONDS)
        else:
            timeout_seconds = float(raw_timeout)
    if timeout_seconds > 0 and start_ts and (time.time() - start_ts) > timeout_seconds:
        raise TimeoutError(f"训练超时（>{int(timeout_seconds)}秒）")


_assert_training_not_timeout = _assert_training_alive


def _load_feather_data(file_path, start_date=None, end_date=None):
    df = pd.read_feather(file_path)

    time_col = None
    for col in df.columns:
        cl = col.lower()
        if cl in ('open_time', 'opentime', 'time', 'date', 'timestamp', 'datetime', 'close_time', 'closetime'):
            time_col = col
            break
    if time_col is None:
        for col in df.columns:
            if any(k in col.lower() for k in ('time', 'date', 'timestamp', 'datetime')):
                time_col = col
                break

    if time_col:
        dt = pd.to_datetime(df[time_col], errors='coerce')
        if start_date:
            dt_start = pd.to_datetime(start_date)
            df = df[dt >= dt_start]
            dt = dt[dt >= dt_start]
        if end_date:
            dt_end = pd.to_datetime(end_date)
            df = df[dt <= dt_end]
            dt = dt[dt <= dt_end]
        df = df.copy()
        df['__time__'] = dt
        df = df.sort_values('__time__').reset_index(drop=True)
        df.index = df['__time__']
        del df['__time__']
        if df.index.has_duplicates:
            logger.warning("数据时间索引存在重复，自动去重(keep=first)")
            df = df[~df.index.duplicated(keep='first')]
    return df


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


def _infer_symbol_timeframe_from_feather_path(file_path: str):
    """
    约定主 K 线 feather 命名类似:
        BTC_USDT_USDT-1h-futures.feather
    取最后倒数第二段为 timeframe，前面合并为 symbol。
    """
    try:
        stem = Path(file_path).stem
        parts = stem.split('-')
        if len(parts) < 3:
            return None, None
        timeframe = parts[-2]
        symbol_raw = '-'.join(parts[:-2])
        return symbol_raw, timeframe
    except Exception:
        return None, None


def _infer_exchange_trade_from_feather_path(file_path: str):
    """从文件路径中推断 exchange / trade_type。"""
    try:
        p = Path(file_path).resolve()
        parts = [x.lower() for x in p.parts]
        exchange = None
        trade_type = None
        for i, seg in enumerate(parts):
            if seg in _EXCHANGE_NAMES:
                exchange = seg
                if i + 1 < len(parts) and parts[i + 1] in _TRADE_TYPE_NAMES:
                    trade_type = parts[i + 1]
                break
        return exchange, trade_type
    except Exception:
        return None, None


def _extra_symbol_candidates(symbol_raw: str):
    """从主数据文件符号推测额外数据文件的 safe_symbol。"""
    if not symbol_raw:
        return []
    cands = [symbol_raw]
    # 常见主文件为 BTC_USDT_USDT，额外数据为 BTC_USDT
    if symbol_raw.endswith('_USDT_USDT'):
        cands.append(symbol_raw[:-len('_USDT')])
    # 也兼容其它写法：如果包含多余的 _USDT，可尝试只保留一个 _USDT
    if cands[-1] != symbol_raw and '_USDT_USDT' in symbol_raw:
        cands.append(symbol_raw.replace('_USDT_USDT', '_USDT', 1))
    # 去重保持顺序
    out = []
    seen = set()
    for s in cands:
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _left_join_extra_numeric_columns(main_df: pd.DataFrame, extra_df: pd.DataFrame, source_prefix: str = "extra"):
    """
    按主表 DatetimeIndex 左连接额外数据的数值列，并用 forward fill 对齐。
    返回 (main_df, added_columns)。
    """
    if main_df is None or main_df.empty:
        return main_df, []
    if extra_df is None or extra_df.empty:
        return main_df, []

    main = main_df
    main_index = main.index
    main_dt_index = _normalize_dt_index(main_index)

    extra = extra_df.copy()
    if 'date' in extra.columns:
        extra['date'] = pd.to_datetime(extra['date'], errors='coerce')
        extra = extra.set_index('date')
    elif isinstance(extra.index, pd.DatetimeIndex):
        extra.index = _normalize_dt_index(extra.index)
    else:
        # 无法定位时间列，跳过
        return main_df, []

    extra.index = _normalize_dt_index(extra.index)
    extra = extra[~extra.index.duplicated(keep='last')].sort_index()
    extra_numeric = extra.select_dtypes(include=[np.number])
    if extra_numeric.empty:
        return main_df, []

    aligned = extra_numeric.reindex(main_dt_index, method='ffill')
    added = []
    renamed = {}
    for col in aligned.columns:
        target_col = col
        if target_col in main.columns:
            base = f"extra_{source_prefix}_{col}"
            target_col = base
            idx = 1
            while target_col in main.columns:
                target_col = f"{base}_{idx}"
                idx += 1
            renamed[col] = target_col
        if target_col not in main.columns:
            added.append(target_col)
        main[target_col] = aligned[col].values
    return main, added, renamed


def _try_join_extra_files_for_tsfm(df: pd.DataFrame, file_path: str):
    """
    TSFM训练前自动把额外数据(feather)并入主数据df，确保控制变量能被TSFM看到。
    只在能推断出 symbol/timeframe 且对应文件存在时才会 join。
    """
    symbol_raw, timeframe = _infer_symbol_timeframe_from_feather_path(file_path)
    if not symbol_raw or not timeframe:
        return df, {'extra_files': [], 'extra_columns': []}

    exchange, trade_type = _infer_exchange_trade_from_feather_path(file_path)
    trade_prefix = 'futures' if trade_type in (None, '', 'futures', 'swap') else trade_type
    data_root = _get_allowed_data_root()
    root = data_root / (exchange or 'binance')
    safe_symbols = _extra_symbol_candidates(symbol_raw)
    if not safe_symbols:
        return df, {'extra_files': [], 'extra_columns': [], 'renamed_columns': {}}

    candidates = [
        (
            'metrics',
            lambda ss: [
                root / f'{trade_prefix}_metrics' / f'{ss}-{timeframe}-metrics.feather',
                root / f'{trade_prefix}_metrics' / f'{ss}-5m-metrics.feather',
            ],
        ),
        ('funding', lambda ss: [root / f'{trade_prefix}_funding' / f'{ss}-funding.feather']),
        ('mark', lambda ss: [root / f'{trade_prefix}_markprice' / f'{ss}-{timeframe}-mark.feather']),
        ('index', lambda ss: [root / f'{trade_prefix}_indexprice' / f'{ss}-{timeframe}-index.feather']),
        ('liquidations', lambda ss: [root / f'{trade_prefix}_liquidations' / f'{ss}-{timeframe}-liquidations.feather']),
        ('macro', lambda ss: [root / f'{trade_prefix}_macro' / f'{ss}-{timeframe}-macro.feather']),
        ('sentiment', lambda ss: [root / f'{trade_prefix}_sentiment' / f'{ss}-{timeframe}-sentiment.feather']),
    ]

    extra_files = []
    extra_columns = []
    renamed_columns = {}
    out = df
    for ss in safe_symbols:
        for dt_name, p_fn in candidates:
            p_list = p_fn(ss)
            picked = None
            for p in p_list:
                if p.exists():
                    picked = p
                    break
            if picked is None:
                continue
            try:
                extra_df = pd.read_feather(picked)
                out, added_cols, renamed_map = _left_join_extra_numeric_columns(out, extra_df, source_prefix=dt_name)
                extra_files.append(str(picked))
                extra_columns.extend(added_cols)
                if renamed_map:
                    renamed_columns[str(picked)] = renamed_map
            except Exception as e:
                logger.warning(f"TSFM并入额外数据失败 {picked}: {e}")

    if 'mark_close' in out.columns and 'index_close' in out.columns and 'basis' not in out.columns:
        out['basis'] = (out['mark_close'] - out['index_close']) / out['index_close']
        extra_columns.append('basis')
    elif 'index_close' in out.columns and 'close' in out.columns and 'basis' not in out.columns:
        out['basis'] = (out['close'] - out['index_close']) / out['index_close']
        extra_columns.append('basis')

    # 去重保持顺序
    uniq_cols = []
    seen = set()
    for c in extra_columns:
        if c in seen:
            continue
        seen.add(c)
        uniq_cols.append(c)
    return out, {'extra_files': extra_files, 'extra_columns': uniq_cols, 'renamed_columns': renamed_columns}


def _sanitize_feature_names(columns):
    """规范化列名，避免 LightGBM 对特殊字符报错，同时保证唯一性。"""
    seen = {}
    mapping = {}
    sanitized = []
    for col in columns:
        safe = re.sub(r'[^A-Za-z0-9_]+', '_', str(col)).strip('_')
        if not safe:
            safe = 'feat'
        idx = seen.get(safe, 0)
        if idx > 0:
            unique = f"{safe}_{idx}"
        else:
            unique = safe
        seen[safe] = idx + 1
        mapping[col] = unique
        sanitized.append(unique)
    return sanitized, mapping


def _build_features(df, factor_ids, engine):
    series_list = []
    failed_factors = []
    for fid in factor_ids:
        try:
            series = engine.compute_single_factor(fid, df)
            if series is None:
                failed_factors.append({'factor_id': fid, 'reason': '返回空'})
                continue
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            series = series.copy()
            series.name = fid
            series_list.append(series)
        except Exception as e:
            failed_factors.append({'factor_id': fid, 'reason': str(e)[:200]})
            logger.warning(f"计算因子 {fid} 失败: {e}")
    if not series_list:
        return pd.DataFrame(index=df.index), failed_factors
    feature_df = pd.concat(series_list, axis=1).reindex(df.index)
    feature_df = feature_df.dropna(how='all')
    return feature_df, failed_factors


def _build_label(df, label_type, predict_step=1):
    close = None
    for col in df.columns:
        if col.lower() == 'close':
            close = df[col]
            break
    if close is None:
        raise ValueError("数据中找不到 close 列")

    if label_type == 'log_return':
        future_close = close.shift(-predict_step)
        label = np.log(future_close / close)
        label.name = f'log_return_{predict_step}'
    elif label_type == 'direction':
        future_close = close.shift(-predict_step)
        ret = future_close / close - 1
        label = pd.Series(np.nan, index=ret.index, dtype=float)
        label[ret >= 0] = 1.0
        label[ret < 0] = 0.0
        label.name = f'direction_{predict_step}'
    elif label_type == 'composite':
        future_close = close.shift(-predict_step)
        log_ret = np.log(future_close / close)
        label = np.sign(log_ret) * np.abs(log_ret) ** 0.5
        label.name = f'composite_{predict_step}'
    else:
        raise ValueError(f"不支持的标签类型: {label_type}")
    return label


def _split_data_chronological(df, train_ratio=0.8, val_ratio=0.1):
    """按时间顺序 train → val → test 三段式切分。

    val 段位于中间，供 early stopping 使用；test 段位于最末，供最终评估。
    返回顺序与时间顺序一致：(train_df, val_df, test_df)。
    test 集大小 = 1 - train_ratio - val_ratio。
    """
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(f"train_ratio({train_ratio}) + val_ratio({val_ratio}) 必须 < 1.0")
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df


class DirectionAwareMSELoss:
    @staticmethod
    def compute(y_true, y_pred, lambda_=2.0):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if len(y_true) == 0:
            return 0.0
        mse = np.mean((y_true - y_pred) ** 2)
        direction_wrong = (np.sign(y_true) != np.sign(y_pred)).astype(float)
        direction_penalty = np.mean(direction_wrong * (y_true - y_pred) ** 2 * lambda_)
        return mse + direction_penalty


class CompositeLoss:
    @staticmethod
    def compute(y_true, y_pred, alpha=1.0, beta=1.0, k=5.0):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if len(y_true) == 0:
            return 0.0
        mse = np.mean((y_true - y_pred) ** 2)
        direction_loss = np.mean(1 - np.tanh(k * y_true * y_pred))
        return alpha * mse + beta * direction_loss


class MagnitudeWeightedDirectionLoss:
    @staticmethod
    def compute(y_true, y_pred, lambda_=2.0):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if len(y_true) == 0:
            return 0.0
        mse = np.mean((y_true - y_pred) ** 2)
        direction_wrong = (np.sign(y_true) != np.sign(y_pred)).astype(float)
        mag_penalty = np.mean(np.abs(y_true) * direction_wrong) * lambda_
        return mse + mag_penalty


class MSEHingeLoss:
    @staticmethod
    def compute(y_true, y_pred, alpha=1.0, beta=1.0):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if len(y_true) == 0:
            return 0.0
        mse = np.mean((y_true - y_pred) ** 2)
        hinge = np.mean(np.maximum(0.0, -y_true * y_pred))
        return alpha * mse + beta * hinge


LOSS_FUNCTIONS = {
    'mse': {'name': 'MSE', 'description': '均方误差', 'type': 'regression'},
    'mae': {'name': 'MAE', 'description': '平均绝对误差，对极端值(插针)更鲁棒，推荐TSFM使用', 'type': 'regression'},
    'mse_hinge': {'name': 'MSE+铰链损失', 'description': '评估指标: α·MSE + β·Hinge(方向错误惩罚)', 'type': 'regression'},
    'direction_aware_mse': {'name': '方向感知MSE', 'description': '评估指标: 方向错误时MSE放大(1+λ)倍', 'type': 'regression'},
    'composite': {'name': '复合损失(tanh)', 'description': '评估指标: MSE + tanh方向损失', 'type': 'regression'},
    'magnitude_weighted': {'name': '幅度加权方向损失', 'description': '评估指标: 大幅波动方向错误惩罚更重', 'type': 'regression'},
    'log_loss': {'name': '对数损失', 'description': '分类对数损失(交叉熵)', 'type': 'classification'},
}


def _normalize_and_validate_split_ratios(train_ratio, test_ratio, val_ratio):
    try:
        train_ratio = float(train_ratio)
        test_ratio = float(test_ratio)
        val_ratio = float(val_ratio)
    except (TypeError, ValueError):
        raise ValueError("数据集划分比例必须是数字")

    if train_ratio <= 0 or test_ratio <= 0 or val_ratio <= 0:
        raise ValueError("训练/测试/验证集比例必须都大于0")

    total = train_ratio + test_ratio + val_ratio
    if total <= 0:
        raise ValueError("数据集划分比例总和必须大于0")
    return train_ratio / total, test_ratio / total, val_ratio / total


def _is_potential_leakage_feature(column_name):
    cl = str(column_name).lower()
    leakage_tokens = (
        "future", "fwd", "forward", "next_",
        "target", "label", "return_t+",
        "pnl", "profit", "sharpe", "signal",
        "predict", "pred_", "forecast",
    )
    if any(token in cl for token in leakage_tokens):
        return True
    if cl == 'y' or cl.startswith('y_'):
        return True
    return False


def _custom_eval_metric_direction_aware(y_pred, dataset, lambda_=2.0):
    y_true = dataset.get_label()
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return 'dir_aware_mse', 0.0, False
    mse = np.mean((y_true - y_pred) ** 2)
    direction_wrong = (np.sign(y_true) != np.sign(y_pred)).astype(float)
    direction_penalty = np.mean(direction_wrong * (y_true - y_pred) ** 2 * lambda_)
    loss = mse + direction_penalty
    return 'dir_aware_mse', loss, False


def _custom_eval_metric_composite(y_pred, dataset, alpha=1.0, beta=1.0, k=5.0):
    y_true = dataset.get_label()
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return 'composite_loss', 0.0, False
    mse = np.mean((y_true - y_pred) ** 2)
    direction_loss = np.mean(1 - np.tanh(k * y_true * y_pred))
    loss = alpha * mse + beta * direction_loss
    return 'composite_loss', loss, False


def _custom_eval_metric_magnitude_weighted(y_pred, dataset, lambda_=2.0):
    y_true = dataset.get_label()
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return 'mag_weighted_loss', 0.0, False
    mse = np.mean((y_true - y_pred) ** 2)
    direction_wrong = (np.sign(y_true) != np.sign(y_pred)).astype(float)
    mag_penalty = np.mean(np.abs(y_true) * direction_wrong) * lambda_
    loss = mse + mag_penalty
    return 'mag_weighted_loss', loss, False


def _custom_eval_metric_mse_hinge(y_pred, dataset, alpha=1.0, beta=1.0):
    y_true = dataset.get_label()
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return 'mse_hinge_loss', 0.0, False
    mse = np.mean((y_true - y_pred) ** 2)
    hinge = np.mean(np.maximum(0.0, -y_true * y_pred))
    loss = alpha * mse + beta * hinge
    return 'mse_hinge_loss', loss, False


def _train_tsfm_model(session_id, config, df, file_path):
    if not TSFM_AVAILABLE:
        raise ImportError(
            "TSFM依赖未安装。请在conda test环境中执行: "
            "pip install tsfm_public transformers torch"
        )

    from transformers import Trainer, TrainingArguments

    label_type = config.get('label_type', 'log_return')
    predict_step = config.get('predict_step', 1)
    loss_function = config.get('loss_function', 'mae')
    train_ratio = config.get('train_ratio', 0.8)
    test_ratio = config.get('test_ratio', 0.1)
    val_ratio = config.get('val_ratio', 0.1)
    model_params = config.get('model_params', {})
    start_date = config.get('start_date')
    end_date = config.get('end_date')
    factor_ids = config.get('factor_ids', [])

    context_length = int(model_params.get('context_length', 512))
    forecast_length = int(model_params.get('forecast_length', 96))
    num_train_epochs = int(model_params.get('num_train_epochs', 10))
    batch_size = int(model_params.get('per_device_train_batch_size', 32))
    tsfm_lr = float(model_params.get('learning_rate', 1e-3))
    tsfm_weight_decay = float(model_params.get('weight_decay', 0.01))
    freeze_backbone = model_params.get('freeze_backbone', True)
    decoder_mode = model_params.get('decoder_mode', 'default')
    pretrained_model_id = model_params.get(
        'pretrained_model_id', 'ibm-granite/granite-timeseries-ttm-r2'
    )

    with training_sessions_lock:
        training_sessions[session_id]['progress'] = 10
        training_sessions[session_id]['current_step'] = 'data_preparation'
        training_sessions[session_id]['message'] = '正在准备TSFM数据...'

    _assert_training_not_timeout(session_id)

    merged_extra_meta = {'extra_files': [], 'extra_columns': []}
    try:
        # TSFM 训练前：自动并入额外数据文件（metrics/funding/mark/index/liquidations/macro/sentiment）
        df, merged_extra_meta = _try_join_extra_files_for_tsfm(df, file_path)
    except Exception as e:
        logger.warning(f"TSFM并入额外数据失败：{e}")

    # TSFM 训练前：如果用户选择了因子，先计算并作为控制变量输入
    if factor_ids:
        try:
            from factor_miner.core.factor_engine import get_global_engine
            engine = get_global_engine()
            feature_df, failed_factors = _build_features(df, factor_ids, engine)
            if feature_df is None or feature_df.empty:
                reasons = "; ".join(
                    f"{item['factor_id']}: {item['reason']}" for item in (failed_factors or [])[:5]
                )
                raise ValueError(
                    "选中的因子未生成有效特征（TSFM）。"
                    + (f"前5个失败原因: {reasons}" if reasons else "")
                )
            # feature_df index 对齐 df.index
            for c in feature_df.columns:
                if c not in df.columns:
                    df[c] = feature_df[c]
            if failed_factors:
                with training_sessions_lock:
                    training_sessions[session_id]['failed_factors'] = failed_factors
        except Exception as e:
            logger.warning(f"TSFM计算因子特征失败：{e}")

    timestamp_col = None
    for col in df.columns:
        cl = col.lower()
        if cl in ('open_time', 'opentime', 'time', 'date',
                   'timestamp', 'datetime', 'close_time', 'closetime'):
            timestamp_col = col
            break

    if timestamp_col is None:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            timestamp_col = df.columns[0]
        else:
            raise ValueError("数据中找不到时间戳列，TSFM模型需要时间戳列")

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col).reset_index(drop=True)

    col_mapping = {}
    for col in df.columns:
        cl = col.lower()
        if cl == 'open':
            col_mapping[col] = 'Open'
        elif cl == 'high':
            col_mapping[col] = 'High'
        elif cl == 'low':
            col_mapping[col] = 'Low'
        elif cl == 'close':
            col_mapping[col] = 'Close'
        elif cl == 'volume':
            col_mapping[col] = 'Volume'

    tsfm_df = df.rename(columns=col_mapping)
    tsfm_df = tsfm_df.rename(columns={timestamp_col: 'timestamp'})

    if 'Close' not in tsfm_df.columns:
        raise ValueError("数据中找不到Close列，TSFM模型需要Close列作为预测目标")

    min_required = context_length + forecast_length
    if len(tsfm_df) < min_required:
        raise ValueError(
            f"数据量不足: 需要至少{min_required}行(context={context_length}+forecast={forecast_length})，"
            f"当前只有{len(tsfm_df)}行。请减小context_length/forecast_length或扩大数据范围"
        )

    with training_sessions_lock:
        training_sessions[session_id]['progress'] = 20
        training_sessions[session_id]['current_step'] = 'preprocessing'
        training_sessions[session_id]['message'] = '正在初始化TSFM预处理器...'

    _assert_training_not_timeout(session_id)

    train_end = int(len(tsfm_df) * train_ratio)
    val_end = int(len(tsfm_df) * (train_ratio + val_ratio))

    # 先按训练集可用性筛列，避免tsfm_public在scaler阶段因“整列全缺失”报错
    candidate_targets = ['Close']
    if 'High' in tsfm_df.columns:
        candidate_targets.append('High')
    if 'Low' in tsfm_df.columns:
        candidate_targets.append('Low')

    numeric_cols = list(tsfm_df.select_dtypes(include=[np.number]).columns)
    candidate_controls = []
    for c in numeric_cols:
        if c in candidate_targets:
            continue
        if _is_potential_leakage_feature(c):
            continue
        candidate_controls.append(c)

    tsfm_train_raw = tsfm_df.iloc[:train_end].copy()
    target_columns = []
    for c in candidate_targets:
        if tsfm_train_raw[c].notna().any():
            target_columns.append(c)
    if 'Close' not in target_columns:
        raise ValueError("训练集Close列全为空，无法训练TSFM模型")

    control_columns = []
    dropped_all_missing = []
    for c in candidate_controls:
        if tsfm_train_raw[c].notna().any():
            control_columns.append(c)
        else:
            dropped_all_missing.append(c)
    if dropped_all_missing:
        logger.warning("TSFM控制变量中训练集全缺失列已移除: %s", dropped_all_missing[:50])

    factor_input_cols = [c for c in control_columns if c in set(factor_ids)]
    extra_joined_cols = merged_extra_meta.get('extra_columns', [])
    extra_input_cols = [c for c in control_columns if c in set(extra_joined_cols)]
    dropped_factor_cols = [c for c in factor_ids if c not in factor_input_cols]
    logger.info(
        "TSFM输入列统计: targets=%s, controls=%s, 因子入模=%s/%s, 额外数据入模=%s/%s",
        len(target_columns), len(control_columns),
        len(factor_input_cols), len(factor_ids),
        len(extra_input_cols), len(extra_joined_cols)
    )
    if dropped_factor_cols:
        logger.warning("TSFM未入模因子列(可能全缺失/非数值/泄漏过滤): %s", dropped_factor_cols[:50])

    used_columns = ['timestamp'] + target_columns + control_columns
    tsfm_df_model = tsfm_df[used_columns].copy()

    inf_cols = []
    for c in target_columns + control_columns:
        if c in tsfm_df_model.columns:
            s = tsfm_df_model[c]
            mask = np.isinf(s.values) if np.issubdtype(s.dtype, np.floating) else None
            if mask is not None and mask.any():
                inf_count = int(mask.sum())
                inf_cols.append(f"{c}({inf_count})")
    if inf_cols:
        logger.warning("TSFM输入列含infinity值，已替换为NaN: %s", inf_cols[:50])
    tsfm_df_model[target_columns + control_columns] = (
        tsfm_df_model[target_columns + control_columns]
        .replace([np.inf, -np.inf], np.nan)
    )

    all_nan_after_clean = []
    for c in list(target_columns):
        if not tsfm_df_model.iloc[:train_end][c].notna().any():
            all_nan_after_clean.append(c)
            target_columns.remove(c)
    for c in list(control_columns):
        if not tsfm_df_model.iloc[:train_end][c].notna().any():
            all_nan_after_clean.append(c)
            control_columns.remove(c)
    if all_nan_after_clean:
        logger.warning("TSFM清洗infinity后训练集全NaN列已移除: %s", all_nan_after_clean[:50])
    if 'Close' not in target_columns:
        raise ValueError("清洗infinity后Close列在训练集全为NaN，无法训练TSFM模型")
    used_columns = ['timestamp'] + target_columns + control_columns
    tsfm_df_model = tsfm_df_model[used_columns].copy()

    tsfm_train = tsfm_df_model.iloc[:train_end].copy()
    tsfm_val = tsfm_df_model.iloc[train_end:val_end].copy()
    tsfm_test = tsfm_df_model.iloc[val_end:].copy()

    if len(tsfm_train) < min_required:
        raise ValueError(
            f"训练集数据量不足: 需要至少{min_required}行，当前只有{len(tsfm_train)}行"
        )

    column_specifiers = {
        'timestamp_column': 'timestamp',
        'id_columns': [],
        'target_columns': target_columns,
        'control_columns': control_columns,
    }

    tsp = TimeSeriesPreprocessor(
        **column_specifiers,
        context_length=context_length,
        prediction_length=forecast_length,
        scaling=True,
        scaler_type='standard',
    )

    from tsfm_public import get_datasets
    import torch

    split_config = {
        "train": [0, train_end],
        "valid": [train_end, val_end],
        "test": [val_end, len(tsfm_df_model)],
    }
    train_dataset, val_dataset, test_dataset = get_datasets(
        ts_preprocessor=tsp,
        dataset=tsfm_df_model,
        split_config=split_config,
        stride=1,
        use_frequency_token=False,
        enable_padding=True,
    )

    class _TSFMDatasetFixMask(torch.utils.data.Dataset):
        def __init__(self, dataset, forecast_len, context_len):
            self.dataset = dataset
            self.forecast_len = forecast_len
            self.context_len = context_len

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            for mask_key, target_len in [
                ('future_observed_mask', self.forecast_len),
                ('past_observed_mask', self.context_len),
            ]:
                if mask_key not in item:
                    continue
                m = item[mask_key]
                if isinstance(m, torch.Tensor) and m.ndim >= 2 and m.shape[0] == 1 and target_len > 1:
                    item[mask_key] = m.expand(target_len, *m.shape[1:]).contiguous()
                elif isinstance(m, np.ndarray) and m.ndim >= 2 and m.shape[0] == 1 and target_len > 1:
                    item[mask_key] = np.tile(m, (target_len, 1) + (1,) * (m.ndim - 2))
            return item

    if len(train_dataset) > 0:
        sample = train_dataset[0]
        sample_keys = list(sample.keys())
        logger.info("TSFM训练样本键: %s", sample_keys)
        for k in ('future_observed_mask', 'past_observed_mask'):
            if k in sample:
                s = sample[k]
                logger.info("TSFM样本[%s] shape=%s dtype=%s", k, list(s.shape) if hasattr(s, 'shape') else type(s), s.dtype if hasattr(s, 'dtype') else '')

    train_dataset = _TSFMDatasetFixMask(train_dataset, forecast_length, context_length)
    val_dataset = _TSFMDatasetFixMask(val_dataset, forecast_length, context_length)
    test_dataset = _TSFMDatasetFixMask(test_dataset, forecast_length, context_length)

    with training_sessions_lock:
        training_sessions[session_id]['progress'] = 35
        training_sessions[session_id]['current_step'] = 'model_loading'
        training_sessions[session_id]['message'] = f'正在加载预训练模型 {pretrained_model_id}...'
        training_sessions[session_id]['data_info'] = {
            'total_samples': len(tsfm_df_model),
            'train_samples': len(tsfm_train),
            'test_samples': len(tsfm_test),
            'val_samples': len(tsfm_val),
            'feature_count': len(target_columns) + len(control_columns),
            'feature_names': target_columns + control_columns,
            'factor_input_columns': factor_input_cols[:200],
            'dropped_factor_columns': dropped_factor_cols[:200],
            'extra_input_columns': extra_input_cols[:200],
            'label_name': 'Close',
            'train_range': (
                f"{tsfm_train['timestamp'].iloc[0]} ~ {tsfm_train['timestamp'].iloc[-1]}"
                if len(tsfm_train) > 0 else ""
            ),
            'test_range': (
                f"{tsfm_test['timestamp'].iloc[0]} ~ {tsfm_test['timestamp'].iloc[-1]}"
                if len(tsfm_test) > 0 else ""
            ),
            'val_range': (
                f"{tsfm_val['timestamp'].iloc[0]} ~ {tsfm_val['timestamp'].iloc[-1]}"
                if len(tsfm_val) > 0 else ""
            ),
            'extra_joined_columns': merged_extra_meta.get('extra_columns', []),
            'extra_renamed_columns': merged_extra_meta.get('renamed_columns', {}),
        }

    _assert_training_not_timeout(session_id)

    model = TinyTimeMixerForPrediction.from_pretrained(pretrained_model_id)

    model_context_length = getattr(model.config, 'context_length', None)
    if model_context_length is not None and context_length != model_context_length:
        raise ValueError(
            f"预训练模型 {pretrained_model_id} 的 context_length 固定为 {model_context_length}，"
            f"不支持修改（修改会导致 patch/位置编码/MLP 权重维度不匹配）。"
            f"请将 context_length 设为 {model_context_length}，或通过调整数据时间范围适配。"
        )

    model_pred_len = getattr(model.config, 'prediction_length', forecast_length)
    if forecast_length != model_pred_len:
        if forecast_length < model_pred_len:
            model.config.prediction_filter_length = forecast_length
            model.prediction_filter_length = forecast_length
            if hasattr(model, 'head') and hasattr(model.head, 'prediction_filter_length'):
                model.head.prediction_filter_length = forecast_length
            logger.info(
                "TSFM: 用户预测长度(%d) < 模型默认(%d)，已设置prediction_filter_length=%d截断输出",
                forecast_length, model_pred_len, forecast_length,
            )
        else:
            logger.warning(
                "TSFM: 用户预测长度(%d) > 模型默认(%d)，模型只能预测%d步，结果可能不准确",
                forecast_length, model_pred_len, model_pred_len,
            )

    def _fix_mask_hook(module, args, kwargs):
        expected_pred_len = getattr(module.config, 'prediction_filter_length', None) or module.config.prediction_length
        for mask_key, expected_len in [
            ('future_observed_mask', expected_pred_len),
            ('past_observed_mask', module.config.context_length),
        ]:
            mask = kwargs.get(mask_key)
            if mask is not None and mask.ndim >= 2 and mask.shape[1] != expected_len:
                if mask.shape[0] == expected_len:
                    pass
                elif mask.shape[1] == 1:
                    kwargs[mask_key] = mask.repeat(1, expected_len, *(1,) * (mask.ndim - 2))
                    logger.warning(
                        "TSFM forward_hook: %s %s -> %s (repeat dim1 to match expected %d)",
                        mask_key, list(mask.shape), list(kwargs[mask_key].shape), expected_len,
                    )
                elif mask.shape[1] > expected_len:
                    kwargs[mask_key] = mask[:, :expected_len, ...]
                    logger.warning(
                        "TSFM forward_hook: %s %s -> %s (truncate dim1 to match expected %d)",
                        mask_key, list(mask.shape), list(kwargs[mask_key].shape), expected_len,
                    )
        return args, kwargs

    model.register_forward_pre_hook(_fix_mask_hook, with_kwargs=True)

    if freeze_backbone and hasattr(model, 'backbone'):
        for param in model.backbone.parameters():
            param.requires_grad = False

    if decoder_mode == 'mix_channel' and hasattr(model, 'config'):
        try:
            model.config.decoder_mode = 'mix_channel'
        except Exception:
            logger.warning("无法设置decoder_mode=mix_channel，将使用默认模式")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"TSFM模型参数: 总计{total_params}, 可训练{trainable_params}")

    with training_sessions_lock:
        training_sessions[session_id]['progress'] = 45
        training_sessions[session_id]['current_step'] = 'model_training'
        training_sessions[session_id]['message'] = '正在微调TSFM模型...'

    _assert_training_not_timeout(session_id)

    temp_output_dir = MODELS_DIR / f"_tsfm_temp_{session_id[:8]}"
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import TrainerCallback

    class _TSFMProgressCallback(TrainerCallback):
        def __init__(self, sid, total_epochs):
            self.sid = sid
            self.total_epochs = total_epochs

        def on_log(self, args, state, control, logs=None, **kwargs):
            _assert_training_not_timeout(self.sid)
            if state.is_world_process_zero:
                epoch = state.epoch or 0
                pct = 45 + int(min(1.0, epoch / max(1, self.total_epochs)) * 35)
                with training_sessions_lock:
                    if self.sid in training_sessions:
                        training_sessions[self.sid]['progress'] = pct
                        training_sessions[self.sid]['message'] = (
                            f'TSFM微调中 (epoch {epoch:.1f}/{self.total_epochs})'
                        )

    def _tsfm_data_collator(features):
        if not features:
            return {}
        allowed_keys = {
            "past_values",
            "future_values",
            "past_observed_mask",
            "future_observed_mask",
            "freq_token",
            "static_categorical_values",
            "label",
            "label_ids",
        }
        batch = {}
        for key in features[0].keys():
            if key not in allowed_keys:
                continue
            values = [f[key] for f in features if key in f]
            if len(values) != len(features):
                continue
            first = values[0]
            if isinstance(first, torch.Tensor):
                try:
                    batch[key] = torch.stack(values)
                except Exception:
                    batch[key] = torch.tensor(values)
                continue
            if isinstance(first, (np.ndarray, list, tuple)):
                arr0 = np.asarray(first)
                if np.issubdtype(arr0.dtype, np.number) or np.issubdtype(arr0.dtype, np.bool_):
                    batch[key] = torch.tensor(np.stack([np.asarray(v) for v in values]))
                continue
            if isinstance(first, (int, float, np.number, bool)):
                batch[key] = torch.tensor(values)

        if 'future_observed_mask' in batch and 'future_values' in batch:
            mask = batch['future_observed_mask']
            fv = batch['future_values']
            if mask.ndim >= 2 and fv.ndim >= 2 and mask.shape[1] != fv.shape[1]:
                if mask.shape[1] == 1:
                    batch['future_observed_mask'] = mask.repeat(1, fv.shape[1], *(1,) * (mask.ndim - 2))
                    logger.info(
                        "TSFM data_collator: future_observed_mask %s -> %s (repeat to match future_values %s)",
                        list(mask.shape), list(batch['future_observed_mask'].shape), list(fv.shape)
                    )
        if 'past_observed_mask' in batch and 'past_values' in batch:
            mask = batch['past_observed_mask']
            pv = batch['past_values']
            if mask.ndim >= 2 and pv.ndim >= 2 and mask.shape[1] != pv.shape[1]:
                if mask.shape[1] == 1:
                    batch['past_observed_mask'] = mask.repeat(1, pv.shape[1], *(1,) * (mask.ndim - 2))
                    logger.info(
                        "TSFM data_collator: past_observed_mask %s -> %s (repeat to match past_values %s)",
                        list(mask.shape), list(batch['past_observed_mask'].shape), list(pv.shape)
                    )

        return batch

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        try:
            device_count = torch.cuda.device_count()
            current_idx = torch.cuda.current_device()
            current_name = torch.cuda.get_device_name(current_idx)
            all_devices = [torch.cuda.get_device_name(i) for i in range(device_count)]
            logger.info(
                "TSFM训练设备: CUDA可用=True, 设备数=%s, 当前设备=%s(%s), 全部设备=%s, torch.cuda=%s",
                device_count, current_idx, current_name, all_devices, torch.version.cuda
            )
        except Exception as e:
            logger.warning("TSFM读取CUDA设备信息失败，将继续训练: %s", e)
    else:
        logger.warning(
            "TSFM训练设备: CUDA可用=False，将使用CPU训练。可通过安装/配置CUDA版本PyTorch启用GPU加速。"
        )
    no_cuda = not cuda_available

    training_args = TrainingArguments(
        output_dir=str(temp_output_dir),
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=tsfm_lr,
        weight_decay=tsfm_weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_steps=10,
        report_to="none",
        disable_tqdm=True,
        no_cuda=no_cuda,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=_tsfm_data_collator,
        callbacks=[_TSFMProgressCallback(session_id, num_train_epochs)],
    )

    trainer.train()

    with training_sessions_lock:
        training_sessions[session_id]['progress'] = 80
        training_sessions[session_id]['current_step'] = 'evaluation'
        training_sessions[session_id]['message'] = '正在评估TSFM模型...'

    _assert_training_not_timeout(session_id)

    metrics = {'test': {}, 'val': {}}
    evals_result = {}
    try:
        test_output = trainer.predict(test_dataset)
        val_output = trainer.predict(val_dataset)

        n_targets = len(target_columns)
        close_idx = target_columns.index('Close') if 'Close' in target_columns else 0

        def _extract_from_dataset(dataset, keys, max_samples=None):
            results = {k: [] for k in keys}
            indices = range(len(dataset)) if max_samples is None else range(min(max_samples, len(dataset)))
            for i in indices:
                item = dataset[i]
                for k in keys:
                    if k in item:
                        v = item[k]
                        if hasattr(v, 'numpy'):
                            v = v.numpy()
                        results[k].append(np.asarray(v, dtype=np.float64))
            out = {}
            for k in keys:
                if results[k]:
                    out[k] = np.stack(results[k], axis=0)
                else:
                    out[k] = np.array([], dtype=np.float64)
            return out

        for split_name, output, ds in [('test', test_output, test_dataset), ('val', val_output, val_dataset)]:
            preds_raw = output.predictions

            if isinstance(preds_raw, tuple):
                preds = preds_raw[0] if len(preds_raw) > 0 else preds_raw
            else:
                preds = preds_raw

            if hasattr(preds, 'cpu'):
                preds = preds.cpu().numpy()
            preds = np.asarray(preds, dtype=np.float64)

            extracted = _extract_from_dataset(ds, ['future_values', 'past_values'])
            labels = extracted['future_values']
            past_vals = extracted['past_values']

            logger.info(
                "TSFM评估[%s] preds shape=%s dtype=%s, labels shape=%s dtype=%s, past_vals shape=%s, n_targets=%s, close_idx=%s",
                split_name, preds.shape, preds.dtype, labels.shape, labels.dtype,
                past_vals.shape if past_vals.ndim >= 2 else 'N/A',
                n_targets, close_idx
            )

            if preds.ndim == 3 and labels.ndim == 3:
                n_steps = min(preds.shape[1], labels.shape[1])
                n_ch = min(preds.shape[2], labels.shape[2])
                preds_trimmed = preds[:, :n_steps, :n_ch]
                labels_trimmed = labels[:, :n_steps, :n_ch]
                ch = min(close_idx, n_ch - 1)
                close_pred = preds_trimmed[:, :, ch].flatten()
                close_true = labels_trimmed[:, :, ch].flatten()
            elif preds.ndim == 3 and labels.ndim == 2:
                close_pred = preds[:, :, min(close_idx, preds.shape[2] - 1)].flatten()
                close_true = labels.flatten()
            elif preds.ndim == 2 and labels.ndim == 2:
                close_pred = preds.flatten()
                close_true = labels.flatten()
            else:
                close_pred = preds.flatten()
                close_true = labels.flatten()

            mask = ~(np.isnan(close_pred) | np.isnan(close_true))
            close_pred = close_pred[mask]
            close_true = close_true[mask]

            if len(close_true) > 0:
                metrics[split_name]['mse'] = float(mean_squared_error(close_true, close_pred))
                metrics[split_name]['rmse'] = float(np.sqrt(metrics[split_name]['mse']))
                metrics[split_name]['mae'] = float(mean_absolute_error(close_true, close_pred))
                metrics[split_name]['r2'] = float(r2_score(close_true, close_pred))

                if preds.ndim == 3 and labels.ndim == 3 and past_vals.ndim == 3:
                    n_samples = min(preds_trimmed.shape[0], past_vals.shape[0])
                    p_ch = min(ch, past_vals.shape[2] - 1)
                    last_obs = past_vals[:n_samples, -1, p_ch]
                    pred_dir = preds_trimmed[:n_samples, :, ch] - last_obs[:, np.newaxis]
                    true_dir = labels_trimmed[:n_samples, :, ch] - last_obs[:, np.newaxis]
                    valid = ~(np.isnan(pred_dir) | np.isnan(true_dir))
                    pred_dir = pred_dir[valid]
                    true_dir = true_dir[valid]
                    if len(true_dir) > 0:
                        dir_acc = float(np.mean(np.sign(pred_dir) == np.sign(true_dir)))
                    else:
                        dir_acc = 0.0
                else:
                    nonzero = close_true != 0
                    if nonzero.sum() > 0:
                        dir_acc = float(np.mean(
                            np.sign(close_true[nonzero]) == np.sign(close_pred[nonzero])
                        ))
                    else:
                        dir_acc = 0.0
                metrics[split_name]['direction_accuracy'] = dir_acc

            if preds.ndim == 3 and labels.ndim == 3 and n_targets > 1:
                for ti, tname in enumerate(target_columns):
                    if ti >= preds_trimmed.shape[2] or ti >= labels_trimmed.shape[2]:
                        break
                    t_pred = preds_trimmed[:, :, ti].flatten()
                    t_true = labels_trimmed[:, :, ti].flatten()
                    t_mask = ~(np.isnan(t_pred) | np.isnan(t_true))
                    t_pred = t_pred[t_mask]
                    t_true = t_true[t_mask]
                    if len(t_true) > 0:
                        metrics[split_name][f'{tname}_mse'] = float(mean_squared_error(t_true, t_pred))
                        metrics[split_name][f'{tname}_mae'] = float(mean_absolute_error(t_true, t_pred))
                        metrics[split_name][f'{tname}_r2'] = float(r2_score(t_true, t_pred))

        if hasattr(test_output, 'metrics') and test_output.metrics:
            evals_result['test'] = {'loss': [float(test_output.metrics.get('test_loss', 0))]}
        if hasattr(val_output, 'metrics') and val_output.metrics:
            evals_result['val'] = {'loss': [float(val_output.metrics.get('test_loss', 0))]}

        try:
            log_history = trainer.state.log_history
            train_losses = []
            val_losses = []
            for entry in log_history:
                if 'loss' in entry and 'epoch' in entry:
                    train_losses.append(float(entry['loss']))
                if 'eval_loss' in entry:
                    val_losses.append(float(entry['eval_loss']))
            if train_losses:
                evals_result['train'] = {'loss': train_losses}
            if val_losses:
                evals_result.setdefault('val', {})['loss'] = val_losses
        except Exception:
            pass

    except Exception as e:
        import traceback
        logger.warning("TSFM评估失败: %s\n%s", e, traceback.format_exc())

    with training_sessions_lock:
        training_sessions[session_id]['progress'] = 90
        training_sessions[session_id]['current_step'] = 'saving'
        training_sessions[session_id]['message'] = '正在保存TSFM模型...'

    _assert_training_not_timeout(session_id)

    model_id = (
        f"tsfm_{label_type}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    )
    model_dir = MODELS_DIR / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    hf_model_dir = model_dir / "hf_model"
    model.save_pretrained(str(hf_model_dir))

    preprocessor_path = model_dir / "preprocessor.pkl"
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(tsp, f)

    save_config = {
        'model_id': model_id,
        'model_type': 'tsfm',
        'task_type': 'regression',
        'label_type': label_type,
        'predict_step': forecast_length,
        'loss_function': loss_function,
        'loss_params': config.get('loss_params', {}),
        'train_ratio': train_ratio,
        'test_ratio': test_ratio,
        'val_ratio': val_ratio,
        'feature_names': target_columns + control_columns,
        'feature_name_mapping': col_mapping,
        'label_name': 'Close',
        'model_params': {
            'context_length': context_length,
            'forecast_length': forecast_length,
            'num_train_epochs': num_train_epochs,
            'per_device_train_batch_size': batch_size,
            'learning_rate': tsfm_lr,
            'weight_decay': tsfm_weight_decay,
            'freeze_backbone': freeze_backbone,
            'decoder_mode': decoder_mode,
            'pretrained_model_id': pretrained_model_id,
            'trainable_params': trainable_params,
            'total_params': total_params,
        },
        'data_file': str(file_path),
        'start_date': start_date,
        'end_date': end_date,
        'factor_ids': factor_ids,
        'failed_factors': [],
        'created_at': datetime.now().isoformat(),
        'data_info': training_sessions[session_id].get('data_info', {}),
        'metrics': metrics,
        'model_storage': 'tsfm_hf',
        'model_file': 'hf_model',
        'scaler_file': None,
        'preprocessor_file': 'preprocessor.pkl',
        'column_specifiers': column_specifiers,
        'target_columns': target_columns,
        'control_columns': control_columns,
        'extra_joined_columns': merged_extra_meta.get('extra_columns', []),
        'loss_function_note': (
            'TSFM模型使用HuggingFace Trainer微调，'
            f'损失函数为{loss_function}，'
            f'骨干{"已冻结(少样本微调)" if freeze_backbone else "未冻结(全量微调)"}'
        ),
    }
    if evals_result:
        save_config['evals_result'] = evals_result
    with open(model_dir / "config.json", 'w', encoding='utf-8') as f:
        json.dump(save_config, f, ensure_ascii=False, indent=2, default=str)

    try:
        import shutil
        if temp_output_dir.exists():
            shutil.rmtree(temp_output_dir, ignore_errors=True)
    except Exception:
        pass

    with training_sessions_lock:
        training_sessions[session_id]['status'] = 'completed'
        training_sessions[session_id]['progress'] = 100
        training_sessions[session_id]['current_step'] = 'completed'
        training_sessions[session_id]['message'] = '训练完成'
        training_sessions[session_id]['results'] = {
            'model_id': model_id,
            'model_type': 'tsfm',
            'task_type': 'regression',
            'label_type': label_type,
            'predict_step': forecast_length,
            'loss_function': loss_function,
            'metrics': metrics,
            'feature_importance': {},
            'data_info': training_sessions[session_id].get('data_info', {}),
            'model_dir': str(model_dir),
            'evals_result': save_config.get('evals_result'),
        }
        training_sessions[session_id]['completed_time'] = datetime.now().isoformat()
        training_sessions[session_id]['completed_ts'] = time.time()


def _train_model(session_id, config):
    try:
        with training_sessions_lock:
            training_sessions[session_id]['status'] = 'running'
            training_sessions[session_id]['progress'] = 5
            training_sessions[session_id]['current_step'] = 'data_loading'
            training_sessions[session_id]['message'] = '正在加载数据...'

        _assert_training_not_timeout(session_id)
        file_path = _validate_training_data_file_path(config['file_path'])
        start_date = config.get('start_date')
        end_date = config.get('end_date')
        factor_ids = config.get('factor_ids', [])
        label_type = config.get('label_type', 'log_return')
        predict_step = config.get('predict_step', 1)
        model_type = config.get('model_type', 'lightgbm')
        task_type = config.get('task_type', 'regression')
        loss_function = config.get('loss_function', 'mse')
        train_ratio = config.get('train_ratio', 0.8)
        test_ratio = config.get('test_ratio', 0.1)
        val_ratio = config.get('val_ratio', 0.1)
        model_params = config.get('model_params', {})
        loss_params = config.get('loss_params', {})
        train_ratio, test_ratio, val_ratio = _normalize_and_validate_split_ratios(
            train_ratio, test_ratio, val_ratio
        )

        df = _load_feather_data(str(file_path), start_date, end_date)
        if df is None or len(df) == 0:
            raise ValueError("数据加载失败或数据为空")

        if model_type == 'tsfm':
            _train_tsfm_model(session_id, config, df, file_path)
            return

        if predict_step >= len(df):
            raise ValueError(
                f"predict_step({predict_step}) 不能大于等于样本数({len(df)})，请减小预测步长或扩大数据范围"
            )

        with training_sessions_lock:
            training_sessions[session_id]['progress'] = 20
            training_sessions[session_id]['current_step'] = 'feature_building'
            training_sessions[session_id]['message'] = '正在计算因子特征...'
        _assert_training_not_timeout(session_id)

        failed_factors = []
        if factor_ids:
            from factor_miner.core.factor_engine import get_global_engine
            engine = get_global_engine()
            feature_df, failed_factors = _build_features(df, factor_ids, engine)
            if feature_df is None or feature_df.empty:
                reasons = "; ".join(
                    f"{item['factor_id']}: {item['reason']}" for item in failed_factors[:5]
                )
                extra = f"（前5个失败原因: {reasons}）" if reasons else ""
                raise ValueError(
                    "选中的因子未生成有效特征（可能因子计算失败或结果全为空），"
                    "请减少因子数量或改用基础特征重试" + extra
                )
            with training_sessions_lock:
                training_sessions[session_id]['failed_factors'] = failed_factors
        else:
            ohlcv_candidates = {
                'open': ['open', 'opn', 'o'],
                'high': ['high', 'hi', 'h'],
                'low': ['low', 'lo', 'l'],
                'close': ['close', 'cl', 'c'],
                'volume': ['volume', 'vol', 'v'],
            }
            feature_cols = []
            for col in df.columns:
                cl = col.lower()
                matched = False
                for field, tokens in ohlcv_candidates.items():
                    if any(t == cl for t in tokens):
                        matched = True
                        break
                if not matched and cl not in ('time', 'date', 'timestamp', 'datetime', 'open_time', 'opentime', 'close_time', 'closetime'):
                    if _is_potential_leakage_feature(col):
                        continue
                    feature_cols.append(col)
            if not feature_cols:
                for col in df.columns:
                    cl = col.lower()
                    if cl not in ('time', 'date', 'timestamp', 'datetime', 'open_time', 'opentime', 'close_time', 'closetime'):
                        if _is_potential_leakage_feature(col):
                            continue
                        feature_cols.append(col)
            feature_df = df[feature_cols].select_dtypes(include=[np.number]).copy()
            if feature_df is None or feature_df.empty:
                raise ValueError("未找到可用的数值特征列，请检查数据文件内容")

        with training_sessions_lock:
            training_sessions[session_id]['progress'] = 35
            training_sessions[session_id]['current_step'] = 'label_building'
            training_sessions[session_id]['message'] = '正在构建标签...'
        _assert_training_not_timeout(session_id)

        label_series = _build_label(df, label_type, predict_step)
        label_col = label_series.name
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        merged_df = feature_df.join(label_series, how='inner')
        raw_rows = len(merged_df)
        if raw_rows == 0:
            raise ValueError("特征与标签时间索引无法对齐，请检查数据时间列与日期范围")

        feature_cols = [c for c in merged_df.columns if c != label_col]
        min_non_na = max(5, int(raw_rows * 0.05))
        valid_feature_cols = [
            c for c in feature_cols
            if int(merged_df[c].notna().sum()) >= min_non_na
        ]
        if not valid_feature_cols:
            raise ValueError(
                f"所有特征列缺失值过多（raw_rows={raw_rows}, min_non_na={min_non_na}），请减少因子数量或扩大数据范围"
            )
        dropped_feature_cols = len(feature_cols) - len(valid_feature_cols)
        if dropped_feature_cols > 0:
            logger.warning("因缺失值过多移除特征列: %s", dropped_feature_cols)

        merged_df = merged_df[valid_feature_cols + [label_col]].dropna()
        sanitized_cols, feature_name_mapping = _sanitize_feature_names(valid_feature_cols)
        rename_map = dict(zip(valid_feature_cols, sanitized_cols))
        merged_df = merged_df.rename(columns=rename_map)
        feature_df = merged_df

        if len(feature_df) < 50:
            raise ValueError(
                f"有效样本数不足: {len(feature_df)} < 50，请扩大数据范围或减少因子数量；"
                f"可尝试减小predict_step({predict_step})、减少因子数、或放宽日期范围"
            )

        with training_sessions_lock:
            training_sessions[session_id]['progress'] = 45
            training_sessions[session_id]['current_step'] = 'data_splitting'
            training_sessions[session_id]['message'] = '正在划分数据集...'
        _assert_training_not_timeout(session_id)

        train_df, val_df, test_df = _split_data_chronological(
            feature_df, train_ratio=train_ratio, val_ratio=val_ratio
        )
        if len(train_df) == 0 or len(test_df) == 0 or len(val_df) == 0:
            raise ValueError(
                f"划分后数据为空，请调整比例。train={len(train_df)}, test={len(test_df)}, val={len(val_df)}"
            )
        min_split_samples = 10
        for name, split_df in [('训练集', train_df), ('测试集', test_df), ('验证集', val_df)]:
            if len(split_df) < min_split_samples:
                raise ValueError(
                    f"{name}样本数过少({len(split_df)}<{min_split_samples})，请扩大数据范围或调整划分比例"
                )

        X_train = train_df.drop(columns=[label_col])
        y_train = train_df[label_col]
        X_test = test_df.drop(columns=[label_col])
        y_test = test_df[label_col]
        X_val = val_df.drop(columns=[label_col])
        y_val = val_df[label_col]

        is_classification = (task_type == 'classification') or (label_type == 'direction')
        if is_classification:
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)
            y_val = y_val.astype(int)
            if loss_function != 'log_loss':
                logger.warning("分类任务仅支持log_loss，已自动覆盖原loss_function=%s", loss_function)
                loss_function = 'log_loss'

        is_tree_model = model_type in ('lightgbm', 'xgboost', 'random_forest')
        if is_tree_model:
            scaler = None
            X_train_scaled = X_train
            X_test_scaled = X_test
            X_val_scaled = X_val
        else:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_val_scaled = scaler.transform(X_val)

        with training_sessions_lock:
            training_sessions[session_id]['progress'] = 55
            training_sessions[session_id]['current_step'] = 'model_training'
            training_sessions[session_id]['message'] = f'正在训练 {model_type} 模型...'
            training_sessions[session_id]['data_info'] = {
                'total_samples': len(feature_df),
                'train_samples': len(train_df),
                'test_samples': len(test_df),
                'val_samples': len(val_df),
                'feature_count': X_train.shape[1],
                'feature_names': list(X_train.columns),
                'label_name': label_col,
                'train_range': f"{train_df.index[0]} ~ {train_df.index[-1]}" if len(train_df) > 0 else "",
                'test_range': f"{test_df.index[0]} ~ {test_df.index[-1]}" if len(test_df) > 0 else "",
                'val_range': f"{val_df.index[0]} ~ {val_df.index[-1]}" if len(val_df) > 0 else "",
            }

        model = None
        evals_result = {}
        _assert_training_not_timeout(session_id)

        if model_type == 'lightgbm':
            import lightgbm as lgb

            objective = 'binary' if is_classification else 'regression'
            metric = 'binary_logloss' if is_classification else 'mse'

            params = {
                'objective': objective,
                'metric': metric,
                'verbosity': -1,
                'seed': 42,
            }
            lgb_train_rounds = int(model_params.get('n_estimators', 200))
            lgb_es_rounds = int(model_params.get('early_stopping_rounds', 20))
            lgb_model_params = {
                k: v for k, v in model_params.items()
                if k not in ('n_estimators', 'early_stopping_rounds')
            }
            params.update(lgb_model_params)

            train_data = lgb.Dataset(X_train_scaled, label=y_train.values, feature_name=list(X_train.columns))
            val_data = lgb.Dataset(X_val_scaled, label=y_val.values, feature_name=list(X_train.columns), reference=train_data)

            custom_eval = None
            if not is_classification:
                if loss_function == 'direction_aware_mse':
                    lambda_ = loss_params.get('lambda', 2.0)
                    custom_eval = lambda y_pred, dataset: _custom_eval_metric_direction_aware(y_pred, dataset, lambda_)
                elif loss_function == 'composite':
                    alpha = loss_params.get('alpha', 1.0)
                    beta = loss_params.get('beta', 1.0)
                    k = loss_params.get('k', 5.0)
                    custom_eval = lambda y_pred, dataset: _custom_eval_metric_composite(y_pred, dataset, alpha, beta, k)
                elif loss_function == 'magnitude_weighted':
                    lambda_ = loss_params.get('lambda', 2.0)
                    custom_eval = lambda y_pred, dataset: _custom_eval_metric_magnitude_weighted(y_pred, dataset, lambda_)
                elif loss_function == 'mse_hinge':
                    alpha = loss_params.get('alpha', 1.0)
                    beta = loss_params.get('beta', 1.0)
                    custom_eval = lambda y_pred, dataset: _custom_eval_metric_mse_hinge(y_pred, dataset, alpha, beta)

            def _lgb_iter_callback(env):
                _assert_training_not_timeout(session_id)
                if lgb_train_rounds > 0:
                    iter_pct = min(1.0, (env.iteration + 1) / lgb_train_rounds)
                    mapped = 55 + int(iter_pct * 25)
                    with training_sessions_lock:
                        if session_id in training_sessions:
                            training_sessions[session_id]['progress'] = mapped
                            training_sessions[session_id]['message'] = (
                                f'LightGBM 训练中 ({env.iteration + 1}/{lgb_train_rounds})'
                            )

            callbacks = [
                lgb.log_evaluation(0),
                lgb.record_evaluation(evals_result),
                lgb.early_stopping(stopping_rounds=lgb_es_rounds, verbose=False),
                _lgb_iter_callback,
            ]

            model = lgb.train(
                params,
                train_data,
                num_boost_round=lgb_train_rounds,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'val'],
                feval=custom_eval,
                callbacks=callbacks,
            )

        elif model_type == 'xgboost':
            import xgboost as xgb

            objective = 'binary:logistic' if is_classification else 'reg:squarederror'

            params = {
                'objective': objective,
                'seed': 42,
                'verbosity': 0,
            }
            xgb_model_params = {
                k: v for k, v in model_params.items()
                if k not in ('n_estimators', 'early_stopping_rounds')
            }
            if xgb_model_params.get('max_depth', None) == -1:
                xgb_model_params['max_depth'] = 0
            params.update(xgb_model_params)

            dtrain = xgb.DMatrix(X_train_scaled, label=y_train.values, feature_names=list(X_train.columns))
            dtest = xgb.DMatrix(X_test_scaled, label=y_test.values, feature_names=list(X_test.columns))
            dval = xgb.DMatrix(X_val_scaled, label=y_val.values, feature_names=list(X_val.columns))

            custom_feval = None
            if not is_classification:
                if loss_function == 'direction_aware_mse':
                    lambda_ = loss_params.get('lambda', 2.0)
                    custom_feval = lambda y_pred, dmat: _custom_eval_metric_direction_aware(y_pred, dmat, lambda_)
                elif loss_function == 'composite':
                    alpha = loss_params.get('alpha', 1.0)
                    beta = loss_params.get('beta', 1.0)
                    k = loss_params.get('k', 5.0)
                    custom_feval = lambda y_pred, dmat: _custom_eval_metric_composite(y_pred, dmat, alpha, beta, k)
                elif loss_function == 'magnitude_weighted':
                    lambda_ = loss_params.get('lambda', 2.0)
                    custom_feval = lambda y_pred, dmat: _custom_eval_metric_magnitude_weighted(y_pred, dmat, lambda_)
                elif loss_function == 'mse_hinge':
                    alpha = loss_params.get('alpha', 1.0)
                    beta = loss_params.get('beta', 1.0)
                    custom_feval = lambda y_pred, dmat: _custom_eval_metric_mse_hinge(y_pred, dmat, alpha, beta)

            xgb_num_rounds = int(model_params.get('n_estimators', 200))

            class _XgbProgressCallback(xgb.callback.TrainingCallback):
                def after_iteration(self, model, epoch, evals_log):
                    _assert_training_not_timeout(session_id)
                    if xgb_num_rounds > 0:
                        iter_pct = min(1.0, (epoch + 1) / xgb_num_rounds)
                        mapped = 55 + int(iter_pct * 25)
                        with training_sessions_lock:
                            if session_id in training_sessions:
                                training_sessions[session_id]['progress'] = mapped
                                training_sessions[session_id]['message'] = (
                                    f'XGBoost 训练中 ({epoch + 1}/{xgb_num_rounds})'
                                )
                    return False

            xgb_train_kwargs = dict(
                params=params,
                dtrain=dtrain,
                num_boost_round=xgb_num_rounds,
                evals=[(dtrain, 'train'), (dval, 'val')],
                verbose_eval=False,
                evals_result=evals_result,
                callbacks=[_XgbProgressCallback()],
            )
            # xgboost >=1.6 将 feval / early_stopping_rounds 移至 callback/参数可选项，
            # 这里做版本兼容（优先使用 kwargs，失败时降级）
            try:
                model = xgb.train(
                    **xgb_train_kwargs,
                    custom_metric=custom_feval,
                    early_stopping_rounds=model_params.get('early_stopping_rounds', 20),
                )
            except TypeError:
                model = xgb.train(
                    **xgb_train_kwargs,
                    feval=custom_feval,
                    early_stopping_rounds=model_params.get('early_stopping_rounds', 20),
                )

        elif model_type == 'logistic_regression':
            _assert_training_not_timeout(session_id)
            if is_classification:
                model = LogisticRegression(
                    max_iter=model_params.get('max_iter', 1000),
                    C=model_params.get('C', 1.0),
                    random_state=42,
                )
            else:
                model = Ridge(
                    alpha=model_params.get('alpha', 1.0),
                    random_state=42,
                )
            model.fit(X_train_scaled, y_train.values)

        elif model_type == 'random_forest':
            _assert_training_not_timeout(session_id)
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            if is_classification:
                model = RandomForestClassifier(
                    n_estimators=model_params.get('n_estimators', 200),
                    max_depth=model_params.get('max_depth', None),
                    random_state=42,
                    n_jobs=-1,
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=model_params.get('n_estimators', 200),
                    max_depth=model_params.get('max_depth', None),
                    random_state=42,
                    n_jobs=-1,
                )
            model.fit(X_train_scaled, y_train.values)

        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        with training_sessions_lock:
            training_sessions[session_id]['progress'] = 80
            training_sessions[session_id]['current_step'] = 'evaluation'
            training_sessions[session_id]['message'] = '正在评估模型...'

        if is_classification:
            if model_type == 'lightgbm':
                y_score_test = model.predict(X_test_scaled, num_iteration=model.best_iteration)
                y_score_val = model.predict(X_val_scaled, num_iteration=model.best_iteration)
            elif model_type == 'xgboost':
                y_score_test = model.predict(dtest)
                y_score_val = model.predict(dval)
            else:
                if hasattr(model, 'predict_proba'):
                    y_score_test = model.predict_proba(X_test_scaled)[:, 1]
                    y_score_val = model.predict_proba(X_val_scaled)[:, 1]
                elif hasattr(model, 'decision_function'):
                    test_raw = model.decision_function(X_test_scaled)
                    val_raw = model.decision_function(X_val_scaled)
                    y_score_test = 1.0 / (1.0 + np.exp(-test_raw))
                    y_score_val = 1.0 / (1.0 + np.exp(-val_raw))
                else:
                    y_score_test = model.predict(X_test_scaled)
                    y_score_val = model.predict(X_val_scaled)

            y_pred_test = np.asarray(y_score_test, dtype=float)
            y_pred_val = np.asarray(y_score_val, dtype=float)
        else:
            if model_type == 'lightgbm':
                y_pred_test = model.predict(X_test_scaled, num_iteration=model.best_iteration)
                y_pred_val = model.predict(X_val_scaled, num_iteration=model.best_iteration)
            elif model_type == 'xgboost':
                y_pred_test = model.predict(dtest)
                y_pred_val = model.predict(dval)
            else:
                y_pred_test = model.predict(X_test_scaled)
                y_pred_val = model.predict(X_val_scaled)

        metrics = {}
        if is_classification:
            y_pred_test_cls = (y_pred_test > 0.5).astype(int)
            y_pred_val_cls = (y_pred_val > 0.5).astype(int)

            metrics['test'] = {
                'accuracy': float(accuracy_score(y_test, y_pred_test_cls)),
                'precision': float(precision_score(y_test, y_pred_test_cls, zero_division=0)),
                'recall': float(recall_score(y_test, y_pred_test_cls, zero_division=0)),
                'f1': float(f1_score(y_test, y_pred_test_cls, zero_division=0)),
            }
            metrics['val'] = {
                'accuracy': float(accuracy_score(y_val, y_pred_val_cls)),
                'precision': float(precision_score(y_val, y_pred_val_cls, zero_division=0)),
                'recall': float(recall_score(y_val, y_pred_val_cls, zero_division=0)),
                'f1': float(f1_score(y_val, y_pred_val_cls, zero_division=0)),
            }
            try:
                metrics['test']['auc'] = float(roc_auc_score(y_test, y_pred_test))
                metrics['val']['auc'] = float(roc_auc_score(y_val, y_pred_val))
            except Exception:
                pass
            try:
                metrics['test']['log_loss'] = float(log_loss(y_test, y_pred_test))
                metrics['val']['log_loss'] = float(log_loss(y_val, y_pred_val))
            except Exception:
                pass
        else:
            metrics['test'] = {
                'mse': float(mean_squared_error(y_test, y_pred_test)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
                'mae': float(mean_absolute_error(y_test, y_pred_test)),
                'r2': float(r2_score(y_test, y_pred_test)),
            }
            metrics['val'] = {
                'mse': float(mean_squared_error(y_val, y_pred_val)),
                'rmse': float(np.sqrt(mean_squared_error(y_val, y_pred_val))),
                'mae': float(mean_absolute_error(y_val, y_pred_val)),
                'r2': float(r2_score(y_val, y_pred_val)),
            }

            nonzero_test = y_test.values != 0
            if nonzero_test.sum() > 0:
                direction_acc_test = float(np.mean(
                    np.sign(y_test.values[nonzero_test]) == np.sign(y_pred_test[nonzero_test])
                ))
            else:
                direction_acc_test = 0.0
            nonzero_val = y_val.values != 0
            if nonzero_val.sum() > 0:
                direction_acc_val = float(np.mean(
                    np.sign(y_val.values[nonzero_val]) == np.sign(y_pred_val[nonzero_val])
                ))
            else:
                direction_acc_val = 0.0
            metrics['test']['direction_accuracy'] = direction_acc_test
            metrics['val']['direction_accuracy'] = direction_acc_val

            if loss_function == 'direction_aware_mse':
                lambda_ = loss_params.get('lambda', 2.0)
                metrics['test']['direction_aware_mse'] = float(DirectionAwareMSELoss.compute(y_test.values, y_pred_test, lambda_))
                metrics['val']['direction_aware_mse'] = float(DirectionAwareMSELoss.compute(y_val.values, y_pred_val, lambda_))
            elif loss_function == 'composite':
                alpha = loss_params.get('alpha', 1.0)
                beta = loss_params.get('beta', 1.0)
                k = loss_params.get('k', 5.0)
                metrics['test']['composite_loss'] = float(CompositeLoss.compute(y_test.values, y_pred_test, alpha, beta, k))
                metrics['val']['composite_loss'] = float(CompositeLoss.compute(y_val.values, y_pred_val, alpha, beta, k))
            elif loss_function == 'magnitude_weighted':
                lambda_ = loss_params.get('lambda', 2.0)
                metrics['test']['magnitude_weighted_loss'] = float(MagnitudeWeightedDirectionLoss.compute(y_test.values, y_pred_test, lambda_))
                metrics['val']['magnitude_weighted_loss'] = float(MagnitudeWeightedDirectionLoss.compute(y_val.values, y_pred_val, lambda_))
            elif loss_function == 'mse_hinge':
                alpha = loss_params.get('alpha', 1.0)
                beta = loss_params.get('beta', 1.0)
                metrics['test']['mse_hinge_loss'] = float(MSEHingeLoss.compute(y_test.values, y_pred_test, alpha, beta))
                metrics['val']['mse_hinge_loss'] = float(MSEHingeLoss.compute(y_val.values, y_pred_val, alpha, beta))

        feature_importance = {}
        if model_type == 'lightgbm':
            imp = model.feature_importance(importance_type='gain')
            for fname, iv in zip(X_train.columns, imp):
                feature_importance[fname] = float(iv)
        elif model_type == 'xgboost':
            imp = model.get_score(importance_type='gain')
            for fname in X_train.columns:
                feature_importance[fname] = float(imp.get(fname, 0.0))
        elif model_type in ('random_forest',):
            imp = model.feature_importances_
            for fname, iv in zip(X_train.columns, imp):
                feature_importance[fname] = float(iv)
        elif model_type in ('logistic_regression',):
            if hasattr(model, 'coef_'):
                imp = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                for fname, iv in zip(X_train.columns, imp):
                    feature_importance[fname] = float(iv)

        sorted_imp = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        feature_importance = dict(sorted_imp)

        with training_sessions_lock:
            training_sessions[session_id]['progress'] = 90
            training_sessions[session_id]['current_step'] = 'saving'
            training_sessions[session_id]['message'] = '正在保存模型...'
        _assert_training_not_timeout(session_id)

        model_id = (
            f"{model_type}_{label_type}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        )
        model_dir = MODELS_DIR / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        model_storage = 'pickle'
        model_path = model_dir / "model.pkl"
        scaler_path = model_dir / "scaler.pkl"
        config_path = model_dir / "config.json"

        if model_type == 'lightgbm':
            model_storage = 'lightgbm_booster'
            model_path = model_dir / "model.txt"
            model.save_model(str(model_path))
            scaler = None
        elif model_type == 'xgboost':
            model_storage = 'xgboost_booster'
            model_path = model_dir / "model.json"
            model.save_model(str(model_path))
            scaler = None
        else:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)

        save_config = {
            'model_id': model_id,
            'model_type': model_type,
            'task_type': task_type,
            'label_type': label_type,
            'predict_step': predict_step,
            'loss_function': loss_function,
            'loss_params': loss_params,
            'train_ratio': train_ratio,
            'test_ratio': test_ratio,
            'val_ratio': val_ratio,
            'feature_names': list(X_train.columns),
            'feature_name_mapping': feature_name_mapping,
            'label_name': label_col,
            'model_params': model_params,
            'data_file': str(file_path),
            'start_date': start_date,
            'end_date': end_date,
            'factor_ids': factor_ids,
            'failed_factors': failed_factors,
            'created_at': datetime.now().isoformat(),
            'data_info': training_sessions[session_id].get('data_info', {}),
            'metrics': metrics,
            'model_storage': model_storage,
            'model_file': model_path.name,
            'scaler_file': scaler_path.name if scaler_path.exists() else None,
            'loss_function_note': (
                '自定义损失当前作为评估指标/early stopping参考，'
                '树模型训练目标仍使用各模型默认回归objective'
                if (not is_classification and loss_function != 'mse') else ''
            ),
        }
        if evals_result:
            serializable_evals = {}
            for dataset_name, dataset_metrics in evals_result.items():
                serializable_evals[dataset_name] = {}
                for metric_name, values in dataset_metrics.items():
                    serializable_evals[dataset_name][metric_name] = [float(v) for v in values]
            save_config['evals_result'] = serializable_evals
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(save_config, f, ensure_ascii=False, indent=2, default=str)

        pred_test_df = pd.DataFrame({
            'y_true': y_test.values,
            'y_pred': y_pred_test,
        }, index=y_test.index)
        pred_test_path = model_dir / "predictions_test.csv"
        pred_test_df.to_csv(pred_test_path)

        pred_val_df = pd.DataFrame({
            'y_true': y_val.values,
            'y_pred': y_pred_val,
        }, index=y_val.index)
        pred_val_path = model_dir / "predictions_val.csv"
        pred_val_df.to_csv(pred_val_path)

        with training_sessions_lock:
            training_sessions[session_id]['status'] = 'completed'
            training_sessions[session_id]['progress'] = 100
            training_sessions[session_id]['current_step'] = 'completed'
            training_sessions[session_id]['message'] = '训练完成'
            training_sessions[session_id]['results'] = {
                'model_id': model_id,
                'model_type': model_type,
                'task_type': task_type,
                'label_type': label_type,
                'predict_step': predict_step,
                'loss_function': loss_function,
                'metrics': metrics,
                'feature_importance': feature_importance,
                'data_info': training_sessions[session_id].get('data_info', {}),
                'model_dir': str(model_dir),
                'evals_result': save_config.get('evals_result'),
            }
            training_sessions[session_id]['completed_time'] = datetime.now().isoformat()
            training_sessions[session_id]['completed_ts'] = time.time()

    except TrainingCancelled as e:
        logger.info(f"训练任务已被用户取消: {e}")
        with training_sessions_lock:
            if session_id in training_sessions:
                training_sessions[session_id]['status'] = 'cancelled'
                training_sessions[session_id]['current_step'] = 'cancelled'
                training_sessions[session_id]['message'] = '训练已取消'
                training_sessions[session_id]['error'] = str(e)
                training_sessions[session_id]['completed_ts'] = time.time()
                training_sessions[session_id]['completed_time'] = datetime.now().isoformat()
    except Exception as e:
        logger.error(f"训练任务失败: {e}")
        import traceback
        traceback.print_exc()
        with training_sessions_lock:
            if session_id in training_sessions:
                training_sessions[session_id]['status'] = 'failed'
                training_sessions[session_id]['progress'] = 0
                training_sessions[session_id]['current_step'] = 'failed'
                training_sessions[session_id]['message'] = f'训练失败: {str(e)}'
                training_sessions[session_id]['error'] = str(e)
                training_sessions[session_id]['completed_ts'] = time.time()
                training_sessions[session_id]['completed_time'] = datetime.now().isoformat()


@bp.route('/models', methods=['GET'])
def get_available_models():
    models = [
        {'id': 'lightgbm', 'name': 'LightGBM', 'type': 'tree', 'description': '轻量级梯度提升树，速度快，适合大规模数据', 'supports': ['regression', 'classification']},
        {'id': 'xgboost', 'name': 'XGBoost', 'type': 'tree', 'description': '极端梯度提升树，精度高，支持自定义评估指标', 'supports': ['regression', 'classification']},
        {'id': 'logistic_regression', 'name': '逻辑回归/Ridge', 'type': 'linear', 'description': '线性模型，分类用逻辑回归，回归用Ridge', 'supports': ['regression', 'classification']},
        {'id': 'random_forest', 'name': '随机森林', 'type': 'tree', 'description': '集成学习方法，鲁棒性强', 'supports': ['regression', 'classification']},
        {'id': 'tsfm', 'name': 'TSFM (Granite TTM)', 'type': 'foundation', 'description': 'IBM Granite时序基础模型，少样本微调，5%-10%数据即可适应BTC波动', 'supports': ['regression'], 'available': TSFM_AVAILABLE},
    ]
    return jsonify({'success': True, 'models': models})


@bp.route('/loss-functions', methods=['GET'])
def get_loss_functions():
    return jsonify({'success': True, 'loss_functions': LOSS_FUNCTIONS})


@bp.route('/label-types', methods=['GET'])
def get_label_types():
    labels = [
        {'id': 'log_return', 'name': '对数收益率', 'description': 'ln(P_{t+n}/P_t)，连续收益率', 'task': 'regression'},
        {'id': 'direction', 'name': '收益方向', 'description': '涨=1, 跌=0，二分类标签', 'task': 'classification'},
        {'id': 'composite', 'name': '综合(方向×幅度)', 'description': '方向正确时放大收益，方向错误时惩罚', 'task': 'regression'},
    ]
    return jsonify({'success': True, 'label_types': labels})


@bp.route('/start', methods=['POST'])
def start_training():
    try:
        data = request.get_json() or {}

        file_path = data.get('file_path')
        if not file_path:
            return jsonify({'success': False, 'error': '缺少文件路径'}), 400
        try:
            validated_file_path = _validate_training_data_file_path(file_path)
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)}), 400

        session_id = str(uuid.uuid4())
        default_timeout = int(current_app.config.get(
            'TRAINING_TIMEOUT_SECONDS', DEFAULT_TRAINING_TIMEOUT_SECONDS
        ))
        try:
            timeout_seconds = int(data.get('timeout_seconds') or default_timeout)
        except (TypeError, ValueError):
            timeout_seconds = default_timeout
        if timeout_seconds > 0:
            timeout_seconds = max(60, min(timeout_seconds, 24 * 3600))
        else:
            timeout_seconds = 0

        config = {
            'file_path': str(validated_file_path),
            'start_date': data.get('start_date'),
            'end_date': data.get('end_date'),
            'factor_ids': data.get('factor_ids', []),
            'label_type': data.get('label_type', 'log_return'),
            'predict_step': data.get('predict_step', 1),
            'model_type': data.get('model_type', 'lightgbm'),
            'task_type': data.get('task_type', 'regression'),
            'loss_function': data.get('loss_function', 'mse'),
            'train_ratio': data.get('train_ratio', 0.8),
            'test_ratio': data.get('test_ratio', 0.1),
            'val_ratio': data.get('val_ratio', 0.1),
            'model_params': data.get('model_params', {}),
            'loss_params': data.get('loss_params', {}),
        }
        try:
            config['train_ratio'], config['test_ratio'], config['val_ratio'] = _normalize_and_validate_split_ratios(
                config['train_ratio'], config['test_ratio'], config['val_ratio']
            )
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)}), 400

        with training_sessions_lock:
            _cleanup_training_sessions_locked()
            max_concurrent = int(current_app.config.get(
                'TRAINING_MAX_CONCURRENT', DEFAULT_MAX_CONCURRENT_TRAININGS
            ))
            running_count = sum(
                1 for s in training_sessions.values() if s.get('status') in ('pending', 'running')
            )
            if running_count >= max_concurrent:
                return jsonify({
                    'success': False,
                    'error': f'当前有{running_count}个训练任务正在运行，请稍后再试'
                }), 429
            training_sessions[session_id] = {
                'status': 'pending',
                'progress': 0,
                'current_step': 'initializing',
                'message': '正在初始化训练任务...',
                'start_time': datetime.now().isoformat(),
                'start_ts': time.time(),
                'timeout_seconds': timeout_seconds,
                'config': config,
            }

        app = current_app._get_current_object()

        def _train_with_app_context(sid, cfg):
            with app.app_context():
                _train_model(sid, cfg)

        thread = threading.Thread(target=_train_with_app_context, args=(session_id, config))
        thread.daemon = True
        thread.start()

        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': '训练任务已启动'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/status/<session_id>', methods=['GET'])
def get_training_status(session_id):
    with training_sessions_lock:
        _cleanup_training_sessions_locked()
        if session_id not in training_sessions:
            return jsonify({'success': False, 'error': '会话不存在'}), 404
        session = copy.deepcopy(training_sessions[session_id])
    return jsonify({
        'success': True,
        'status': session['status'],
        'progress': session['progress'],
        'current_step': session['current_step'],
        'message': session['message'],
        'start_time': session.get('start_time'),
        'completed_time': session.get('completed_time'),
        'data_info': session.get('data_info'),
        'cancel_requested': bool(session.get('cancel_requested')),
        'failed_factors': session.get('failed_factors') or [],
    })


@bp.route('/active-sessions', methods=['GET'])
def get_active_sessions():
    with training_sessions_lock:
        _cleanup_training_sessions_locked()
        active = []
        for sid, session in training_sessions.items():
            if session.get('status') in ('pending', 'running'):
                active.append({
                    'session_id': sid,
                    'status': session.get('status'),
                    'progress': session.get('progress', 0),
                    'current_step': session.get('current_step', ''),
                    'message': session.get('message', ''),
                    'start_time': session.get('start_time'),
                    'model_type': session.get('config', {}).get('model_type', ''),
                })
    return jsonify({'success': True, 'sessions': active})


@bp.route('/cancel/<session_id>', methods=['POST'])
def cancel_training(session_id):
    """请求取消指定的训练会话。

    取消是合作式的：依赖 lightgbm/xgboost 的迭代回调在下一轮检测到
    ``cancel_requested`` 后主动抛出 ``TrainingCancelled``。
    对已完成/失败/取消的会话调用此接口将返回幂等成功。
    """
    with training_sessions_lock:
        session = training_sessions.get(session_id)
        if not session:
            return jsonify({'success': False, 'error': '会话不存在'}), 404
        current_status = session.get('status')
        if current_status in ('completed', 'failed', 'cancelled'):
            return jsonify({
                'success': True,
                'message': f'会话已处于终止状态: {current_status}',
                'status': current_status,
            })
        session['cancel_requested'] = True
        session['message'] = '正在等待训练安全退出...'
    return jsonify({'success': True, 'message': '已提交取消请求'})


@bp.route('/result/<session_id>', methods=['GET'])
def get_training_result(session_id):
    with training_sessions_lock:
        _cleanup_training_sessions_locked()
        if session_id not in training_sessions:
            return jsonify({'success': False, 'error': '会话不存在'}), 404
        session = copy.deepcopy(training_sessions[session_id])
    if session['status'] != 'completed':
        return jsonify({'success': False, 'error': f"训练尚未完成，当前状态: {session['status']}"})
    return jsonify({
        'success': True,
        'results': session.get('results', {}),
        'config': session.get('config', {}),
        'completed_time': session.get('completed_time'),
    })


@bp.route('/history', methods=['GET'])
def get_training_history():
    try:
        history = []
        if not MODELS_DIR.exists():
            return jsonify({'success': True, 'history': [], 'total': 0})

        search = (request.args.get('search') or '').strip().lower()
        try:
            limit = int(request.args.get('limit') or 0)
        except ValueError:
            limit = 0
        try:
            offset = max(0, int(request.args.get('offset') or 0))
        except ValueError:
            offset = 0

        for model_dir in MODELS_DIR.iterdir():
            if not model_dir.is_dir():
                continue
            config_file = model_dir / "config.json"
            if not config_file.exists():
                continue
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                item = {
                    'model_id': cfg.get('model_id', model_dir.name),
                    'model_type': cfg.get('model_type'),
                    'task_type': cfg.get('task_type'),
                    'label_type': cfg.get('label_type'),
                    'predict_step': cfg.get('predict_step'),
                    'loss_function': cfg.get('loss_function'),
                    'metrics': cfg.get('metrics', {}),
                    'created_at': cfg.get('created_at'),
                    'data_info': cfg.get('data_info', {}),
                }
                if search:
                    haystack = ' '.join([
                        str(item.get('model_id') or ''),
                        str(item.get('model_type') or ''),
                        str(item.get('label_type') or ''),
                        str(item.get('loss_function') or ''),
                    ]).lower()
                    if search not in haystack:
                        continue
                history.append(item)
            except Exception:
                continue

        history.sort(key=lambda x: x.get('created_at') or '', reverse=True)
        total = len(history)
        if offset:
            history = history[offset:]
        if limit and limit > 0:
            history = history[:limit]
        return jsonify({'success': True, 'history': history, 'total': total})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/predict/<model_id>', methods=['POST'])
def predict(model_id):
    try:
        try:
            model_dir = _resolve_model_dir(model_id)
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        if not model_dir.exists() or not model_dir.is_dir():
            return jsonify({'success': False, 'error': '模型不存在'})

        with open(model_dir / "config.json", 'r', encoding='utf-8') as f:
            cfg = json.load(f)

        model_storage = cfg.get('model_storage', 'pickle')
        model_file = cfg.get('model_file', 'model.pkl')
        scaler_file = cfg.get('scaler_file', 'scaler.pkl')
        model_path = (model_dir / model_file).resolve()
        if not _is_path_under(model_dir, model_path):
            return jsonify({'success': False, 'error': '非法模型文件路径'})

        model = None
        scaler = None
        if model_storage == 'lightgbm_booster':
            import lightgbm as lgb_mod
            model = lgb_mod.Booster(model_file=str(model_path))
        elif model_storage == 'xgboost_booster':
            import xgboost as xgb_mod
            model = xgb_mod.Booster()
            model.load_model(str(model_path))
        elif model_storage == 'tsfm_hf':
            if not TSFM_AVAILABLE:
                return jsonify({'success': False, 'error': 'TSFM依赖未安装，无法加载模型'}), 400
            hf_dir = (model_dir / model_file).resolve()
            if not _is_path_under(model_dir, hf_dir) or not hf_dir.exists():
                return jsonify({'success': False, 'error': 'TSFM模型文件不存在'})
            model = TinyTimeMixerForPrediction.from_pretrained(str(hf_dir))
            preprocessor_file = cfg.get('preprocessor_file', 'preprocessor.pkl')
            preprocessor_path = (model_dir / preprocessor_file).resolve()
            if _is_path_under(model_dir, preprocessor_path) and preprocessor_path.exists():
                tsp = _safe_pickle_load(str(preprocessor_path))
            else:
                return jsonify({'success': False, 'error': 'TSFM预处理器文件不存在'})
        else:
            model = _safe_pickle_load(str(model_path))
            scaler_path = (model_dir / scaler_file).resolve()
            if _is_path_under(model_dir, scaler_path) and scaler_path.exists():
                scaler = _safe_pickle_load(str(scaler_path))

        data = request.get_json() or {}
        file_path = data.get('file_path')
        if not file_path:
            return jsonify({'success': False, 'error': '缺少文件路径'})
        try:
            validated_file_path = _validate_training_data_file_path(file_path)
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)}), 400

        df = _load_feather_data(str(validated_file_path), data.get('start_date'), data.get('end_date'))
        factor_ids = cfg.get('factor_ids', [])
        feature_name_mapping = cfg.get('feature_name_mapping') or {}
        if factor_ids:
            from factor_miner.core.factor_engine import get_global_engine
            engine = get_global_engine()
            feature_df, _ = _build_features(df, factor_ids, engine)
        else:
            feature_cols = list(feature_name_mapping.keys()) or cfg.get('feature_names', [])
            available_cols = [c for c in feature_cols if c in df.columns]
            feature_df = df[available_cols].select_dtypes(include=[np.number]).copy()

        if feature_name_mapping:
            feature_df = feature_df.rename(columns={k: v for k, v in feature_name_mapping.items() if k in feature_df.columns})

        feature_df = feature_df.dropna()
        if len(feature_df) == 0:
            return jsonify({'success': False, 'error': '无有效特征数据'})

        feature_names = cfg.get('feature_names', [])
        if feature_names:
            missing_cols = [c for c in feature_names if c not in feature_df.columns]
            if missing_cols:
                return jsonify({'success': False, 'error': f'预测特征缺失: {missing_cols[:10]}'})
            feature_df = feature_df[feature_names]

        if scaler is not None:
            X = scaler.transform(feature_df)
        else:
            X = feature_df

        model_type = cfg.get('model_type')
        if model_type == 'tsfm':
            import torch

            timestamp_col_pred = None
            for col in df.columns:
                cl = col.lower()
                if cl in ('open_time', 'opentime', 'time', 'date',
                           'timestamp', 'datetime', 'close_time', 'closetime'):
                    timestamp_col_pred = col
                    break
            if timestamp_col_pred is None:
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index()
                    timestamp_col_pred = df.columns[0]
                else:
                    return jsonify({'success': False, 'error': '预测数据中找不到时间戳列'})

            pred_col_mapping = {}
            for col in df.columns:
                cl = col.lower()
                if cl == 'open':
                    pred_col_mapping[col] = 'Open'
                elif cl == 'high':
                    pred_col_mapping[col] = 'High'
                elif cl == 'low':
                    pred_col_mapping[col] = 'Low'
                elif cl == 'close':
                    pred_col_mapping[col] = 'Close'
                elif cl == 'volume':
                    pred_col_mapping[col] = 'Volume'

            tsfm_pred_df = df.rename(columns=pred_col_mapping)
            tsfm_pred_df = tsfm_pred_df.rename(columns={timestamp_col_pred: 'timestamp'})
            tsfm_pred_df['timestamp'] = pd.to_datetime(tsfm_pred_df['timestamp'], errors='coerce')
            tsfm_pred_df = tsfm_pred_df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

            numeric_pred_cols = tsfm_pred_df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_pred_cols:
                tsfm_pred_df[numeric_pred_cols] = (
                    tsfm_pred_df[numeric_pred_cols].replace([np.inf, -np.inf], np.nan)
                )

            pred_dataset = tsp.preprocess(tsfm_pred_df)

            from torch.utils.data import Dataset as TorchDataset, DataLoader
            tsfm_allowed_keys_pred = [
                'past_values',
                'future_values',
                'past_observed_mask',
                'future_observed_mask',
                'freq_token',
                'static_categorical_values',
                'metadata',
                'label',
                'label_ids',
            ]

            def _coerce_tsfm_pred_value_for_tensor(value):
                if value is None:
                    return np.nan
                if isinstance(value, pd.Timestamp):
                    return np.nan if pd.isna(value) else float(value.timestamp())
                if isinstance(value, np.datetime64):
                    if np.isnat(value):
                        return np.nan
                    return float(pd.Timestamp(value).timestamp())
                if isinstance(value, pd.Timedelta):
                    return float(value.total_seconds())
                if isinstance(value, np.timedelta64):
                    return float(pd.to_timedelta(value).total_seconds())
                if isinstance(value, pd.Series):
                    value = value.to_numpy()
                if isinstance(value, (list, tuple, np.ndarray)):
                    arr = np.asarray(value)
                    if np.issubdtype(arr.dtype, np.datetime64):
                        arr_ns = arr.astype('datetime64[ns]').astype('int64')
                        arr_sec = np.where(arr_ns == np.iinfo(np.int64).min, np.nan, arr_ns / 1e9)
                        return arr_sec.astype(np.float32)
                    if np.issubdtype(arr.dtype, np.timedelta64):
                        return (arr.astype('timedelta64[ns]').astype('int64') / 1e9).astype(np.float32)
                    if arr.dtype == object:
                        return np.array([_coerce_tsfm_pred_value_for_tensor(v) for v in arr], dtype=np.float32)
                    return arr
                if pd.isna(value):
                    return np.nan
                return value

            class _PredTorchDataset(TorchDataset):
                def __init__(self, hf_dataset):
                    if hasattr(hf_dataset, 'to_pandas'):
                        self.df = hf_dataset.to_pandas() if not isinstance(hf_dataset, pd.DataFrame) else hf_dataset
                    elif isinstance(hf_dataset, pd.DataFrame):
                        self.df = hf_dataset
                    else:
                        self.df = pd.DataFrame(hf_dataset)
                    self._columns = [c for c in tsfm_allowed_keys_pred if c in self.df.columns]
                    dropped_cols = [c for c in self.df.columns if c not in self._columns]
                    if 'past_values' not in self._columns:
                        raise ValueError(
                            "TSFM预测预处理结果缺少 past_values。"
                            f"当前列: {list(self.df.columns)}"
                        )
                    if dropped_cols:
                        logger.warning("TSFM预测数据集中忽略非模型输入列: %s", dropped_cols)

                def __len__(self):
                    return len(self.df)

                def __getitem__(self, idx):
                    row = self.df.iloc[idx]
                    return {
                        col: torch.tensor(_coerce_tsfm_pred_value_for_tensor(row[col]), dtype=torch.float32)
                        for col in self._columns
                    }

            pred_dataset = _PredTorchDataset(pred_dataset)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)
            model.eval()

            pred_loader = DataLoader(pred_dataset, batch_size=32, shuffle=False)

            all_preds = []
            with torch.no_grad():
                for batch in pred_loader:
                    if isinstance(batch, dict):
                        input_ids = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}
                        outputs = model(**input_ids)
                    else:
                        batch = [b.to(device) if hasattr(b, 'to') else b for b in batch]
                        outputs = model(*batch)
                    if hasattr(outputs, 'prediction_outputs'):
                        pred_vals = outputs.prediction_outputs
                    elif hasattr(outputs, 'logits'):
                        pred_vals = outputs.logits
                    elif isinstance(outputs, tuple):
                        pred_vals = outputs[0]
                    else:
                        pred_vals = outputs
                    if hasattr(pred_vals, 'cpu'):
                        pred_vals = pred_vals.cpu().numpy()
                    all_preds.append(np.asarray(pred_vals))

            if all_preds:
                all_preds = np.concatenate(all_preds, axis=0)
            else:
                return jsonify({'success': False, 'error': 'TSFM预测结果为空'})

            if all_preds.ndim == 3:
                close_preds = all_preds[:, :, 0]
                predictions = close_preds.mean(axis=1)
            elif all_preds.ndim == 2:
                predictions = all_preds.mean(axis=1) if all_preds.shape[1] > 1 else all_preds.flatten()
            else:
                predictions = all_preds.flatten()

            context_length = cfg.get('model_params', {}).get('context_length', 512)
            forecast_length = cfg.get('model_params', {}).get('forecast_length', 96)
            n_samples = len(predictions)
            pred_timestamps = tsfm_pred_df['timestamp'].iloc[
                context_length:context_length + n_samples
            ] if len(tsfm_pred_df) > context_length else tsfm_pred_df['timestamp'].iloc[:n_samples]

            result_df = pd.DataFrame({
                'prediction': predictions[:len(pred_timestamps)],
            }, index=pred_timestamps.iloc[:len(predictions)])

            return jsonify({
                'success': True,
                'predictions': result_df.to_dict(orient='list'),
                'index': [str(t) for t in result_df.index],
                'model_id': model_id,
            })
        elif model_type == 'lightgbm':
            predictions = model.predict(X, num_iteration=model.best_iteration)
        elif model_type == 'xgboost':
            import xgboost as xgb_mod
            dmatrix = xgb_mod.DMatrix(X, feature_names=feature_names or list(feature_df.columns))
            predictions = model.predict(dmatrix)
        else:
            predictions = model.predict(X)

        result_df = pd.DataFrame({
            'prediction': predictions,
        }, index=feature_df.index)

        return jsonify({
            'success': True,
            'predictions': result_df.to_dict(orient='list'),
            'index': [str(t) for t in result_df.index],
            'model_id': model_id,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/delete/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    try:
        import shutil
        try:
            model_dir = _resolve_model_dir(model_id)
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        if not model_dir.exists() or not model_dir.is_dir():
            return jsonify({'success': False, 'error': '模型不存在'})
        shutil.rmtree(model_dir)
        return jsonify({'success': True, 'message': f'模型 {model_id} 已删除'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/detail/<model_id>', methods=['GET'])
def get_model_detail(model_id):
    try:
        try:
            model_dir = _resolve_model_dir(model_id)
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        if not model_dir.exists() or not model_dir.is_dir():
            return jsonify({'success': False, 'error': '模型不存在'})

        config_file = model_dir / "config.json"
        if not config_file.exists():
            return jsonify({'success': False, 'error': '配置文件不存在'})

        with open(config_file, 'r', encoding='utf-8') as f:
            cfg = json.load(f)

        return jsonify({
            'success': True,
            'detail': {
                'model_id': cfg.get('model_id'),
                'model_type': cfg.get('model_type'),
                'task_type': cfg.get('task_type'),
                'label_type': cfg.get('label_type'),
                'predict_step': cfg.get('predict_step'),
                'loss_function': cfg.get('loss_function'),
                'loss_params': cfg.get('loss_params', {}),
                'train_ratio': cfg.get('train_ratio'),
                'test_ratio': cfg.get('test_ratio'),
                'val_ratio': cfg.get('val_ratio'),
                'feature_names': cfg.get('feature_names', []),
                'label_name': cfg.get('label_name'),
                'model_params': cfg.get('model_params', {}),
                'factor_ids': cfg.get('factor_ids', []),
                'failed_factors': cfg.get('failed_factors', []),
                'data_file': cfg.get('data_file'),
                'start_date': cfg.get('start_date'),
                'end_date': cfg.get('end_date'),
                'created_at': cfg.get('created_at'),
                'data_info': cfg.get('data_info', {}),
                'metrics': cfg.get('metrics', {}),
                'loss_function_note': cfg.get('loss_function_note', ''),
                'model_storage': cfg.get('model_storage'),
            },
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@bp.route('/training-curve/<model_id>', methods=['GET'])
def get_training_curve(model_id):
    try:
        try:
            model_dir = _resolve_model_dir(model_id)
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        if not model_dir.exists() or not model_dir.is_dir():
            return jsonify({'success': False, 'error': '模型不存在'})

        config_file = model_dir / "config.json"
        if not config_file.exists():
            return jsonify({'success': False, 'error': '配置文件不存在'})

        with open(config_file, 'r', encoding='utf-8') as f:
            cfg = json.load(f)

        evals_result = cfg.get('evals_result')
        if not evals_result:
            return jsonify({'success': True, 'curve': None, 'message': '该模型无训练曲线数据'})

        return jsonify({
            'success': True,
            'curve': evals_result,
            'model_type': cfg.get('model_type'),
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
