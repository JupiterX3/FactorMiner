"""
共享算子库 — qlib Alpha158 / Alpha360 / WorldQuant Alpha101 因子复用。

设计原则：
1. 所有算子输入为 pandas.Series（少数为 DataFrame），输出为 pandas.Series。
2. 所有时序算子都只使用"过去 n 根 K 线"的信息，**绝不使用未来数据**。
3. 分母加 1e-12 避免 /0；log(0) 返回 NaN。
4. 算子命名对齐 qlib (`Mean/Std/Max/...`) 与 WorldQuant (`ts_mean/ts_std/...`)
   两套口径，因此有少量别名（如 `delay = ref`、`rank = ts_rank_pct_last`）。

该文件**不是因子**——以下划线开头，FactorEngine 不会把它当因子加载。
"""

import numpy as np
import pandas as pd

EPS = 1e-12


# -----------------------------------------------------------------------------
# 基础时序算子
# -----------------------------------------------------------------------------
def ts_mean(s: pd.Series, n: int) -> pd.Series:
    """滚动均值。min_periods 取 max(2, n//2) 避免窗口早期出 0 方差。"""
    return s.rolling(n, min_periods=max(2, n // 2)).mean()


def ts_std(s: pd.Series, n: int) -> pd.Series:
    """滚动标准差。"""
    return s.rolling(n, min_periods=max(2, n // 2)).std()


def ts_sum(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(2, n // 2)).sum()


def ts_max(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(2, n // 2)).max()


def ts_min(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(2, n // 2)).min()


def ts_quantile(s: pd.Series, n: int, q: float) -> pd.Series:
    return s.rolling(n, min_periods=max(2, n // 2)).quantile(q)


def ts_rank(s: pd.Series, n: int) -> pd.Series:
    """滚动窗口内最后一点的 pct rank。对应 qlib Rank / WQ ts_rank。"""
    return s.rolling(n, min_periods=max(2, n // 2)).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )


def ts_argmax(s: pd.Series, n: int) -> pd.Series:
    """滚动 argmax（窗口内最大值的位置，1..n）。qlib Idxmax，WQ ts_argmax。"""
    return s.rolling(n, min_periods=max(2, n // 2)).apply(
        lambda x: float(np.argmax(x)) + 1.0, raw=True
    )


def ts_argmin(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(2, n // 2)).apply(
        lambda x: float(np.argmin(x)) + 1.0, raw=True
    )


def ts_corr(a: pd.Series, b: pd.Series, n: int) -> pd.Series:
    return a.rolling(n, min_periods=max(2, n // 2)).corr(b)


def ts_cov(a: pd.Series, b: pd.Series, n: int) -> pd.Series:
    return a.rolling(n, min_periods=max(2, n // 2)).cov(b)


# -----------------------------------------------------------------------------
# 点算子
# -----------------------------------------------------------------------------
def delta(s: pd.Series, n: int) -> pd.Series:
    return s.diff(n)


def ref(s: pd.Series, n: int) -> pd.Series:
    """qlib Ref(x, n) = 向前取 n 期的值 = s.shift(n)。"""
    return s.shift(n)


# WQ 中 `delay` 与 `ref` 同义
delay = ref


def sign(s: pd.Series) -> pd.Series:
    return np.sign(s)


def log(s: pd.Series) -> pd.Series:
    """自然对数；0/负数 → NaN，避免 -inf 污染。"""
    return np.log(s.where(s > 0))


def signed_power(s, p: float):
    """
    符号幂：sign(x) * |x|^p，常见于 WQ 公式。
    同时接受 pandas.Series 和 numpy.ndarray。
    """
    if isinstance(s, pd.Series):
        return np.sign(s) * (s.abs() ** p)
    arr = np.asarray(s, dtype=float)
    return np.sign(arr) * (np.abs(arr) ** p)


def abs_(s):
    """避开内置 abs 被局部变量遮蔽的情况；同时接受 Series 与 ndarray。"""
    if isinstance(s, pd.Series):
        return s.abs()
    return np.abs(np.asarray(s, dtype=float))


def scale(s: pd.Series, k: float = 1.0) -> pd.Series:
    """按 L1 范数归一到 k。常用于 WQ alpha024/025 等。"""
    denom = s.abs().sum()
    if denom == 0 or not np.isfinite(denom):
        return s * 0.0
    return k * s / denom


def decay_linear(s: pd.Series, n: int) -> pd.Series:
    """线性加权衰减：窗口内权重 1..n（越近越大），归一化后求加权和。"""
    w = np.arange(1, n + 1, dtype=float)
    w = w / w.sum()

    def _apply(x):
        if len(x) < n:
            return np.nan
        return float(np.dot(x, w))

    return s.rolling(n, min_periods=n).apply(_apply, raw=True)


# -----------------------------------------------------------------------------
# qlib BETA / RSQR / RESI — 对时间的滚动线性回归
# -----------------------------------------------------------------------------
def _rolling_linreg_stats(y: pd.Series, n: int) -> pd.DataFrame:
    """
    对 y 关于时间索引 t = 0..n-1 做滚动线性回归，返回每个窗口的 (beta, rsqr, resi_last)。
    resi_last 为窗口最后一点的回归残差。
    """
    arr = y.to_numpy(dtype=float)
    T = len(arr)
    beta = np.full(T, np.nan)
    rsqr = np.full(T, np.nan)
    resi = np.full(T, np.nan)
    if n < 2 or T < n:
        return pd.DataFrame({'beta': beta, 'rsqr': rsqr, 'resi': resi}, index=y.index)

    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    x_dev = x - x_mean
    x_var = (x_dev * x_dev).sum()

    for end in range(n - 1, T):
        window = arr[end - n + 1 : end + 1]
        if np.any(~np.isfinite(window)):
            continue
        y_mean = window.mean()
        y_dev = window - y_mean
        cov = (x_dev * y_dev).sum()
        if x_var <= 0:
            continue
        b = cov / x_var
        y_var = (y_dev * y_dev).sum()
        r2 = (cov * cov) / (x_var * y_var) if y_var > 0 else np.nan
        y_hat_last = y_mean + b * (x[-1] - x_mean)
        beta[end] = b
        rsqr[end] = r2
        resi[end] = window[-1] - y_hat_last

    return pd.DataFrame({'beta': beta, 'rsqr': rsqr, 'resi': resi}, index=y.index)


def rolling_beta(y: pd.Series, n: int) -> pd.Series:
    """qlib BETA(y, n) — 对时间的滚动回归斜率。"""
    return _rolling_linreg_stats(y, n)['beta']


def rolling_rsqr(y: pd.Series, n: int) -> pd.Series:
    """qlib RSQR(y, n) — 滚动回归 R²。"""
    return _rolling_linreg_stats(y, n)['rsqr']


def rolling_resi(y: pd.Series, n: int) -> pd.Series:
    """qlib RESI(y, n) — 窗口最后一点的回归残差。"""
    return _rolling_linreg_stats(y, n)['resi']


# -----------------------------------------------------------------------------
# 派生量
# -----------------------------------------------------------------------------
def adv(volume: pd.Series, n: int = 20) -> pd.Series:
    """WQ101 中的 advN — N 日平均成交量。"""
    return ts_mean(volume, n)


def vwap_proxy(data: pd.DataFrame) -> pd.Series:
    """
    WQ101 中的 vwap — 单币种 K 线数据没有逐笔成交价，用典型价 (H+L+C)/3 近似。
    与真实 VWAP 的差异在加密 1h/1d 粒度下通常 < 0.5%。
    """
    return (data['high'] + data['low'] + data['close']) / 3.0


def returns(close: pd.Series) -> pd.Series:
    """WQ101 中的 returns — 1 期收益率。"""
    return close.pct_change()


# -----------------------------------------------------------------------------
# 便捷工具（给注册脚本使用）
# -----------------------------------------------------------------------------
def safe_div(a, b):
    """a / (b + EPS)，避免除 0。"""
    return a / (b + EPS)


def where(cond, a, b) -> pd.Series:
    """
    np.where 的 Series 安全版本。
    - 自动推断索引（优先来自 cond，再到 a，再到 b）
    - 返回 pd.Series 以便后续 ts_* 算子可直接 rolling
    """
    for candidate in (cond, a, b):
        if isinstance(candidate, pd.Series):
            idx = candidate.index
            break
    else:
        return pd.Series(np.where(cond, a, b))
    return pd.Series(np.where(cond, a, b), index=idx)


__all__ = [
    'EPS',
    # 时序
    'ts_mean', 'ts_std', 'ts_sum', 'ts_max', 'ts_min', 'ts_quantile',
    'ts_rank', 'ts_argmax', 'ts_argmin', 'ts_corr', 'ts_cov',
    # 点算子
    'delta', 'ref', 'delay', 'sign', 'log', 'signed_power', 'abs_',
    'scale', 'decay_linear',
    # 回归
    'rolling_beta', 'rolling_rsqr', 'rolling_resi',
    # 派生量
    'adv', 'vwap_proxy', 'returns',
    # 工具
    'safe_div', 'where',
]
