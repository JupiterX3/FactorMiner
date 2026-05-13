import re
import numpy as np
import pandas as pd


ARITH_OPS = {"add", "sub", "mul", "div", "max", "min"}
TS_OPS = {
    "delay",
    "ts_mean",
    "ts_std",
    "ts_max",
    "ts_min",
    "ts_rank",
    "ts_zscore",
    "ts_decay",
    "ts_momentum",
    "ts_skewness",
    "ts_kurtosis",
}
BINARY_TS_OPS = {"ts_corr", "ts_cov"}
CS_OPS = {"cs_rank", "cs_zscore", "cs_mad_norm"}


def _split_top_level(content: str, sep: str) -> list[str]:
    parts = []
    depth = 0
    start = 0
    i = 0
    while i < len(content):
        ch = content[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and content.startswith(sep, i):
            parts.append(content[start:i].strip())
            i += len(sep)
            start = i
            continue
        i += 1
    parts.append(content[start:].strip())
    return parts


def _extract_wrapped(expr: str):
    expr = expr.strip()
    if not expr.endswith(")"):
        return None, None
    idx = expr.find("(")
    if idx <= 0:
        return None, None
    prefix = expr[:idx].strip()
    inside = expr[idx + 1:-1].strip()
    return prefix, inside


def _rolling_rank(series: pd.Series, w: int) -> pd.Series:
    return series.rolling(window=w, min_periods=1).rank(pct=True)


def _compute_features(data: pd.DataFrame) -> dict[str, pd.Series]:
    c = data["close"]
    o = data["open"]
    h = data["high"]
    l = data["low"]
    v = data["volume"]

    features = {
        "returns": c.pct_change(),
        "log_ret": np.log(c / (c.shift(1) + 1e-8)),
        "volatility": c.pct_change().rolling(20, min_periods=1).std(),
        "volume_ratio": v / (v.rolling(20, min_periods=1).mean() + 1e-8),
        "price_position": (c - l.rolling(20, min_periods=1).min()) / (
            h.rolling(20, min_periods=1).max() - l.rolling(20, min_periods=1).min() + 1e-8
        ),
        "momentum": c / (c.shift(10) + 1e-8) - 1,
        "high_low_range": (h - l) / (c + 1e-8),
        "close_open_diff": (c - o) / (o + 1e-8),
        "volume_price_corr": c.pct_change().rolling(20, min_periods=2).corr(v.pct_change()),
        "ma_deviation": (c - c.rolling(20, min_periods=1).mean()) / (c.rolling(20, min_periods=1).std() + 1e-8),
    }

    if "basis" in data.columns:
        features["basis"] = data["basis"].fillna(0)
    else:
        features["basis"] = pd.Series(0.0, index=data.index)

    if "open_interest" in data.columns:
        features["oi_change"] = data["open_interest"].pct_change(12).fillna(0)
    else:
        features["oi_change"] = pd.Series(0.0, index=data.index)

    if "lsr_global_account" in data.columns and "lsr_top_account" in data.columns:
        features["lsr_spread"] = (data["lsr_global_account"] - data["lsr_top_account"]).fillna(0)
    else:
        features["lsr_spread"] = pd.Series(0.0, index=data.index)

    if "taker_buy_base" in data.columns and "volume" in data.columns:
        vol = data["volume"].replace(0, np.nan)
        features["taker_imbalance"] = (data["taker_buy_base"] / vol - 0.5).fillna(0)
    else:
        features["taker_imbalance"] = pd.Series(0.0, index=data.index)

    if "funding_rate" in data.columns:
        features["funding_rate"] = data["funding_rate"].fillna(0)
    else:
        features["funding_rate"] = pd.Series(0.0, index=data.index)

    for k in list(features.keys()):
        features[k] = features[k].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features


def _evaluate(expr: str, features: dict[str, pd.Series], index: pd.Index):
    expr = expr.strip()
    if not expr:
        return pd.Series(0.0, index=index)

    if expr in features:
        return features[expr]
    try:
        return pd.Series(float(expr), index=index)
    except Exception:
        pass

    if expr.startswith("(") and expr.endswith(")"):
        inner = expr[1:-1].strip()
        for op in ARITH_OPS:
            sep = f" {op} "
            parts = _split_top_level(inner, sep)
            if len(parts) == 2:
                l = _evaluate(parts[0], features, index)
                r = _evaluate(parts[1], features, index)
                if op == "add":
                    return l + r
                if op == "sub":
                    return l - r
                if op == "mul":
                    return l * r
                if op == "div":
                    return l / (r + 1e-8)
                if op == "max":
                    return np.maximum(l, r)
                if op == "min":
                    return np.minimum(l, r)
        return _evaluate(inner, features, index)

    if expr.startswith("gate(") and expr.endswith(")"):
        args = _split_top_level(expr[5:-1], ",")
        if len(args) == 3:
            c = _evaluate(args[0], features, index)
            y = _evaluate(args[1], features, index)
            z = _evaluate(args[2], features, index)
            return pd.Series(np.where(c > 0, y, z), index=index)

    wm_direct = re.match(r"^([a-zA-Z_]+)\(w=(\d+)\)\((.*)\)$", expr)
    if wm_direct:
        op = wm_direct.group(1)
        window = int(wm_direct.group(2))
        arg_expr = wm_direct.group(3).strip()

        if op in BINARY_TS_OPS:
            args = _split_top_level(arg_expr, ",")
            if len(args) == 2:
                x = _evaluate(args[0].strip(), features, index)
                y = _evaluate(args[1].strip(), features, index)
                if op == "ts_corr":
                    return x.rolling(window=window, min_periods=2).corr(y)
                if op == "ts_cov":
                    return x.rolling(window=window, min_periods=2).cov(y)

        x = _evaluate(arg_expr, features, index)
        if op == "delay":
            return x.shift(window)
        if op == "ts_mean":
            return x.rolling(window=window, min_periods=1).mean()
        if op == "ts_std":
            return x.rolling(window=window, min_periods=1).std()
        if op == "ts_max":
            return x.rolling(window=window, min_periods=1).max()
        if op == "ts_min":
            return x.rolling(window=window, min_periods=1).min()
        if op == "ts_rank":
            return _rolling_rank(x, window)
        if op == "ts_zscore":
            m = x.rolling(window=window, min_periods=1).mean()
            s = x.rolling(window=window, min_periods=1).std()
            return (x - m) / (s + 1e-8)
        if op == "ts_decay":
            w = np.array([0.9 ** i for i in range(window)])[::-1]
            w = w / w.sum()
            return x.rolling(window=window, min_periods=1).apply(lambda v: np.dot(v, w[:len(v)]), raw=True)
        if op == "ts_momentum":
            return x / (x.shift(window) + 1e-8) - 1
        if op == "ts_skewness":
            return x.rolling(window=window, min_periods=max(3, window // 2)).skew()
        if op == "ts_kurtosis":
            return x.rolling(window=window, min_periods=max(4, window // 2)).kurt()

    prefix, inside = _extract_wrapped(expr)
    if prefix is None:
        return pd.Series(0.0, index=index)

    wm = re.match(r"^([a-zA-Z_]+)(?:\(w=(\d+)\))?$", prefix)
    if not wm:
        return pd.Series(0.0, index=index)
    op = wm.group(1)
    window = int(wm.group(2)) if wm.group(2) else 20

    args = _split_top_level(inside, ",")

    if op in {"abs", "neg", "sqrt", "log", "sign"} and len(args) == 1:
        x = _evaluate(args[0], features, index)
        if op == "abs":
            return x.abs()
        if op == "neg":
            return -x
        if op == "sqrt":
            return np.sqrt(np.abs(x))
        if op == "log":
            return np.log(np.abs(x) + 1e-8)
        if op == "sign":
            return np.sign(x)

    if op in TS_OPS and len(args) == 1:
        x = _evaluate(args[0], features, index)
        if op == "delay":
            return x.shift(window)
        if op == "ts_mean":
            return x.rolling(window=window, min_periods=1).mean()
        if op == "ts_std":
            return x.rolling(window=window, min_periods=1).std()
        if op == "ts_max":
            return x.rolling(window=window, min_periods=1).max()
        if op == "ts_min":
            return x.rolling(window=window, min_periods=1).min()
        if op == "ts_rank":
            return _rolling_rank(x, window)
        if op == "ts_zscore":
            m = x.rolling(window=window, min_periods=1).mean()
            s = x.rolling(window=window, min_periods=1).std()
            return (x - m) / (s + 1e-8)
        if op == "ts_decay":
            w = np.array([0.9 ** i for i in range(window)])[::-1]
            w = w / w.sum()
            return x.rolling(window=window, min_periods=1).apply(lambda v: np.dot(v, w[:len(v)]), raw=True)
        if op == "ts_momentum":
            return x / (x.shift(window) + 1e-8) - 1
        if op == "ts_skewness":
            return x.rolling(window=window, min_periods=max(3, window // 2)).skew()
        if op == "ts_kurtosis":
            return x.rolling(window=window, min_periods=max(4, window // 2)).kurt()

    if op in BINARY_TS_OPS and len(args) == 2:
        x = _evaluate(args[0], features, index)
        y = _evaluate(args[1], features, index)
        if op == "ts_corr":
            return x.rolling(window=window, min_periods=2).corr(y)
        if op == "ts_cov":
            return x.rolling(window=window, min_periods=2).cov(y)

    if op in CS_OPS and len(args) == 1:
        x = _evaluate(args[0], features, index)
        if op == "cs_rank":
            return x.rank(pct=True)
        if op == "cs_zscore":
            s = x.std()
            return (x - x.mean()) / (s + 1e-8)
        if op == "cs_mad_norm":
            med = x.median()
            mad = (x - med).abs().median()
            return ((x - med) / (mad + 1e-8)).clip(-5, 5)

    return pd.Series(0.0, index=index)


def calculate_single_factor(
    data: pd.DataFrame,
    factor_name: str,
    factor_id: str = "",
    computation_data: dict | None = None,
    **kwargs,
) -> pd.Series:
    comp_data = computation_data or {}
    perf = comp_data.get("performance_metrics") or {}
    expression = perf.get("expression") or comp_data.get("expression")
    if not expression:
        return pd.Series(0.0, index=data.index)

    features = _compute_features(data)
    result = _evaluate(str(expression), features, data.index)
    return pd.Series(result, index=data.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
