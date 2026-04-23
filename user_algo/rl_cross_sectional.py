import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "RET",
    "VOL",
    "V_CHG",
    "PV",
    "TREND",
    "HL_RANGE",
    "CLOSE_POS",
    "MA_DEV",
    "VOLATILITY",
    "MOMENTUM",
]


def _features(data: pd.DataFrame) -> dict[str, pd.Series]:
    c = data["close"]
    h = data["high"]
    l = data["low"]
    v = data["volume"]
    ret = np.log(c / (c.shift(1) + 1e-9)).fillna(0.0)
    log_vol = np.log1p(v).fillna(0.0)
    v_chg = ((v - v.shift(1)) / (v.shift(1) + 1.0)).fillna(0.0)
    ma5 = c.rolling(5, min_periods=1).mean()
    trend = ((c - ma5) / (ma5 + 1e-6)).fillna(0.0)
    hl_range = ((h - l) / (c + 1e-6)).fillna(0.0)
    close_pos = ((c - l) / (h - l + 1e-6)).fillna(0.0)
    ma_dev = ((c - ma5) / (ma5 + 1e-6)).fillna(0.0)
    volatility = np.sqrt((ret ** 2).rolling(10, min_periods=1).mean() + 1e-9).fillna(0.0)
    momentum = ret.rolling(5, min_periods=1).sum().fillna(0.0)
    return {
        "RET": ret,
        "VOL": log_vol,
        "V_CHG": v_chg,
        "PV": (ret * log_vol).fillna(0.0),
        "TREND": trend,
        "HL_RANGE": hl_range,
        "CLOSE_POS": close_pos,
        "MA_DEV": ma_dev,
        "VOLATILITY": volatility,
        "MOMENTUM": momentum,
    }


def _rolling_rank(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(window=w, min_periods=1).rank(pct=True)


def _apply_op(op: str, args: list[pd.Series]) -> pd.Series | None:
    if op == "ADD" and len(args) >= 2:
        return args[-2] + args[-1]
    if op == "SUB" and len(args) >= 2:
        return args[-2] - args[-1]
    if op == "MUL" and len(args) >= 2:
        return args[-2] * args[-1]
    if op == "DIV" and len(args) >= 2:
        return args[-2] / (args[-1] + 1e-6)
    if op == "MAX" and len(args) >= 2:
        return np.maximum(args[-2], args[-1])
    if op == "MIN" and len(args) >= 2:
        return np.minimum(args[-2], args[-1])
    if op == "NEG" and len(args) >= 1:
        return -args[-1]
    if op == "ABS" and len(args) >= 1:
        return args[-1].abs()
    if op == "SIGN" and len(args) >= 1:
        return np.sign(args[-1])
    if op == "SQRT" and len(args) >= 1:
        return np.sqrt(np.abs(args[-1]) + 1e-8)
    if op == "LOG" and len(args) >= 1:
        return np.log(np.abs(args[-1]) + 1e-8)
    if op == "DELAY1" and len(args) >= 1:
        return args[-1].shift(1)
    if op == "DELAY3" and len(args) >= 1:
        return args[-1].shift(3)
    if op == "MA5" and len(args) >= 1:
        return args[-1].rolling(5, min_periods=1).mean()
    if op == "MA10" and len(args) >= 1:
        return args[-1].rolling(10, min_periods=1).mean()
    if op == "MA20" and len(args) >= 1:
        return args[-1].rolling(20, min_periods=1).mean()
    if op == "STD5" and len(args) >= 1:
        return args[-1].rolling(5, min_periods=1).std()
    if op == "STD10" and len(args) >= 1:
        return args[-1].rolling(10, min_periods=1).std()
    if op == "RANK5" and len(args) >= 1:
        return _rolling_rank(args[-1], 5)
    if op == "MOM5" and len(args) >= 1:
        return args[-1] / (args[-1].shift(5) + 1e-6) - 1
    if op == "MOM10" and len(args) >= 1:
        return args[-1] / (args[-1].shift(10) + 1e-6) - 1
    if op == "DECAY" and len(args) >= 1:
        x = args[-1]
        return x + 0.8 * x.shift(1) + 0.6 * x.shift(2)
    if op == "CS_RANK" and len(args) >= 1:
        return args[-1].rank(pct=True)
    if op == "CS_ZSCORE" and len(args) >= 1:
        x = args[-1]
        return (x - x.mean()) / (x.std() + 1e-6)
    if op == "CS_MAD" and len(args) >= 1:
        x = args[-1]
        med = x.median()
        mad = (x - med).abs().median() + 1e-6
        return ((x - med) / mad).clip(-5.0, 5.0)
    if op == "JUMP" and len(args) >= 1:
        x = args[-1]
        z = (x - x.mean()) / (x.std() + 1e-6)
        return np.maximum(z - 3.0, 0.0)
    if op == "GATE" and len(args) >= 3:
        c, x, y = args[-3], args[-2], args[-1]
        return pd.Series(np.where(c > 0, x, y), index=x.index)
    return None


def _eval_postfix(expression: str, feats: dict[str, pd.Series], index: pd.Index) -> pd.Series:
    tokens = [t for t in str(expression).split(" ") if t]
    stack: list[pd.Series] = []
    for token in tokens:
        if token in feats:
            stack.append(feats[token])
            continue
        out = _apply_op(token, stack)
        if out is None:
            continue
        arity = 3 if token == "GATE" else (2 if token in {"ADD", "SUB", "MUL", "DIV", "MAX", "MIN"} else 1)
        for _ in range(min(arity, len(stack))):
            stack.pop()
        stack.append(pd.Series(out, index=index))
    if len(stack) != 1:
        return pd.Series(0.0, index=index)
    return stack[0]


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
    feats = _features(data)
    result = _eval_postfix(str(expression), feats, data.index)
    return pd.Series(result, index=data.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
