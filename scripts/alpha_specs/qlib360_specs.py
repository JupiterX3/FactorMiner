"""
qlib Alpha360 精简版因子规格 — 30 个因子。

原版 Alpha360 = {open, high, low, close, vwap, volume} × lag(0..59) / close — 共 360 个。
在我们的场景下 close_lag1 与 close_lag2 相关系数 > 0.99，保留全量价值低。

精简策略（Fibonacci lag + 关键比率 + 辅助因子）：
- close × 9 lag (1,2,3,5,8,13,21,34,55)
- open/high/low/vwap × 3 lag (1,5,21)
- volume × 3 lag (1,5,21)
- 3 个辅助因子（比率型）
合计 9 + 3*4 + 3 + 3 = 27 + 3 = 30
"""
from typing import List, Dict


BASE_IMPORTS = [
    "import pandas as pd",
    "import numpy as np",
    "from _alpha_ops import *  # noqa: F401,F403",
]


CLOSE_LAGS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
OTHER_LAGS = [1, 5, 21]


def _lag_specs() -> List[Dict]:
    specs: List[Dict] = []

    for lag in CLOSE_LAGS:
        specs.append({
            "factor_id": f"alpha360_close_lag_{lag}",
            "name": f"qlib360-CLOSE_lag{lag}",
            "description": f"qlib Alpha360 — close.shift({lag}) / close",
            "category": "qlib360",
            "subcategory": "close",
            "is_window": False,
            "min_warmup_bars": lag + 1,
            "imports": BASE_IMPORTS,
            "body": (
                "def calculate(data, **kwargs):\n"
                f"    return ref(data['close'], {lag}) / (data['close'] + EPS)\n"
            ),
        })

    for field_name, field_expr in [
        ("open",  "data['open']"),
        ("high",  "data['high']"),
        ("low",   "data['low']"),
        ("vwap",  "vwap_proxy(data)"),
    ]:
        for lag in OTHER_LAGS:
            specs.append({
                "factor_id": f"alpha360_{field_name}_lag_{lag}",
                "name": f"qlib360-{field_name.upper()}_lag{lag}",
                "description": f"qlib Alpha360 — {field_name}.shift({lag}) / close",
                "category": "qlib360",
                "subcategory": field_name,
                "is_window": False,
                "min_warmup_bars": lag + 1,
                "imports": BASE_IMPORTS,
                "body": (
                    "def calculate(data, **kwargs):\n"
                    f"    return ref({field_expr}, {lag}) / (data['close'] + EPS)\n"
                ),
            })

    for lag in OTHER_LAGS:
        specs.append({
            "factor_id": f"alpha360_volume_lag_{lag}",
            "name": f"qlib360-VOLUME_lag{lag}",
            "description": f"qlib Alpha360 — volume.shift({lag}) / (volume+eps)",
            "category": "qlib360",
            "subcategory": "volume",
            "is_window": False,
            "min_warmup_bars": lag + 1,
            "imports": BASE_IMPORTS,
            "body": (
                "def calculate(data, **kwargs):\n"
                f"    return ref(data['volume'], {lag}) / (data['volume'] + EPS)\n"
            ),
        })

    return specs


def _aux_specs() -> List[Dict]:
    """5 个补齐因子（中期动量 / 中期量能 / 振幅 / 短长期动量比 / OHLC均价）。"""
    return [
        {
            "factor_id": "alpha360_close_ratio_1_20",
            "name": "qlib360-CLOSE_ratio_1_20",
            "description": "qlib Alpha360 辅助 — close.shift(1) / close.shift(20)，中期动量",
            "category": "qlib360",
            "subcategory": "ratio",
            "is_window": False,
            "min_warmup_bars": 21,
            "imports": BASE_IMPORTS,
            "body": (
                "def calculate(data, **kwargs):\n"
                "    return ref(data['close'], 1) / (ref(data['close'], 20) + EPS)\n"
            ),
        },
        {
            "factor_id": "alpha360_close_ratio_5_60",
            "name": "qlib360-CLOSE_ratio_5_60",
            "description": "qlib Alpha360 辅助 — close.shift(5) / close.shift(60)，长期动量",
            "category": "qlib360",
            "subcategory": "ratio",
            "is_window": False,
            "min_warmup_bars": 61,
            "imports": BASE_IMPORTS,
            "body": (
                "def calculate(data, **kwargs):\n"
                "    return ref(data['close'], 5) / (ref(data['close'], 60) + EPS)\n"
            ),
        },
        {
            "factor_id": "alpha360_volume_ratio_1_20",
            "name": "qlib360-VOLUME_ratio_1_20",
            "description": "qlib Alpha360 辅助 — volume.shift(1) / volume.shift(20)，中期量能",
            "category": "qlib360",
            "subcategory": "ratio",
            "is_window": False,
            "min_warmup_bars": 21,
            "imports": BASE_IMPORTS,
            "body": (
                "def calculate(data, **kwargs):\n"
                "    return ref(data['volume'], 1) / (ref(data['volume'], 20) + EPS)\n"
            ),
        },
        {
            "factor_id": "alpha360_hl_range_lag_5",
            "name": "qlib360-HL_range_lag_5",
            "description": "qlib Alpha360 辅助 — (high.shift(5) - low.shift(5)) / close，5 日前振幅",
            "category": "qlib360",
            "subcategory": "ratio",
            "is_window": False,
            "min_warmup_bars": 6,
            "imports": BASE_IMPORTS,
            "body": (
                "def calculate(data, **kwargs):\n"
                "    return (ref(data['high'], 5) - ref(data['low'], 5)) / (data['close'] + EPS)\n"
            ),
        },
        {
            "factor_id": "alpha360_ohlc_avg_lag_1",
            "name": "qlib360-OHLC_avg_lag_1",
            "description": "qlib Alpha360 辅助 — (O+H+L+C).shift(1)/4 / close，1 期前 OHLC 均价相对比",
            "category": "qlib360",
            "subcategory": "ratio",
            "is_window": False,
            "min_warmup_bars": 2,
            "imports": BASE_IMPORTS,
            "body": (
                "def calculate(data, **kwargs):\n"
                "    ohlc = (data['open'] + data['high'] + data['low'] + data['close']) / 4.0\n"
                "    return ref(ohlc, 1) / (data['close'] + EPS)\n"
            ),
        },
    ]


def build_specs() -> List[Dict]:
    specs: List[Dict] = []
    specs.extend(_lag_specs())   # 10 + 3*4 + 3 = 25
    specs.extend(_aux_specs())   # 5
    assert len(specs) == 30, f"Alpha360 精简规格数应为 30，当前 {len(specs)}"
    return specs


if __name__ == "__main__":
    specs = build_specs()
    print(f"qlib Alpha360（精简）规格数量：{len(specs)}")
    for s in specs[:5]:
        print(f"  {s['factor_id']} / {s['subcategory']} / warmup={s['min_warmup_bars']}")
