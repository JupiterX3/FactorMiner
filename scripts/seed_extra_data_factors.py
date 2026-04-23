"""
种子脚本：为 factorlib/derivatives/ 和 factorlib/funding/ 生成示例因子。

所有因子都是纯 function 类型，从 DataLoader.load_with_extras 得到的 DataFrame
中按列取值。实盘可通过同名接口获取相同字段从而保持一致。

运行：
    python scripts/seed_extra_data_factors.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Dict, List

ROOT = Path(__file__).parent.parent
FACTORLIB = ROOT / "factorlib"


def _write_factor(
    group: str,
    factor_id: str,
    name: str,
    description: str,
    subcategory: str,
    function_code: str,
    parameters: Dict | None = None,
    imports: List[str] | None = None,
) -> None:
    def_dir = FACTORLIB / group / "definitions"
    fn_dir = FACTORLIB / group / "functions"
    def_dir.mkdir(parents=True, exist_ok=True)
    fn_dir.mkdir(parents=True, exist_ok=True)

    # 写 .py 文件
    fn_path = fn_dir / f"{factor_id}.py"
    header = "\n".join(imports or ["import pandas as pd", "import numpy as np"])
    with open(fn_path, "w", encoding="utf-8") as f:
        f.write(header + "\n\n" + function_code.strip() + "\n")

    # 写 JSON
    def_path = def_dir / f"{factor_id}.json"
    payload = {
        "factor_id": factor_id,
        "name": name,
        "description": description,
        "category": group,
        "subcategory": subcategory,
        "computation_type": "function",
        "computation_data": {
            "function_file": f"{group}/functions/{factor_id}.py",
            "function_code": function_code.strip(),
            "entry_point": "calculate",
            "imports": imports or ["import pandas as pd", "import numpy as np"],
        },
        "parameters": parameters or {},
        "dependencies": [],
        "output_type": "series",
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "source_family": group,
            "requires_extras": [group],
        },
    }
    with open(def_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ==========================================================================
# derivatives/*
# ==========================================================================

DERIV_FACTORS = [
    {
        "factor_id": "oi_change_pct",
        "name": "持仓量变化率",
        "description": "open_interest 在 period 周期内的百分比变化，反映资金入场/出场强度。",
        "subcategory": "open_interest",
        "parameters": {"period": 12},
        "code": dedent(
            """
            def calculate(data, period=12, **kwargs):
                if 'open_interest' not in data.columns:
                    return pd.Series(index=data.index, dtype=float)
                return data['open_interest'].pct_change(period)
            """
        ),
    },
    {
        "factor_id": "oi_change_zscore",
        "name": "持仓变化 z-score",
        "description": "持仓量变化率相对 window 滚动均值/标准差的 z-score。",
        "subcategory": "open_interest",
        "parameters": {"period": 12, "window": 96},
        "code": dedent(
            """
            def calculate(data, period=12, window=96, **kwargs):
                if 'open_interest' not in data.columns:
                    return pd.Series(index=data.index, dtype=float)
                chg = data['open_interest'].pct_change(period)
                mu = chg.rolling(window).mean()
                sd = chg.rolling(window).std()
                return (chg - mu) / sd
            """
        ),
    },
    {
        "factor_id": "oi_to_volume_ratio",
        "name": "持仓 / 成交量比",
        "description": "open_interest 与 volume 的比值，衡量持仓相对活跃度。"
        "数值越大说明筹码沉淀越多、交易频率越低。",
        "subcategory": "open_interest",
        "parameters": {"window": 20},
        "code": dedent(
            """
            def calculate(data, window=20, **kwargs):
                if 'open_interest' not in data.columns:
                    return pd.Series(index=data.index, dtype=float)
                vol = data['volume'].rolling(window).mean().replace(0, np.nan)
                return data['open_interest'] / vol
            """
        ),
    },
    {
        "factor_id": "lsr_top_position_zscore",
        "name": "大户持仓比 z-score",
        "description": "topLongShortPositionRatio 的 window 滚动 z-score，"
        "捕捉大户多空极端配置。正向高值=大户显著偏多。",
        "subcategory": "long_short_ratio",
        "parameters": {"window": 96},
        "code": dedent(
            """
            def calculate(data, window=96, **kwargs):
                if 'lsr_top_position' not in data.columns:
                    return pd.Series(index=data.index, dtype=float)
                x = data['lsr_top_position']
                return (x - x.rolling(window).mean()) / x.rolling(window).std()
            """
        ),
    },
    {
        "factor_id": "lsr_retail_vs_top_spread",
        "name": "散户-大户多空比差",
        "description": "全局账户多空比（近似散户）与大户账户多空比的差值，"
        "正向较大时 = 散户更激进。常见反向指标。",
        "subcategory": "long_short_ratio",
        "parameters": {},
        "code": dedent(
            """
            def calculate(data, **kwargs):
                if 'lsr_global_account' not in data.columns \\
                        or 'lsr_top_account' not in data.columns:
                    return pd.Series(index=data.index, dtype=float)
                return data['lsr_global_account'] - data['lsr_top_account']
            """
        ),
    },
    {
        "factor_id": "taker_buy_ratio",
        "name": "主动买入成交占比",
        "description": "taker_buy_base 占总成交量的比重，反映买卖盘压差。"
        "来源于每根 K 线的原生字段。",
        "subcategory": "taker_volume",
        "parameters": {"window": 1},
        "code": dedent(
            """
            def calculate(data, window=1, **kwargs):
                if 'taker_buy_base' not in data.columns:
                    return pd.Series(index=data.index, dtype=float)
                vol = data['volume'].replace(0, np.nan)
                ratio = data['taker_buy_base'] / vol
                if window and window > 1:
                    ratio = ratio.rolling(window).mean()
                return ratio
            """
        ),
    },
    {
        "factor_id": "taker_buy_imbalance_zscore",
        "name": "主动买入失衡 z-score",
        "description": "(taker_buy_base / volume - 0.5) 相对自身 window 的 z-score，"
        "捕捉显著偏多 / 偏空盘面。",
        "subcategory": "taker_volume",
        "parameters": {"window": 48},
        "code": dedent(
            """
            def calculate(data, window=48, **kwargs):
                if 'taker_buy_base' not in data.columns:
                    return pd.Series(index=data.index, dtype=float)
                vol = data['volume'].replace(0, np.nan)
                imbalance = data['taker_buy_base'] / vol - 0.5
                mu = imbalance.rolling(window).mean()
                sd = imbalance.rolling(window).std()
                return (imbalance - mu) / sd
            """
        ),
    },
    {
        "factor_id": "basis_zscore",
        "name": "基差 z-score",
        "description": "basis = (close - index_close) / index_close，相对自身的 z-score。"
        "正向高=期货升水；负向高=期货贴水。",
        "subcategory": "basis",
        "parameters": {"window": 96},
        "code": dedent(
            """
            def calculate(data, window=96, **kwargs):
                if 'basis' not in data.columns:
                    return pd.Series(index=data.index, dtype=float)
                x = data['basis']
                return (x - x.rolling(window).mean()) / x.rolling(window).std()
            """
        ),
    },
    {
        "factor_id": "basis_ma_deviation",
        "name": "基差对均值的偏离",
        "description": "basis 减去其 window 滚动均值。低频 mean-reversion 信号。",
        "subcategory": "basis",
        "parameters": {"window": 48},
        "code": dedent(
            """
            def calculate(data, window=48, **kwargs):
                if 'basis' not in data.columns:
                    return pd.Series(index=data.index, dtype=float)
                return data['basis'] - data['basis'].rolling(window).mean()
            """
        ),
    },
]


# ==========================================================================
# funding/*
# ==========================================================================

FUNDING_FACTORS = [
    {
        "factor_id": "funding_rate_current",
        "name": "当前资金费率",
        "description": "当前资金费率（ffill 到 K 线），正=多头付空头，负=空头付多头。",
        "subcategory": "funding_carry",
        "parameters": {},
        "code": dedent(
            """
            def calculate(data, **kwargs):
                if 'funding_rate' not in data.columns:
                    return pd.Series(index=data.index, dtype=float)
                return data['funding_rate']
            """
        ),
    },
    {
        "factor_id": "funding_rate_ma",
        "name": "资金费率滚动均值",
        "description": "funding_rate 在 window 小时上的滚动均值，反映中期 carry 水平。",
        "subcategory": "funding_carry",
        "parameters": {"window": 72},
        "code": dedent(
            """
            def calculate(data, window=72, **kwargs):
                if 'funding_rate' not in data.columns:
                    return pd.Series(index=data.index, dtype=float)
                return data['funding_rate'].rolling(window).mean()
            """
        ),
    },
    {
        "factor_id": "funding_rate_zscore",
        "name": "资金费率 z-score",
        "description": "funding_rate 在 window 上的 z-score，识别异常拥挤仓位。",
        "subcategory": "funding_carry",
        "parameters": {"window": 240},
        "code": dedent(
            """
            def calculate(data, window=240, **kwargs):
                if 'funding_rate' not in data.columns:
                    return pd.Series(index=data.index, dtype=float)
                x = data['funding_rate']
                return (x - x.rolling(window).mean()) / x.rolling(window).std()
            """
        ),
    },
    {
        "factor_id": "funding_rate_sign_balance",
        "name": "资金费率符号占比差",
        "description": "近 window 根 K 线内正 funding 占比减去负占比，"
        "接近 +1 = 持续升水拥挤，接近 -1 = 持续贴水拥挤。",
        "subcategory": "funding_arbitrage",
        "parameters": {"window": 168},
        "code": dedent(
            """
            def calculate(data, window=168, **kwargs):
                if 'funding_rate' not in data.columns:
                    return pd.Series(index=data.index, dtype=float)
                sign = np.sign(data['funding_rate']).fillna(0)
                return sign.rolling(window).mean()
            """
        ),
    },
]


def main() -> None:
    for spec in DERIV_FACTORS:
        _write_factor(
            group="derivatives",
            factor_id=spec["factor_id"],
            name=spec["name"],
            description=spec["description"],
            subcategory=spec["subcategory"],
            function_code=spec["code"],
            parameters=spec.get("parameters"),
        )
    for spec in FUNDING_FACTORS:
        _write_factor(
            group="funding",
            factor_id=spec["factor_id"],
            name=spec["name"],
            description=spec["description"],
            subcategory=spec["subcategory"],
            function_code=spec["code"],
            parameters=spec.get("parameters"),
        )

    print(
        f"derivatives/: 新增 {len(DERIV_FACTORS)} 个因子；"
        f"funding/: 新增 {len(FUNDING_FACTORS)} 个因子"
    )


if __name__ == "__main__":
    main()
