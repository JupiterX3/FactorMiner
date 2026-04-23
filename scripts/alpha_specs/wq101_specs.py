"""
WorldQuant Alpha101 可实现子集 — 约 38 个因子。

语义折衷（详见 design/qlib_wq101_factors_integration_20260422.md §6.2）：
- 最外层 `rank(x)` → 直接返回 `x`（截面评估自会排名）
- 中间层 `rank(x)` → 直接使用 `x`（丢失 cross-sectional Spearman 近似）
  或用 `ts_rank(x, 10)` 替代（时序 pct rank）
- `indneutralize(x, g)` → 直接返回 `x`（加密市场无行业映射）
- `adv20` → `ts_mean(volume, 20)`（单币）
- `vwap` → `(H+L+C)/3` 近似
- `cap` / `IndClass.*` → 放弃整个 alpha

每个因子的 description 必须包含原始公式及折衷说明。
"""
from typing import List, Dict


BASE_IMPORTS = [
    "import pandas as pd",
    "import numpy as np",
    "from _alpha_ops import *  # noqa: F401,F403",
]


def _spec(
    alpha_no: str,
    raw_formula: str,
    simplifications: str,
    body_expr: str,
    min_warmup: int,
    is_window: bool = True,
) -> Dict:
    """统一构造 wq101 规格字典。"""
    fid = f"wq101_alpha{alpha_no}"
    desc = (
        f"WorldQuant Alpha#{alpha_no}。原始公式：{raw_formula}。"
        f" 语义折衷：{simplifications}。"
    )
    return {
        "factor_id": fid,
        "name": f"WQ101-Alpha{alpha_no}",
        "description": desc,
        "category": "wq101",
        "subcategory": f"alpha{alpha_no}",
        "is_window": is_window,
        "min_warmup_bars": int(min_warmup),
        "imports": BASE_IMPORTS,
        "body": (
            "def calculate(data, **kwargs):\n"
            "    close = data['close']\n"
            "    open_ = data['open']\n"
            "    high = data['high']\n"
            "    low = data['low']\n"
            "    volume = data['volume']\n"
            "    ret = returns(close)\n"
            "    vwap = vwap_proxy(data)\n"
            f"    return {body_expr}\n"
        ),
    }


def build_specs() -> List[Dict]:
    specs: List[Dict] = []

    specs.append(_spec(
        "001",
        "rank(Ts_ArgMax(SignedPower((returns<0?stddev(returns,20):close), 2.), 5)) - 0.5",
        "去掉最外层 rank；返回 Ts_ArgMax/5 居中（保留原公式的[-0.5,0.5] 区间语义）",
        "ts_argmax(signed_power(where(ret < 0, ts_std(ret, 20), close), 2.0), 5) / 5.0 - 0.5",
        min_warmup=25,
    ))

    specs.append(_spec(
        "002",
        "-1 * correlation(rank(delta(log(volume),2)), rank((close-open)/open), 6)",
        "去掉两个内层 rank（Spearman→Pearson）",
        "-1 * ts_corr(delta(log(volume), 2), (close - open_) / (open_ + EPS), 6)",
        min_warmup=8,
    ))

    specs.append(_spec(
        "003",
        "-1 * correlation(rank(open), rank(volume), 10)",
        "去掉内层 rank",
        "-1 * ts_corr(open_, volume, 10)",
        min_warmup=10,
    ))

    specs.append(_spec(
        "004",
        "-1 * Ts_Rank(rank(low), 9)",
        "去掉内层 rank",
        "-1 * ts_rank(low, 9)",
        min_warmup=9,
    ))

    specs.append(_spec(
        "005",
        "rank(open - sum(vwap,10)/10) * (-1 * abs(rank(close - vwap)))",
        "去掉全部 rank（丢失截面语义，保留时序方向）",
        "(open_ - ts_mean(vwap, 10)) * (-1 * (close - vwap).abs())",
        min_warmup=10,
    ))

    specs.append(_spec(
        "006",
        "-1 * correlation(open, volume, 10)",
        "原式无 rank，直接实现",
        "-1 * ts_corr(open_, volume, 10)",
        min_warmup=10,
    ))

    specs.append(_spec(
        "007",
        "(adv20<volume) ? (-1*ts_rank(abs(delta(close,7)),60)*sign(delta(close,7))) : -1",
        "直接实现；adv20 = ts_mean(volume,20)",
        "pd.Series(np.where(adv(volume, 20) < volume, "
        "-ts_rank(delta(close, 7).abs(), 60) * sign(delta(close, 7)), -1.0), index=close.index)",
        min_warmup=67,
    ))

    specs.append(_spec(
        "008",
        "-1 * rank((sum(open,5)*sum(returns,5)) - delay(sum(open,5)*sum(returns,5), 10))",
        "去掉最外层 rank",
        "-1 * (ts_sum(open_, 5) * ts_sum(ret, 5) - ref(ts_sum(open_, 5) * ts_sum(ret, 5), 10))",
        min_warmup=16,
    ))

    specs.append(_spec(
        "009",
        "(0<ts_min(delta(close,1),5)) ? delta(close,1) : ((ts_max(delta(close,1),5)<0) ? delta(close,1) : -delta(close,1))",
        "直接实现",
        "pd.Series(np.where(ts_min(delta(close, 1), 5) > 0, delta(close, 1), "
        "np.where(ts_max(delta(close, 1), 5) < 0, delta(close, 1), -delta(close, 1))), index=close.index)",
        min_warmup=6,
    ))

    specs.append(_spec(
        "010",
        "rank((0<ts_min(delta(close,1),4)) ? delta(close,1) : ((ts_max(delta(close,1),4)<0) ? delta(close,1) : -delta(close,1)))",
        "去掉最外层 rank；逻辑同 alpha009 但窗口=4",
        "pd.Series(np.where(ts_min(delta(close, 1), 4) > 0, delta(close, 1), "
        "np.where(ts_max(delta(close, 1), 4) < 0, delta(close, 1), -delta(close, 1))), index=close.index)",
        min_warmup=5,
    ))

    specs.append(_spec(
        "012",
        "sign(delta(volume,1)) * (-1*delta(close,1))",
        "原式无 rank",
        "sign(delta(volume, 1)) * (-1 * delta(close, 1))",
        min_warmup=2,
        is_window=False,
    ))

    specs.append(_spec(
        "013",
        "-1 * rank(covariance(rank(close), rank(volume), 5))",
        "去掉全部 rank",
        "-1 * ts_cov(close, volume, 5)",
        min_warmup=5,
    ))

    specs.append(_spec(
        "014",
        "(-1 * rank(delta(returns,3))) * correlation(open, volume, 10)",
        "去掉最外层 rank",
        "-1 * delta(ret, 3) * ts_corr(open_, volume, 10)",
        min_warmup=10,
    ))

    specs.append(_spec(
        "015",
        "-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)",
        "去掉全部 rank",
        "-1 * ts_sum(ts_corr(high, volume, 3), 3)",
        min_warmup=6,
    ))

    specs.append(_spec(
        "016",
        "-1 * rank(covariance(rank(high), rank(volume), 5))",
        "去掉全部 rank",
        "-1 * ts_cov(high, volume, 5)",
        min_warmup=5,
    ))

    specs.append(_spec(
        "018",
        "-1 * rank(stddev(abs(close-open),5) + (close-open) + correlation(close, open, 10))",
        "去掉最外层 rank",
        "-1 * (ts_std((close - open_).abs(), 5) + (close - open_) + ts_corr(close, open_, 10))",
        min_warmup=10,
    ))

    specs.append(_spec(
        "019",
        "-sign(((close - delay(close,7)) + delta(close,7))) * (1 + rank(1 + sum(returns, 250)))",
        "去掉最外层 rank；保留 250 期长窗口",
        "-sign((close - ref(close, 7)) + delta(close, 7)) * (2.0 + ts_sum(ret, 250))",
        min_warmup=250,
    ))

    specs.append(_spec(
        "020",
        "-rank(open-delay(high,1)) * rank(open-delay(close,1)) * rank(open-delay(low,1))",
        "去掉全部 rank（三元乘积保留符号结构）",
        "-1 * (open_ - ref(high, 1)) * (open_ - ref(close, 1)) * (open_ - ref(low, 1))",
        min_warmup=2,
        is_window=False,
    ))

    specs.append(_spec(
        "021",
        "((sum(close,8)/8 + stddev(close,8)) < sum(close,2)/2) ? -1 : "
        "((sum(close,2)/2 < (sum(close,8)/8 - stddev(close,8))) ? 1 : "
        "((volume/adv20 >= 1) ? 1 : -1))",
        "直接实现；adv20 = ts_mean(volume,20)",
        "pd.Series(np.where(ts_sum(close, 8) / 8 + ts_std(close, 8) < ts_sum(close, 2) / 2, -1.0, "
        "np.where(ts_sum(close, 2) / 2 < ts_sum(close, 8) / 8 - ts_std(close, 8), 1.0, "
        "np.where(volume / (adv(volume, 20) + EPS) >= 1.0, 1.0, -1.0))), index=close.index)",
        min_warmup=20,
    ))

    specs.append(_spec(
        "022",
        "-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close,20)))",
        "去掉最外层 rank",
        "-1 * delta(ts_corr(high, volume, 5), 5) * ts_std(close, 20)",
        min_warmup=20,
    ))

    specs.append(_spec(
        "028",
        "scale(correlation(adv20, low, 5) + (high+low)/2 - close)",
        "去掉 scale（截面层归一化），adv20 = ts_mean(volume,20)",
        "ts_corr(adv(volume, 20), low, 5) + (high + low) / 2 - close",
        min_warmup=20,
    ))

    specs.append(_spec(
        "034",
        "(1 - rank(stddev(returns,2)/stddev(returns,5))) + (1 - rank(delta(close,1)))",
        "去掉两个 rank；改为直接求和（仍保留单调方向）",
        "(1.0 - ts_std(ret, 2) / (ts_std(ret, 5) + EPS)) + (1.0 - delta(close, 1))",
        min_warmup=6,
    ))

    specs.append(_spec(
        "035",
        "ts_rank(volume,32) * (1-ts_rank((close+high-low),16)) * (1-ts_rank(returns,32))",
        "原式已是 ts_rank，直接实现",
        "ts_rank(volume, 32) * (1.0 - ts_rank(close + high - low, 16)) * (1.0 - ts_rank(ret, 32))",
        min_warmup=32,
    ))

    specs.append(_spec(
        "037",
        "rank(correlation(delay(open-close,1), close, 200)) + rank(open-close)",
        "去掉 rank；200 → 60 以缩短冷启动",
        "ts_corr(ref(open_ - close, 1), close, 60) + (open_ - close)",
        min_warmup=61,
    ))

    specs.append(_spec(
        "038",
        "(-1 * rank(Ts_Rank(close,10))) * rank(close/open)",
        "去掉最外层 rank",
        "-1 * ts_rank(close, 10) * (close / (open_ + EPS))",
        min_warmup=10,
    ))

    specs.append(_spec(
        "040",
        "(-1 * rank(stddev(high, 10))) * correlation(high, volume, 10)",
        "去掉最外层 rank",
        "-1 * ts_std(high, 10) * ts_corr(high, volume, 10)",
        min_warmup=10,
    ))

    specs.append(_spec(
        "043",
        "ts_rank(volume/adv20, 20) * ts_rank(-1*delta(close,7), 8)",
        "原式已是 ts_rank，直接实现",
        "ts_rank(volume / (adv(volume, 20) + EPS), 20) * ts_rank(-1 * delta(close, 7), 8)",
        min_warmup=20,
    ))

    specs.append(_spec(
        "046",
        "0.25 < ((delay(close,20)-delay(close,10))/10 - (delay(close,10)-close)/10) ? -1 : "
        "(((delay(close,20)-delay(close,10))/10 - (delay(close,10)-close)/10) < 0 ? 1 : (-1*(close-delay(close,1))))",
        "直接实现",
        "pd.Series(np.where(((ref(close, 20) - ref(close, 10)) / 10 - (ref(close, 10) - close) / 10) > 0.25, -1.0, "
        "np.where(((ref(close, 20) - ref(close, 10)) / 10 - (ref(close, 10) - close) / 10) < 0, 1.0, "
        "-(close - ref(close, 1)))), index=close.index)",
        min_warmup=21,
    ))

    specs.append(_spec(
        "049",
        "(((delay(close,20)-delay(close,10))/10) - ((delay(close,10)-close)/10)) < -0.1 ? 1 : -(close-delay(close,1))",
        "直接实现（WQ 原式阈值 -0.1）",
        "pd.Series(np.where(((ref(close, 20) - ref(close, 10)) / 10 - (ref(close, 10) - close) / 10) < -0.1, 1.0, "
        "-(close - ref(close, 1))), index=close.index)",
        min_warmup=21,
    ))

    specs.append(_spec(
        "050",
        "-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5)",
        "去掉全部 rank",
        "-1 * ts_max(ts_corr(volume, vwap, 5), 5)",
        min_warmup=10,
    ))

    specs.append(_spec(
        "051",
        "(((delay(close,20)-delay(close,10))/10) - ((delay(close,10)-close)/10)) < -0.05 ? 1 : -(close-delay(close,1))",
        "直接实现（WQ 原式阈值 -0.05）",
        "pd.Series(np.where(((ref(close, 20) - ref(close, 10)) / 10 - (ref(close, 10) - close) / 10) < -0.05, 1.0, "
        "-(close - ref(close, 1))), index=close.index)",
        min_warmup=21,
    ))

    specs.append(_spec(
        "053",
        "-1 * delta(((close-low)-(high-close)) / (close-low), 9)",
        "原式无 rank，直接实现",
        "-1 * delta(((close - low) - (high - close)) / (close - low + EPS), 9)",
        min_warmup=10,
    ))

    specs.append(_spec(
        "054",
        "-1 * ((low-close) * (open^5)) / ((low-high) * (close^5))",
        "原式无 rank",
        "-1 * ((low - close) * open_**5) / ((low - high) * close**5 + EPS)",
        min_warmup=1,
        is_window=False,
    ))

    specs.append(_spec(
        "055",
        "-1 * correlation(rank((close-ts_min(low,12))/(ts_max(high,12)-ts_min(low,12))), rank(volume), 6)",
        "去掉全部 rank",
        "-1 * ts_corr((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12) + EPS), volume, 6)",
        min_warmup=18,
    ))

    specs.append(_spec(
        "083",
        "(rank(delay((high-low)/(sum(close,5)/5),2)) * rank(rank(volume))) / "
        "(((high-low)/(sum(close,5)/5)) / (vwap-close))",
        "去掉全部 rank；vwap ≈ (H+L+C)/3",
        "(ref((high - low) / (ts_sum(close, 5) / 5 + EPS), 2) * volume) / "
        "(((high - low) / (ts_sum(close, 5) / 5 + EPS)) / (vwap - close + EPS) + EPS)",
        min_warmup=7,
    ))

    specs.append(_spec(
        "084",
        "SignedPower(Ts_Rank(vwap - ts_max(vwap,15), 21), delta(close,5))",
        "直接实现",
        "signed_power(ts_rank(vwap - ts_max(vwap, 15), 21), 1.0) * "
        "sign(delta(close, 5)) * (ts_rank(vwap - ts_max(vwap, 15), 21).abs() ** delta(close, 5).abs().clip(upper=10))",
        min_warmup=36,
    ))

    specs.append(_spec(
        "099",
        "(rank(correlation(sum((high+low)/2,20), sum(adv60,20), 9)) < rank(correlation(low,volume,6))) ? -1 : 1",
        "去掉全部 rank；adv60 = ts_mean(volume,60)",
        "pd.Series(np.where(ts_corr(ts_sum((high + low) / 2, 20), ts_sum(adv(volume, 60), 20), 9) < "
        "ts_corr(low, volume, 6), -1.0, 1.0), index=close.index)",
        min_warmup=80,
    ))

    specs.append(_spec(
        "101",
        "(close-open) / (high-low+0.001)",
        "原式无 rank",
        "(close - open_) / (high - low + 0.001)",
        min_warmup=1,
        is_window=False,
    ))

    return specs


if __name__ == "__main__":
    specs = build_specs()
    print(f"WQ101 可实现子集规格数量：{len(specs)}")
    for s in specs[:5]:
        print(f"  {s['factor_id']} / warmup={s['min_warmup_bars']} / is_window={s['is_window']}")
