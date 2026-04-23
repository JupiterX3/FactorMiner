"""
截面K线因子重复检测脚本（优化版）

核心优化：
1. 向量化Spearman计算 - 每个截面一次性计算全量相关矩阵
2. 减少数据量 - 20币种 x 300根K线，跳过前50根warm-up
3. rank后直接corr - 等价于Spearman但快得多
"""

import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from scipy import stats

os.environ['PYTHONIOENCODING'] = 'utf-8'
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from factor_miner.core.factor_engine import FactorEngine
from factor_miner.core.factor_storage import TransparentFactorStorage


N_SYMBOLS = 20
N_BARS = 300
WARMUP = 50
SEED = 42
SPEARMAN_THRESHOLD = 0.999
RANK_CONSISTENCY_THRESHOLD = 0.99
MIN_VALID_RATIO = 0.5


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def generate_random_ohlcv(n_bars: int, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, n_bars)))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n_bars)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n_bars)))
    open_ = close * (1 + rng.normal(0, 0.003, n_bars))
    volume = np.exp(rng.normal(15, 1.5, n_bars))
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    dates = pd.date_range('2024-01-01', periods=n_bars, freq='1h')
    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume
    }, index=dates)


def get_kline_factor_ids(storage: TransparentFactorStorage) -> list:
    """扫描 basic_kline/ 目录下所有非事件、非 ML 类因子，作为查重基准集。"""
    basic_dir = storage.storage_dir / "basic_kline" / "definitions"
    if not basic_dir.exists():
        return []
    factor_ids = []
    event_keywords = ['cross', 'gap', 'breakout', 'breakdown', 'signal',
                      'event', 'direction', 'engulfing', 'morning_star',
                      'evening_star', 'hammer', 'shooting_star', 'doji',
                      'fractal_up', 'fractal_down', 'triangle_pattern',
                      'market_structure', 'macd_cross', 'gap_fill',
                      'gap_up', 'gap_down']
    for f in basic_dir.glob("*.json"):
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
            comp_type = data.get('computation_type', '')
            subcategory = (data.get('subcategory') or '').lower()
            factor_id = data.get('factor_id', '').lower()
            factor_name = data.get('name', '').lower()
            # v4 已移除 ml_model；兼容旧 JSON
            if comp_type == 'ml_model':
                continue
            if subcategory == 'event':
                continue
            is_event = False
            for kw in event_keywords:
                if kw in factor_id or kw in factor_name:
                    is_event = True
                    break
            if is_event:
                continue
            factor_ids.append(data['factor_id'])
        except Exception:
            continue
    return factor_ids


def compute_all_factors(engine: FactorEngine, factor_ids: list,
                        n_symbols: int, n_bars: int) -> tuple:
    all_data = {}
    for s in range(n_symbols):
        all_data[f'sym_{s}'] = generate_random_ohlcv(n_bars, seed=SEED + s)

    factor_values = {}
    failed = []
    for i, fid in enumerate(factor_ids):
        sym_results = {}
        ok = True
        for sym, data in all_data.items():
            try:
                val = engine.compute_single_factor(fid, data)
                if val is None:
                    ok = False
                    break
                val = pd.to_numeric(val, errors='coerce')
                val.index = data.index
                sym_results[sym] = val
            except Exception:
                ok = False
                break
        if ok and len(sym_results) == n_symbols:
            factor_values[fid] = sym_results
        else:
            failed.append(fid)
        if (i + 1) % 20 == 0 or (i + 1) == len(factor_ids):
            print(f"  [{i+1}/{len(factor_ids)}] {len(factor_values)} ok, {len(failed)} failed")

    if failed:
        print(f"\n[WARN] {len(failed)} factors failed:")
        for f in failed[:10]:
            print(f"  - {f}")
        if len(failed) > 10:
            print(f"  ... and {len(failed)-10} more")

    return factor_values, all_data


def build_panel_and_compute_spearman(factor_values: dict, all_data: dict,
                                      warmup: int) -> tuple:
    successful_ids = list(factor_values.keys())
    n_factors = len(successful_ids)
    n_symbols = len(all_data)
    sym_names = list(all_data.keys())

    sample_data = list(all_data.values())[0]
    dates = sample_data.index[warmup:]
    n_dates = len(dates)

    print(f"  Panel: {n_factors} factors x {n_symbols} symbols x {n_dates} dates (after warmup={warmup})")

    panel = np.full((n_dates, n_symbols, n_factors), np.nan)

    for f_idx, fid in enumerate(successful_ids):
        for s_idx, sym in enumerate(sym_names):
            series = factor_values[fid][sym]
            panel[:, s_idx, f_idx] = series.values[warmup:]

    spearman_sum = np.zeros((n_factors, n_factors))
    rank_consist_sum = np.zeros((n_factors, n_factors))
    valid_date_count = np.zeros((n_factors, n_factors), dtype=int)

    min_valid = max(10, int(n_symbols * MIN_VALID_RATIO))

    for t in range(n_dates):
        if (t + 1) % 50 == 0:
            print(f"  cross-section progress: {t+1}/{n_dates}")

        cross_section = panel[t, :, :]

        df = pd.DataFrame(cross_section, columns=successful_ids)

        valid_per_col = df.notna().sum()
        valid_cols = valid_per_col[valid_per_col >= min_valid].index.tolist()
        if len(valid_cols) < 2:
            continue

        df_valid = df[valid_cols].copy()
        for col in df_valid.columns:
            s = df_valid[col].dropna()
            if s.std() < 1e-12:
                df_valid[col] = np.nan

        corr_matrix = df_valid.corr(method='spearman')

        ranked = df_valid.rank()
        rank_eq_count = np.zeros((len(valid_cols), len(valid_cols)))
        rank_eq_total = np.zeros((len(valid_cols), len(valid_cols)), dtype=int)

        for ci, col_i in enumerate(valid_cols):
            ri = ranked[col_i].values
            vi = ~np.isnan(ri)
            for cj, col_j in enumerate(valid_cols):
                if cj <= ci:
                    continue
                rj = ranked[col_j].values
                vj = ~np.isnan(rj)
                both = vi & vj
                n_both = both.sum()
                if n_both < min_valid:
                    continue
                eq = (ri[both] == rj[both]).sum()
                rank_eq_count[ci, cj] = eq
                rank_eq_count[cj, ci] = eq
                rank_eq_total[ci, cj] = n_both
                rank_eq_total[cj, ci] = n_both

        for ci, col_i in enumerate(valid_cols):
            for cj, col_j in enumerate(valid_cols):
                if cj < ci:
                    continue
                rho = corr_matrix.loc[col_i, col_j]
                if np.isnan(rho):
                    continue
                fi = successful_ids.index(col_i)
                fj = successful_ids.index(col_j)
                spearman_sum[fi, fj] += rho
                spearman_sum[fj, fi] += rho
                valid_date_count[fi, fj] += 1
                valid_date_count[fj, fi] += 1
                if ci != cj and rank_eq_total[ci, cj] > 0:
                    rc = rank_eq_count[ci, cj] / rank_eq_total[ci, cj]
                    rank_consist_sum[fi, fj] += rc
                    rank_consist_sum[fj, fi] += rc

    with np.errstate(invalid='ignore', divide='ignore'):
        mean_spearman = np.where(valid_date_count > 0,
                                  spearman_sum / valid_date_count, 0.0)
        mean_rank_consist = np.where(valid_date_count > 0,
                                      rank_consist_sum / valid_date_count, 0.0)

    return successful_ids, mean_spearman, mean_rank_consist, valid_date_count


def find_duplicate_groups(factor_ids: list, mean_spearman: np.ndarray,
                           mean_rank_consist: np.ndarray,
                           valid_date_count: np.ndarray,
                           spearman_thresh: float,
                           rank_consist_thresh: float) -> list:
    n = len(factor_ids)
    uf = UnionFind(n)

    duplicate_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if valid_date_count[i, j] < 10:
                continue
            sp = mean_spearman[i, j]
            rc = mean_rank_consist[i, j]
            if sp > spearman_thresh and rc > rank_consist_thresh:
                uf.union(i, j)
                duplicate_pairs.append((factor_ids[i], factor_ids[j], sp, rc))

    groups = {}
    for i in range(n):
        root = uf.find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    duplicate_groups = []
    for root, members in groups.items():
        if len(members) > 1:
            group_info = {
                'factors': [factor_ids[m] for m in members],
                'pair_details': []
            }
            for mi, mj in combinations(members, 2):
                sp = mean_spearman[mi, mj]
                rc = mean_rank_consist[mi, mj]
                vc = int(valid_date_count[mi, mj])
                group_info['pair_details'].append({
                    'factor_a': factor_ids[mi],
                    'factor_b': factor_ids[mj],
                    'mean_spearman': round(float(sp), 6),
                    'rank_consistency': round(float(rc), 6),
                    'valid_dates': vc
                })
            duplicate_groups.append(group_info)

    duplicate_groups.sort(key=lambda g: -len(g['factors']))

    return duplicate_groups, duplicate_pairs


def print_results(duplicate_groups: list, factor_ids: list,
                   mean_spearman: np.ndarray, mean_rank_consist: np.ndarray,
                   valid_date_count: np.ndarray):
    n = len(factor_ids)
    print("\n" + "=" * 80)
    print("[RESULT] Cross-Sectional K-Line Factor Duplicate Detection")
    print("=" * 80)

    total_pairs = n * (n - 1) // 2
    high_corr_pairs = sum(1 for i in range(n) for j in range(i+1, n)
                          if valid_date_count[i, j] > 0 and mean_spearman[i, j] > 0.95)
    very_high_corr = sum(1 for i in range(n) for j in range(i+1, n)
                         if valid_date_count[i, j] > 0 and mean_spearman[i, j] > 0.999)

    print(f"\nTotal factors: {n}")
    print(f"Total factor pairs: {total_pairs}")
    print(f"Spearman > 0.95: {high_corr_pairs} pairs")
    print(f"Spearman > 0.999: {very_high_corr} pairs")
    print(f"Duplicate groups: {len(duplicate_groups)}")
    print(f"Duplicate factors: {sum(len(g['factors']) for g in duplicate_groups)}")

    if not duplicate_groups:
        print("\n[OK] No duplicate factors found!")
        return

    print(f"\n{'='*80}")
    print("[DUPLICATE] Duplicate Factor Groups")
    print(f"{'='*80}")

    for gi, group in enumerate(duplicate_groups, 1):
        factors = group['factors']
        print(f"\n--- Group #{gi} ({len(factors)} factors) ---")
        for f in factors:
            print(f"  * {f}")
        print("  Pair details:")
        for pd_ in group['pair_details']:
            print(f"    {pd_['factor_a']} <-> {pd_['factor_b']}: "
                  f"Spearman={pd_['mean_spearman']:.6f}, "
                  f"RankConsistency={pd_['rank_consistency']:.4f}, "
                  f"ValidDates={pd_['valid_dates']}")

    print(f"\n{'='*80}")
    print("[NEAR] High-correlation pairs (0.95 < Spearman <= 0.999)")
    print(f"{'='*80}")

    near_dupes = []
    for i in range(n):
        for j in range(i + 1, n):
            if valid_date_count[i, j] < 10:
                continue
            sp = mean_spearman[i, j]
            rc = mean_rank_consist[i, j]
            if 0.95 < sp <= 0.999:
                near_dupes.append((factor_ids[i], factor_ids[j], sp, rc))

    near_dupes.sort(key=lambda x: -x[2])

    if near_dupes:
        for fa, fb, sp, rc in near_dupes[:30]:
            print(f"  {fa} <-> {fb}: Spearman={sp:.6f}, RankConsistency={rc:.4f}")
        if len(near_dupes) > 30:
            print(f"  ... {len(near_dupes) - 30} more pairs")
    else:
        print("  None")

    print(f"\n{'='*80}")
    print("[MATRIX] All pairs with Spearman > 0.9")
    print(f"{'='*80}")

    high_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if valid_date_count[i, j] > 0 and mean_spearman[i, j] > 0.9:
                high_pairs.append((factor_ids[i], factor_ids[j],
                                   float(mean_spearman[i, j]),
                                   float(mean_rank_consist[i, j]),
                                   int(valid_date_count[i, j])))

    high_pairs.sort(key=lambda x: -x[2])
    for fa, fb, sp, rc, vc in high_pairs:
        tag = "[DUP]" if sp > SPEARMAN_THRESHOLD and rc > RANK_CONSISTENCY_THRESHOLD else ""
        print(f"  {fa:40s} <-> {fb:40s}  rho={sp:.6f}  RC={rc:.4f}  n={vc}  {tag}")


def main():
    print("=" * 80)
    print("[START] Cross-Sectional K-Line Factor Duplicate Detection")
    print(f"Config: {N_SYMBOLS} symbols x {N_BARS} bars (warmup={WARMUP})")
    print(f"Threshold: Spearman > {SPEARMAN_THRESHOLD}, RankConsistency > {RANK_CONSISTENCY_THRESHOLD}")
    print("=" * 80)

    print("\n[1/5] Initializing factor engine...")
    storage = TransparentFactorStorage()
    engine = FactorEngine(storage)

    print("\n[2/5] Loading K-line factor list...")
    factor_ids = get_kline_factor_ids(storage)
    print(f"  Found {len(factor_ids)} K-line factors (excluded event/ML factors)")

    if not factor_ids:
        print("[ERROR] No K-line factors found")
        return

    print(f"\n[3/5] Generating random data and computing factors...")
    factor_values, all_data = compute_all_factors(engine, factor_ids, N_SYMBOLS, N_BARS)

    successful_ids = list(factor_values.keys())
    print(f"\n  Successfully computed {len(successful_ids)} factors")

    if len(successful_ids) < 2:
        print("[ERROR] Less than 2 factors computed, cannot detect duplicates")
        return

    print(f"\n[4/5] Building cross-sectional panel and computing Spearman matrix...")
    successful_ids, mean_spearman, mean_rank_consist, valid_date_count = \
        build_panel_and_compute_spearman(factor_values, all_data, WARMUP)

    print(f"\n[5/5] Identifying duplicate factor groups...")
    duplicate_groups, duplicate_pairs = find_duplicate_groups(
        successful_ids, mean_spearman, mean_rank_consist, valid_date_count,
        SPEARMAN_THRESHOLD, RANK_CONSISTENCY_THRESHOLD
    )

    print_results(duplicate_groups, successful_ids, mean_spearman,
                   mean_rank_consist, valid_date_count)

    output_path = PROJECT_ROOT / "duplicate_factors_report.json"
    report = {
        'config': {
            'n_symbols': N_SYMBOLS,
            'n_bars': N_BARS,
            'warmup': WARMUP,
            'seed': SEED,
            'spearman_threshold': SPEARMAN_THRESHOLD,
            'rank_consistency_threshold': RANK_CONSISTENCY_THRESHOLD,
        },
        'summary': {
            'total_factors': len(successful_ids),
            'duplicate_groups': len(duplicate_groups),
            'duplicate_factors': sum(len(g['factors']) for g in duplicate_groups),
        },
        'duplicate_groups': duplicate_groups,
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[SAVED] Report saved to: {output_path}")


if __name__ == '__main__':
    main()
