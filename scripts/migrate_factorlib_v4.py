"""
因子库 v4 目录迁移脚本

作用：
  把历史的 factorlib/technicals/ 和 factorlib/minactors/ 迁移到
  v4 新结构 factorlib/{basic_kline, derivatives, funding}/，
  并规整 JSON 的 category 与 subcategory 两层分类。

  同时：
  - 删除「真正的」ML 预训练因子（category="ml" 且函数内加载 .pkl）
  - `mined_cross_sectional_*` 保持 computation_type="ml_model" 不变
    （它们实际是 user_algo/gp_cross_sectional.py 的算法代理，不是 ML 预训练）
  - `technical_mining_*` / `Hazel-*` 等挖掘因子 → basic_kline，subcategory=mined

使用：
  # 预演（不落盘）
  python scripts/migrate_factorlib_v4.py --dry-run

  # 真正执行迁移（旧目录会被重命名为 *_archived_YYYYMMDD 作为备份）
  python scripts/migrate_factorlib_v4.py --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).parent.parent
FACTORLIB = ROOT / "factorlib"

# 要删除的 ML 预训练因子 ID（category="ml" + 依赖 pkl 推理）
ML_PRETRAINED_IDS = {
    "adaptive_ml_factor",
    "rolling_ml_factor",
    "ensemble_gradient_boosting",
    "ensemble_lasso",
    "ensemble_random_forest",
    "ensemble_ridge",
    "feature_selection_f_regression_mean",
    "feature_selection_f_regression_weighted",
    "feature_selection_mutual_info_mean",
    "feature_selection_mutual_info_weighted",
    "ml_gradient_boosting",
    "ml_pca_component_1",
    "ml_pca_component_2",
    "ml_pca_component_3",
    "ml_selected_feature_1",
    "ml_selected_feature_2",
    "ml_selected_feature_3",
    "ml_selected_feature_4",
    "ml_selected_feature_5",
    # pca_component_1..10 为无监督降维，同样依赖预训练 pkl，一并删除
    *(f"pca_component_{i}" for i in range(1, 11)),
}

EVENT_KEYWORDS = (
    "cross", "gap", "breakout", "breakdown", "signal", "event", "direction",
    "engulfing", "morning_star", "evening_star", "hammer", "shooting_star",
    "doji", "fractal_up", "fractal_down", "triangle_pattern", "market_structure",
    "macd_cross", "gap_fill", "gap_up", "gap_down",
)

MINED_NAME_PREFIXES = (
    "mined_", "technical_mining_", "hazel-", "hazel_",
)


def _classify(def_data: Dict) -> Tuple[str, str]:
    """
    返回 (新一级 category, 新二级 subcategory)。

    规则：
      - ML 预训练因子（ID in ML_PRETRAINED_IDS）→ 返回 ("_DROP", "")
        由调用方删除。
      - 其余全部归到 basic_kline；二级按 id/name 关键字区分。
    """
    factor_id = (def_data.get("factor_id") or "").lower()
    name = (def_data.get("name") or "").lower()
    category = (def_data.get("category") or "").lower()

    if factor_id in ML_PRETRAINED_IDS or category == "ml":
        return ("_DROP", "")

    # 事件因子（形态/交叉/突破）→ subcategory=event
    for kw in EVENT_KEYWORDS:
        if kw in factor_id or kw in name:
            return ("basic_kline", "event")
    if category == "pattern":
        return ("basic_kline", "event")

    # 挖掘因子（GP、technical mining、Hazel 等）
    for prefix in MINED_NAME_PREFIXES:
        if factor_id.startswith(prefix) or name.startswith(prefix):
            return ("basic_kline", "mined")
    if category in ("gp_cs", "rl_cs", "mined_factor", "挖掘因子"):
        return ("basic_kline", "mined")

    # 其它：默认技术指标
    return ("basic_kline", "technical")


def _collect_legacy(factorlib: Path) -> List[Tuple[str, Path]]:
    """返回 [(legacy_group, def_file_path), ...]。"""
    out: List[Tuple[str, Path]] = []
    for legacy in ("technicals", "minactors"):
        dirp = factorlib / legacy / "definitions"
        if not dirp.exists():
            continue
        for f in sorted(dirp.glob("*.json")):
            out.append((legacy, f))
    return out


def _migrate_one(
    legacy: str, def_file: Path, factorlib: Path, dry_run: bool, report: Dict
) -> None:
    try:
        with open(def_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        report["errors"].append(f"读取失败 {def_file}: {e}")
        return

    factor_id = data.get("factor_id") or def_file.stem
    new_cat, new_sub = _classify(data)

    if new_cat == "_DROP":
        report["drop"].append(factor_id)
        if not dry_run:
            _drop_factor(legacy, factor_id, factorlib)
        return

    # 更新 JSON：category → 一级目录名；subcategory → 二级
    data["category"] = new_cat
    data["subcategory"] = new_sub or data.get("subcategory", "")

    # 修正 function_file 路径（老：technicals/functions/xxx.py → basic_kline/functions/xxx.py）
    comp_data = data.get("computation_data") or {}
    func_file = comp_data.get("function_file")
    if func_file:
        legacy_prefix = f"{legacy}/functions/"
        if func_file.startswith(legacy_prefix) or func_file.startswith(
            legacy_prefix.replace("/", "\\")
        ):
            fname = Path(func_file).name
            comp_data["function_file"] = f"{new_cat}/functions/{fname}"
            data["computation_data"] = comp_data

    dst_def_dir = factorlib / new_cat / "definitions"
    dst_func_dir = factorlib / new_cat / "functions"
    dst_eval_dir = factorlib / new_cat / "evaluations"
    dst_def_dir.mkdir(parents=True, exist_ok=True)
    dst_func_dir.mkdir(parents=True, exist_ok=True)
    dst_eval_dir.mkdir(parents=True, exist_ok=True)

    # 目标路径
    dst_def = dst_def_dir / def_file.name
    src_func = factorlib / legacy / "functions" / f"{factor_id}.py"
    dst_func = dst_func_dir / f"{factor_id}.py"
    src_eval = factorlib / legacy / "evaluations" / def_file.name
    dst_eval = dst_eval_dir / def_file.name

    report["migrated"].append(
        {
            "factor_id": factor_id,
            "from": legacy,
            "to": new_cat,
            "subcategory": new_sub,
        }
    )

    if dry_run:
        return

    # 写 JSON
    with open(dst_def, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 复制 function 源文件
    if src_func.exists() and not dst_func.exists():
        shutil.copy2(src_func, dst_func)

    # 复制评估历史（如有）
    if src_eval.exists() and not dst_eval.exists():
        shutil.copy2(src_eval, dst_eval)


def _drop_factor(legacy: str, factor_id: str, factorlib: Path) -> None:
    """删除 ML 预训练因子的 JSON + py + pkl。"""
    candidates = [
        factorlib / legacy / "definitions" / f"{factor_id}.json",
        factorlib / legacy / "functions" / f"{factor_id}.py",
        factorlib / legacy / "models" / f"{factor_id}.pkl",
        factorlib / legacy / "evaluations" / f"{factor_id}.json",
    ]
    # 同时删除可能已经迁移到新位置的副本
    for group in ("basic_kline", "derivatives", "funding"):
        candidates += [
            factorlib / group / "definitions" / f"{factor_id}.json",
            factorlib / group / "functions" / f"{factor_id}.py",
            factorlib / group / "evaluations" / f"{factor_id}.json",
        ]
    for p in candidates:
        if p.exists():
            p.unlink()


def _archive_legacy_dirs(factorlib: Path, dry_run: bool) -> List[str]:
    """把 technicals/ 和 minactors/ 重命名为 *_archived_{date} 作为只读备份。"""
    stamp = datetime.now().strftime("%Y%m%d")
    archived: List[str] = []
    for legacy in ("technicals", "minactors"):
        src = factorlib / legacy
        if not src.exists():
            continue
        dst = factorlib / f"{legacy}_archived_{stamp}"
        archived.append(f"{src} → {dst}")
        if not dry_run:
            if dst.exists():
                # 再加个短 suffix 避免覆盖
                dst = factorlib / f"{legacy}_archived_{stamp}_{datetime.now().strftime('%H%M%S')}"
            src.rename(dst)
    return archived


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="预演不落盘")
    parser.add_argument("--apply", action="store_true", help="真正执行迁移")
    args = parser.parse_args()

    if not args.dry_run and not args.apply:
        parser.error("必须指定 --dry-run 或 --apply 之一")

    dry_run = args.dry_run and not args.apply

    report: Dict = {
        "migrated": [],
        "drop": [],
        "errors": [],
        "archived_dirs": [],
    }

    legacy_files = _collect_legacy(FACTORLIB)
    print(f"发现 {len(legacy_files)} 个历史因子定义需要处理")

    for legacy, def_file in legacy_files:
        _migrate_one(legacy, def_file, FACTORLIB, dry_run, report)

    if not dry_run:
        report["archived_dirs"] = _archive_legacy_dirs(FACTORLIB, dry_run=False)
    else:
        report["archived_dirs"] = _archive_legacy_dirs(FACTORLIB, dry_run=True)

    # 输出报告
    print("\n===== 迁移报告 =====")
    print(f"迁移:   {len(report['migrated'])}")
    print(f"删除:   {len(report['drop'])}")
    print(f"错误:   {len(report['errors'])}")
    print(f"归档:   {report['archived_dirs']}")
    print()
    if report["drop"]:
        print("待删除的 ML 预训练因子:")
        for fid in report["drop"]:
            print(f"  - {fid}")
    if report["errors"]:
        print("错误:")
        for e in report["errors"]:
            print(f"  ! {e}")

    # 按二级分类统计
    sub_counts: Dict[str, int] = {}
    for item in report["migrated"]:
        key = f"{item['to']}/{item['subcategory']}"
        sub_counts[key] = sub_counts.get(key, 0) + 1
    print("\n新分类分布:")
    for key, cnt in sorted(sub_counts.items()):
        print(f"  {key:32s} {cnt}")

    # 报告落盘
    out_report = FACTORLIB / "exports" / (
        f"migration_v4_{'dryrun' if dry_run else 'applied'}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n详细报告: {out_report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
