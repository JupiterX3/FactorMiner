#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
注册 qlib Alpha158 / Alpha360（精简） / WorldQuant Alpha101 因子到 factorlib。

用法：
    python scripts/register_qlib_wq_factors.py --family qlib158
    python scripts/register_qlib_wq_factors.py --family qlib360
    python scripts/register_qlib_wq_factors.py --family wq101
    python scripts/register_qlib_wq_factors.py --family all
    python scripts/register_qlib_wq_factors.py --family all --overwrite
    python scripts/register_qlib_wq_factors.py --family qlib158 --dry-run

产出（v4）：
    factorlib/basic_kline/definitions/<factor_id>.json
    factorlib/basic_kline/functions/<factor_id>.py

每个因子文件顶部默认 `from _alpha_ops import *`，由 factor_engine.py 的
`_ensure_sys_path(str(func_path.parent))` 机制保证可导入。
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.alpha_specs import (  # noqa: E402
    build_qlib158_specs,
    build_qlib360_specs,
    build_wq101_specs,
)

FACTORLIB_DIR = PROJECT_ROOT / "factorlib"
DEFS_DIR = FACTORLIB_DIR / "basic_kline" / "definitions"
FUNCS_DIR = FACTORLIB_DIR / "basic_kline" / "functions"


def _render_function_file(spec: Dict) -> str:
    """把 spec 渲染成最终写盘的 .py 文本。"""
    imports = spec.get("imports") or []
    body = spec["body"].rstrip() + "\n"
    return "\n".join(imports) + "\n\n" + body


def _render_definition(spec: Dict, function_rel_path: str) -> Dict:
    """构造 JSON 定义（与 FactorDefinition 结构一致）。"""
    function_code = spec["body"].rstrip() + "\n"
    factor_id = spec["factor_id"]

    checksum_content = f"{factor_id}_{spec['name']}_{function_code}"
    checksum = hashlib.md5(checksum_content.encode()).hexdigest()[:8]

    metadata = {
        "checksum": checksum,
        "created_at": datetime.now().isoformat(),
        "is_window": bool(spec.get("is_window", False)),
        "min_warmup_bars": int(spec.get("min_warmup_bars", 1)),
        "source_family": spec.get("category"),
    }

    return {
        "factor_id": factor_id,
        "name": spec["name"],
        "description": spec["description"],
        "category": spec["category"],
        "subcategory": spec.get("subcategory", ""),
        "computation_type": "function",
        "computation_data": {
            "function_file": function_rel_path,
            "function_code": function_code,
            "entry_point": "calculate",
            "imports": spec.get("imports") or [],
        },
        "parameters": {},
        "dependencies": [],
        "output_type": "series",
        "metadata": metadata,
    }


def _write_spec(spec: Dict, overwrite: bool, dry_run: bool) -> str:
    """
    写入单个 spec。返回 'created' / 'overwritten' / 'skipped' / 'dry-run'。
    """
    factor_id = spec["factor_id"]
    py_path = FUNCS_DIR / f"{factor_id}.py"
    json_path = DEFS_DIR / f"{factor_id}.json"
    function_rel_path = f"basic_kline/functions/{factor_id}.py"

    exists = py_path.exists() or json_path.exists()
    if exists and not overwrite:
        return "skipped"

    if dry_run:
        return "dry-run"

    FUNCS_DIR.mkdir(parents=True, exist_ok=True)
    DEFS_DIR.mkdir(parents=True, exist_ok=True)

    with open(py_path, "w", encoding="utf-8") as f:
        f.write(_render_function_file(spec))

    definition = _render_definition(spec, function_rel_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(definition, f, ensure_ascii=False, indent=2)

    return "overwritten" if exists else "created"


def _collect_specs(family: str) -> List[Dict]:
    if family == "qlib158":
        return build_qlib158_specs()
    if family == "qlib360":
        return build_qlib360_specs()
    if family == "wq101":
        return build_wq101_specs()
    if family == "all":
        out: List[Dict] = []
        out.extend(build_qlib158_specs())
        out.extend(build_qlib360_specs())
        out.extend(build_wq101_specs())
        return out
    raise ValueError(f"未知的 factor family: {family}")


def main() -> int:
    parser = argparse.ArgumentParser(description="注册 qlib/WQ101 因子到 factorlib。")
    parser.add_argument(
        "--family",
        choices=["qlib158", "qlib360", "wq101", "all"],
        default="all",
        help="要注册的因子族（默认 all）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的因子文件（默认跳过）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要写入的因子数，不落盘",
    )
    args = parser.parse_args()

    specs = _collect_specs(args.family)
    print(f"[register] 将处理 {len(specs)} 个因子（family={args.family}）")

    stats = {"created": 0, "overwritten": 0, "skipped": 0, "dry-run": 0, "error": 0}
    errors: List[str] = []

    for spec in specs:
        try:
            status = _write_spec(spec, overwrite=args.overwrite, dry_run=args.dry_run)
            stats[status] = stats.get(status, 0) + 1
        except Exception as exc:
            stats["error"] += 1
            errors.append(f"{spec.get('factor_id', '?')}: {exc}")

    print("[register] 结果：")
    for k, v in stats.items():
        if v > 0:
            print(f"  {k:>12} = {v}")

    if errors:
        print("[register] 错误列表：")
        for msg in errors:
            print(f"  - {msg}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
