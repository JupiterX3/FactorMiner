"""
Unified factor executor.

Centralizes execution of function, formula and algorithm_proxy factors.
Migrated from factor_engine.py in phase 4.
"""

import importlib.util
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .factor_repository import FactorRepository
from .factor_schema import FactorDefinition

logger = logging.getLogger(__name__)

_module_cache: Dict[str, Any] = {}
_module_cache_lock = threading.Lock()
_sys_path_dirs: set = set()
_sys_path_lock = threading.Lock()
_alpha_ops_module_lock = threading.Lock()
_alpha_ops_loaded_mtime: Optional[float] = None


def _find_alpha_ops_file(storage_dir: Path) -> Optional[Path]:
    sd = Path(storage_dir)
    preferred = sd / "technicals" / "functions" / "_alpha_ops.py"
    if preferred.is_file():
        return preferred
    try:
        candidates = sorted(sd.glob("*/functions/_alpha_ops.py"))
    except OSError:
        candidates = []
    for p in candidates:
        if p.is_file():
            return p
    return None


def _ensure_alpha_ops_loaded(storage_dir: Path) -> None:
    global _alpha_ops_loaded_mtime
    ops_path = _find_alpha_ops_file(storage_dir)
    if ops_path is None:
        return
    try:
        mtime = ops_path.stat().st_mtime
    except OSError:
        return
    with _alpha_ops_module_lock:
        mod_existing = sys.modules.get("_alpha_ops")
        if mod_existing is not None and _alpha_ops_loaded_mtime == mtime:
            return
        spec = importlib.util.spec_from_file_location("_alpha_ops", ops_path)
        if spec is None or spec.loader is None:
            return
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_alpha_ops"] = mod
        spec.loader.exec_module(mod)
        _alpha_ops_loaded_mtime = mtime


def _ensure_sys_path(dir_path: str) -> None:
    with _sys_path_lock:
        if dir_path not in _sys_path_dirs:
            if dir_path not in sys.path:
                sys.path.insert(0, dir_path)
            _sys_path_dirs.add(dir_path)


def _load_module_cached(cache_key: str, func_path, module_name: str):
    mtime = os.path.getmtime(func_path) if os.path.exists(func_path) else None
    with _module_cache_lock:
        entry = _module_cache.get(cache_key)
        if entry is not None:
            cached_mtime, module = entry
            if mtime is not None and cached_mtime == mtime:
                return module, None
    spec = importlib.util.spec_from_file_location(module_name, func_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        return None, str(e)
    with _module_cache_lock:
        _module_cache[cache_key] = (mtime, module)
    return module, None


class FactorExecutor:
    """Unified execution entry for function, formula and algorithm_proxy factors."""

    def __init__(self, repository: Optional[FactorRepository] = None):
        self.repository = repository or FactorRepository()

    @property
    def storage_dir(self) -> Path:
        return self.repository.storage_dir

    def compute(self, factor_id: str, data: pd.DataFrame, **kwargs) -> Optional[pd.Series]:
        factor_def = self.repository.load_definition(factor_id)
        if factor_def is None:
            raise ValueError(f"factor not found: {factor_id}")
        return self.compute_definition(factor_def, data, **kwargs)

    def compute_definition(
        self,
        factor_def: FactorDefinition,
        data: pd.DataFrame,
        **kwargs,
    ) -> Optional[pd.Series]:
        ct = factor_def.computation_type
        if ct == "function":
            return self._compute_function(factor_def, data, **kwargs)
        if ct == "formula":
            return self._compute_formula(factor_def, data, **kwargs)
        if ct == "algorithm_proxy":
            return self._compute_algorithm_proxy(factor_def, data, **kwargs)
        raise ValueError(f"unsupported computation_type: {ct}")

    def compute_multiple(
        self, factor_ids: List[str], data: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        results = {}
        for fid in factor_ids:
            try:
                r = self.compute(fid, data, **kwargs)
                if r is not None:
                    results[fid] = r
            except Exception as e:
                logger.error(f"compute_multiple failed for {fid}: {e}")
        if not results:
            return pd.DataFrame()
        return pd.DataFrame(results)

    def _compute_function(
        self,
        factor_def: FactorDefinition,
        data: pd.DataFrame,
        **kwargs,
    ) -> Optional[pd.Series]:
        function_file = factor_def.artifacts.function_file
        if not function_file:
            logger.error(f"factor {factor_def.factor_id} missing function_file")
            return None

        func_path = self._resolve_artifact_path(function_file)
        if func_path is None:
            logger.error(f"function file not found: {function_file}")
            return None

        _ensure_alpha_ops_loaded(self.storage_dir)
        _ensure_sys_path(str(self.storage_dir.parent))
        _ensure_sys_path(str(func_path.parent))

        cache_key = str(func_path)
        module, err = _load_module_cached(cache_key, func_path, f"factor_{factor_def.factor_id}")
        if err:
            logger.error(f"load function module failed: {err}")
            return None

        entry_point = factor_def.artifacts.entry_point or "calculate"
        if not hasattr(module, entry_point):
            logger.error(f"entry point not found: {entry_point}")
            return None

        func = getattr(module, entry_point)
        params = dict(factor_def.parameters)
        params.update(kwargs)

        try:
            result = func(data, **params)
            if result is not None:
                return result
            logger.warning(f"factor {factor_def.factor_id} returned None")
            return None
        except Exception as e:
            logger.error(f"call factor function failed: {e}")
            return None

    def _compute_formula(
        self,
        factor_def: FactorDefinition,
        data: pd.DataFrame,
        **kwargs,
    ) -> Optional[pd.Series]:
        formula = factor_def.artifacts.formula_inline
        if not formula:
            formula_file = factor_def.artifacts.formula_file
            if formula_file:
                fpath = self._resolve_artifact_path(formula_file)
                if fpath is not None:
                    try:
                        formula = fpath.read_text(encoding="utf-8")
                    except Exception:
                        formula = None
        if not formula:
            logger.error(f"factor {factor_def.factor_id} missing formula")
            return None

        if formula.strip().startswith("#"):
            return self._compute_formula_as_function(factor_def, data, **kwargs)

        params = dict(factor_def.parameters)
        params.update(kwargs)

        try:
            local_vars = {
                "close": data["close"],
                "open": data["open"],
                "high": data["high"],
                "low": data["low"],
                "volume": data["volume"],
                "data": data,
                "pd": pd,
                "np": np,
                "abs": abs,
                "min": min,
                "max": max,
                "sum": sum,
                "len": len,
                "round": round,
                "float": float,
                "int": int,
                "str": str,
            }
            local_vars.update(params)
            result = eval(formula, {"__builtins__": {}}, local_vars)
            if isinstance(result, pd.Series):
                return result
            logger.warning(f"formula returned non-Series: {type(result)}")
            return None
        except Exception as e:
            logger.error(f"formula eval failed {factor_def.factor_id}: {e}")
            return None

    def _compute_formula_as_function(
        self,
        factor_def: FactorDefinition,
        data: pd.DataFrame,
        **kwargs,
    ) -> Optional[pd.Series]:
        factor_id = factor_def.factor_id
        base_name = factor_id.split("_")[0]
        name_without_suffix = "_".join(factor_id.split("_")[:-1]) if "_" in factor_id else factor_id

        candidates = []
        for group in self.repository.list_source_groups():
            candidates.append(self.storage_dir / group / "functions" / f"{factor_id}.py")
            candidates.append(self.storage_dir / group / "functions" / f"{name_without_suffix}.py")
            candidates.append(self.storage_dir / group / "functions" / f"{base_name}.py")
        candidates.extend([
            self.storage_dir / "technicals" / "functions" / f"{factor_id}.py",
            self.storage_dir / "minactors" / "functions" / f"{factor_id}.py",
            self.storage_dir / "technicals" / "functions" / f"{name_without_suffix}.py",
            self.storage_dir / "technicals" / "functions" / f"{base_name}.py",
        ])

        func_path = None
        for p in candidates:
            if p.exists():
                func_path = p
                break

        if func_path is None:
            logger.warning(f"no function file for formula factor {factor_id}")
            return None

        _ensure_alpha_ops_loaded(self.storage_dir)
        _ensure_sys_path(str(self.storage_dir.parent))

        cache_key = str(func_path)
        module, err = _load_module_cached(cache_key, func_path, f"factor_{factor_id}")
        if err:
            logger.error(f"load function module failed: {err}")
            return None

        if not hasattr(module, "calculate"):
            logger.error(f"function missing calculate entry point")
            return None

        params = dict(factor_def.parameters)
        params.update(kwargs)

        try:
            result = module.calculate(data, **params)
            if result is not None:
                return result
            logger.warning(f"formula-as-function returned None: {factor_id}")
            return None
        except Exception as e:
            logger.error(f"call formula-as-function failed: {e}")
            return None

    def _compute_algorithm_proxy(
        self,
        factor_def: FactorDefinition,
        data: pd.DataFrame,
        **kwargs,
    ) -> Optional[pd.Series]:
        algorithm_name = factor_def.artifacts.algorithm_name
        if not algorithm_name:
            logger.error(f"factor {factor_def.factor_id} missing algorithm_name")
            return None

        algo_module = self._load_algorithm_module(algorithm_name)
        if algo_module is None:
            return None

        if not hasattr(algo_module, "calculate_single_factor"):
            logger.error(f"algorithm module {algorithm_name} missing calculate_single_factor")
            return None

        return algo_module.calculate_single_factor(
            data,
            factor_def.name,
            factor_id=factor_def.factor_id,
            computation_data=factor_def.artifacts.extra,
            **kwargs,
        )

    def _load_algorithm_module(self, algorithm_name: str):
        algo_dir = Path(__file__).parent.parent.parent / "user_algo"
        _ensure_sys_path(str(algo_dir))
        algo_file = algo_dir / f"{algorithm_name}.py"
        if not algo_file.exists():
            logger.error(f"algorithm file not found: {algo_file}")
            return None
        cache_key = str(algo_file)
        module, err = _load_module_cached(cache_key, algo_file, algorithm_name)
        if err:
            logger.error(f"load algorithm module failed {algorithm_name}: {err}")
            return None
        return module

    def _resolve_artifact_path(self, relative_path: str) -> Optional[Path]:
        candidate = self.storage_dir / relative_path
        if candidate.exists():
            return candidate
        fname = Path(relative_path).name
        for group in self.repository.list_source_groups():
            for subdir in ("functions", "formulas"):
                c = self.storage_dir / group / subdir / fname
                if c.exists():
                    return c
        for legacy in ("technicals", "minactors"):
            c = self.storage_dir / legacy / relative_path
            if c.exists():
                return c
            c = self.storage_dir / legacy / "functions" / fname
            if c.exists():
                return c
        return None
