"""
透明因子保存系统 v4.0
因子分类采用「一级=数据来源，二级=用途」的两层结构。

一级分类（对应物理目录，动态扫描 factorlib/ 下带 definitions/ 子目录的文件夹）：
- basic_kline/   仅依赖 OHLCV（最大的一档）
- derivatives/   依赖衍生品微观结构（OI/LSR/taker_buy/basis 等）
- funding/       依赖资金费率

二级分类（写进 JSON 的 subcategory 字段）：
- technical / mined / event / statistical / microstructure / funding_carry 等

因子类型（computation_type）：
- function  函数类型：保存在 {root}/functions/
- formula   纯公式类型：保存在 {root}/formulas/ 或直接嵌在 JSON 里

不再支持 ml_model（v4 已移除，仅保留兼容 function/formula）。
每个一级目录的结构统一为：
    <root>/definitions/     因子定义 JSON
    <root>/functions/       因子函数源码
    <root>/evaluations/     评估结果
    <root>/mining_history/  （可选）挖掘历史，仅 basic_kline 下有 minactors 历史
"""

import json
import logging
import hashlib
import warnings
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import importlib.util
import pandas as pd  # noqa: F401 - 历史接口签名保留

from .factor_schema import FactorDefinition as UnifiedFactorDefinition
from .factor_schema import FactorArtifacts, FactorTraits, normalize_definition

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 一级分类枚举（仅做校验与 fallback 用；物理上是动态扫描）
# -----------------------------------------------------------------------------
KNOWN_SOURCE_GROUPS = ("basic_kline", "derivatives", "funding", "onchain")


@dataclass
class FactorDefinition:
    """完整的因子定义 - 包含所有计算信息。"""

    factor_id: str
    name: str
    description: str
    category: str  # 一级分类，对应目录名（basic_kline / derivatives / funding / ...）
    subcategory: str = ""  # 二级分类：technical / mined / event / statistical / ...

    # v4：只保留 function / formula，ml_model 已移除
    computation_type: str = "function"
    computation_data: Dict = None

    parameters: Dict = None
    dependencies: List = None
    output_type: str = "series"
    metadata: Dict = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}
        if self.computation_data is None:
            self.computation_data = {}

        # v4：computation_type="ml_model" 保留作为「算法代理」语义
        # （由 user_algo/ 下的 `calculate_single_factor` 函数承担实际计算，
        # 典型用途是 GP 截面因子 `gp_cross_sectional`）。
        # 真正的 ML 预训练因子（依赖 pkl 模型推理）已在 v4 中全部移除。

        self.metadata['checksum'] = self._calculate_checksum()
        self.metadata.setdefault('created_at', datetime.now().isoformat())

    def _calculate_checksum(self) -> str:
        content = f"{self.factor_id}_{self.name}_{str(self.computation_data)}"
        return hashlib.md5(content.encode()).hexdigest()[:8]

    def to_dict(self) -> Dict:
        return asdict(self)


class TransparentFactorStorage:
    """
    完全透明的因子存储管理器（v4：按数据来源做一级分类）。

    定义 / 函数 / 评估都存储在每个「数据来源一级目录」下，目录结构统一。
    本类不硬编码任何一级目录名，而是通过扫描发现——任何子目录只要
    含有 definitions/ 子目录就会被识别为一级分类。
    """

    # 仅作为首次部署时的种子目录（不存在时自动创建）
    DEFAULT_SOURCE_GROUPS = KNOWN_SOURCE_GROUPS

    def __init__(self, storage_dir: str = None):
        if storage_dir is None:
            storage_dir = Path(__file__).parent.parent.parent / "factorlib"
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # 种子：保证默认一级分类目录存在
        for group in self.DEFAULT_SOURCE_GROUPS:
            for sub in ("definitions", "functions", "evaluations"):
                (self.storage_dir / group / sub).mkdir(parents=True, exist_ok=True)

        # 其它目录
        self.temp_dir = self.storage_dir / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # 兼容期：挖掘历史仍写到 basic_kline/mining_history
        self.mining_history_dir = self.storage_dir / "basic_kline" / "mining_history"
        self.mining_history_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 目录扫描
    # ------------------------------------------------------------------
    # 目录名中包含这些关键字的被视为备份/归档，不参与因子扫描
    _EXCLUDED_NAME_TOKENS = ("_archived_", "_deprecated", "_backup")

    def list_source_groups(self) -> List[str]:
        """
        返回所有一级分类目录（含 definitions/ 子目录，且名字不是归档备份）。
        """
        groups: List[str] = []
        if not self.storage_dir.exists():
            return groups
        for child in self.storage_dir.iterdir():
            if not child.is_dir():
                continue
            if child.name.startswith((".", "_")):
                continue
            if any(tok in child.name for tok in self._EXCLUDED_NAME_TOKENS):
                continue
            if (child / "definitions").is_dir():
                groups.append(child.name)
        # 稳定顺序：DEFAULT_SOURCE_GROUPS 在前，其余字母序
        front = [g for g in self.DEFAULT_SOURCE_GROUPS if g in groups]
        rest = sorted(g for g in groups if g not in self.DEFAULT_SOURCE_GROUPS)
        return front + rest

    def _definitions_dirs(self) -> List[Path]:
        return [self.storage_dir / g / "definitions" for g in self.list_source_groups()]

    def _functions_dirs(self) -> List[Path]:
        return [self.storage_dir / g / "functions" for g in self.list_source_groups()]

    def _evaluations_dirs(self) -> List[Path]:
        return [self.storage_dir / g / "evaluations" for g in self.list_source_groups()]

    def _group_for_category(self, category: str) -> str:
        """根据 category（一级分类）定位目录名；未知分类回退到 basic_kline。"""
        cat = (category or "").strip().lower()
        if cat in self.list_source_groups():
            return cat
        # 兼容老数据：历史上的 technical/ml/pattern/mined_factor 等 → basic_kline
        return "basic_kline"

    # ------------------------------------------------------------------
    # 计算入口（兼容包装层，代理到 FactorExecutor）
    # ------------------------------------------------------------------
    def compute_factor(self, factor_id: str, data: 'pd.DataFrame', **kwargs) -> Optional['pd.Series']:
        warnings.warn(
            "TransparentFactorStorage.compute_factor() is deprecated, use FactorExecutor.compute()",
            DeprecationWarning,
            stacklevel=2,
        )
        from .factor_executor import FactorExecutor
        executor = FactorExecutor()
        try:
            return executor.compute(factor_id, data, **kwargs)
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"计算因子失败 {factor_id}: {e}")
            return None

    # ------------------------------------------------------------------
    # 定义读写
    # ------------------------------------------------------------------
    def _load_factor_definition_dict(self, factor_id: str) -> Optional[Dict]:
        for d in self._definitions_dirs():
            def_file = d / f"{factor_id}.json"
            if def_file.exists():
                with open(def_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        return None

    def load_factor_definition(self, factor_id: str) -> Optional["UnifiedFactorDefinition"]:
        try:
            data = self._load_factor_definition_dict(factor_id)
            if data is not None:
                return normalize_definition(data, known_source_groups=self.list_source_groups())
            return None
        except Exception as e:
            logger.error(f"加载因子定义失败: {e}")
            return None

    def load_normalized_factor_definition(
        self, factor_id: str
    ) -> Optional[UnifiedFactorDefinition]:
        """
        加载统一 schema 的因子定义（第一阶段接入点）。

        注意：
        - `load_factor_definition()` 继续返回历史的 storage 层定义对象，避免影响旧链路；
        - 新代码可逐步迁移到本方法，拿到 `factor_schema.py` 中的统一定义。
        """
        try:
            data = self._load_factor_definition_dict(factor_id)
            if data is None:
                return None
            return normalize_definition(
                data,
                known_source_groups=self.list_source_groups(),
            )
        except Exception as e:
            logger.error(f"加载统一因子定义失败: {e}")
            return None

    def list_factors(self) -> List[str]:
        ids: List[str] = []
        try:
            for d in self._definitions_dirs():
                if d.exists():
                    ids.extend(f.stem for f in d.glob("*.json"))
            # 去重但保留顺序
            seen = set()
            ordered = []
            for i in ids:
                if i in seen:
                    continue
                seen.add(i)
                ordered.append(i)
            return ordered
        except Exception as e:
            logger.error(f"列出因子失败: {e}")
            return []

    def get_factors_by_category(self, category: str) -> List[str]:
        factors: List[str] = []
        for factor_id in self.list_factors():
            factor_def = self.load_factor_definition(factor_id)
            if factor_def and factor_def.category == category:
                factors.append(factor_id)
        return factors

    def delete_factor(self, factor_id: str) -> bool:
        warnings.warn(
            "TransparentFactorStorage.delete_factor() is deprecated, use FactorLifecycleService.delete_factor()",
            DeprecationWarning,
            stacklevel=2,
        )
        lifecycle = self._get_lifecycle()
        result = lifecycle.delete_factor(factor_id, cascade=True)
        return result.success

    def _save_factor_definition(self, factor_def: FactorDefinition) -> bool:
        try:
            group = self._group_for_category(factor_def.category)
            def_dir = self.storage_dir / group / "definitions"
            def_dir.mkdir(parents=True, exist_ok=True)
            def_file = def_dir / f"{factor_def.factor_id}.json"
            with open(def_file, 'w', encoding='utf-8') as f:
                json.dump(factor_def.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info(f"因子定义已保存: {factor_def.factor_id} → {group}")
            return True
        except Exception as e:
            logger.error(f"保存因子定义失败: {e}")
            return False

    # ------------------------------------------------------------------
    # 便捷 API（兼容包装层，内部代理到 FactorLifecycleService）
    # ------------------------------------------------------------------
    def _get_lifecycle(self):
        from .factor_lifecycle import FactorLifecycleService
        return FactorLifecycleService()

    def save_function_factor(
        self,
        factor_id: str,
        name: str,
        function_code: str,
        description: str = "",
        category: str = "basic_kline",
        subcategory: str = "technical",
        entry_point: str = "calculate",
        imports: List[str] = None,
        parameters: Dict = None,
    ) -> bool:
        warnings.warn(
            "TransparentFactorStorage.save_function_factor() is deprecated, use FactorLifecycleService.save_factor()",
            DeprecationWarning,
            stacklevel=2,
        )
        group = self._group_for_category(category)
        func_rel_path = f"{group}/functions/{factor_id}.py"

        unified_def = UnifiedFactorDefinition(
            factor_id=factor_id,
            name=name,
            description=description,
            source_group=group,
            factor_kind=subcategory or "technical",
            computation_type="function",
            artifacts=FactorArtifacts(
                function_file=func_rel_path,
                entry_point=entry_point,
                extra={"function_code": function_code, "imports": imports or []},
            ),
            parameters=parameters or {},
            traits=FactorTraits(is_mined=subcategory == "mined"),
        )
        artifacts_map = {func_rel_path: function_code}
        result = self._get_lifecycle().save_factor(
            unified_def, artifacts=artifacts_map, overwrite=True, validate=True
        )
        return result.success

    def save_technical_factor(self, *args, **kwargs) -> bool:
        warnings.warn(
            "TransparentFactorStorage.save_technical_factor() is deprecated, use FactorLifecycleService.save_factor()",
            DeprecationWarning,
            stacklevel=2,
        )
        kwargs.setdefault("category", "basic_kline")
        kwargs.setdefault("subcategory", "technical")
        return self.save_function_factor(*args, **kwargs)

    def save_minactor_factor(
        self, factor_id: str, name: str,
        function_code: str = "",
        description: str = "",
        subcategory: str = "mined",
        entry_point: str = "calculate",
        imports: List[str] = None,
        parameters: Dict = None,
        algorithm_name: str = None,
        category: str = "basic_kline",
        performance_metrics: Dict = None,
        **extra_kwargs,
    ) -> bool:
        warnings.warn(
            "TransparentFactorStorage.save_minactor_factor() is deprecated, use FactorLifecycleService.save_factor()",
            DeprecationWarning,
            stacklevel=2,
        )
        group = self._group_for_category(category)
        is_proxy = bool(algorithm_name) and not function_code

        if is_proxy:
            unified_def = UnifiedFactorDefinition(
                factor_id=factor_id,
                name=name,
                description=description,
                source_group=group,
                factor_kind=subcategory or "mined",
                computation_type="algorithm_proxy",
                artifacts=FactorArtifacts(
                    algorithm_name=algorithm_name,
                    proxy_key=algorithm_name,
                    entry_point=entry_point,
                ),
                parameters=parameters or {},
                traits=FactorTraits(is_mined=True),
                metadata=performance_metrics or {},
            )
            result = self._get_lifecycle().save_factor(
                unified_def, overwrite=True, validate=True
            )
        else:
            func_rel_path = f"{group}/functions/{factor_id}.py"
            unified_def = UnifiedFactorDefinition(
                factor_id=factor_id,
                name=name,
                description=description,
                source_group=group,
                factor_kind=subcategory or "mined",
                computation_type="function",
                artifacts=FactorArtifacts(
                    function_file=func_rel_path,
                    entry_point=entry_point,
                    extra={"function_code": function_code, "imports": imports or []},
                ),
                parameters=parameters or {},
                traits=FactorTraits(is_mined=True),
                metadata=performance_metrics or {},
            )
            artifacts_map = {func_rel_path: function_code} if function_code else {}
            result = self._get_lifecycle().save_factor(
                unified_def, artifacts=artifacts_map, overwrite=True, validate=True
            )
        return result.success

    # ------------------------------------------------------------------
    # 评估与挖掘历史
    # ------------------------------------------------------------------
    def save_evaluation(
        self, factor_id: str, evaluation_data: Dict, source: str = None
    ) -> bool:
        warnings.warn(
            "TransparentFactorStorage.save_evaluation() is deprecated, use FactorRepository.save_evaluations()",
            DeprecationWarning,
            stacklevel=2,
        )
        from .factor_catalog import FactorCatalogService
        from .factor_repository import FactorRepository

        try:
            repo = FactorRepository()
            catalog = FactorCatalogService(repo)
            factor_def = catalog.get_factor(factor_id)
            source_group = factor_def.source_group if factor_def else "basic_kline"

            payload = repo.load_evaluations(factor_id)
            evaluations = payload.get("evaluations", [])
            evaluations.append(
                {
                    "evaluated_at": datetime.now().isoformat(),
                    "results": evaluation_data,
                }
            )
            payload["evaluations"] = evaluations
            repo.save_evaluations(factor_id, payload, source_group)

            catalog.invalidate_index_cache()
            catalog.update_index_entry(factor_id)
            return True
        except Exception as e:
            logger.error(f"保存评估失败: {e}")
            return False

    def save_mining_history(self, session_id: str, session_data: Dict) -> bool:
        try:
            self.mining_history_dir.mkdir(parents=True, exist_ok=True)
            sessions_file = self.mining_history_dir / "mining_sessions.json"
            sessions: Dict = {}
            if sessions_file.exists():
                with open(sessions_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        sessions = json.loads(content)
            sessions[session_id] = session_data
            with open(sessions_file, 'w', encoding='utf-8') as f:
                json.dump(sessions, f, ensure_ascii=False, indent=2)

            result_file = self.mining_history_dir / f"mining_results_{session_id}.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"保存挖掘历史失败: {e}")
            return False


_global_storage: Optional[TransparentFactorStorage] = None


def get_global_storage() -> TransparentFactorStorage:
    global _global_storage
    if _global_storage is None:
        _global_storage = TransparentFactorStorage()
    return _global_storage
