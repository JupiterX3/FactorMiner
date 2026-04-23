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
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import importlib.util
import pandas as pd  # noqa: F401 - 历史接口签名保留

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# 一级分类枚举（仅做校验与 fallback 用；物理上是动态扫描）
# -----------------------------------------------------------------------------
KNOWN_SOURCE_GROUPS = ("basic_kline", "derivatives", "funding")


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
    # 计算入口
    # ------------------------------------------------------------------
    def compute_factor(self, factor_id: str, data: 'pd.DataFrame', **kwargs) -> Optional['pd.Series']:
        factor_def = self.load_factor_definition(factor_id)
        if not factor_def:
            raise ValueError(f"因子不存在: {factor_id}")

        params = factor_def.parameters.copy()
        params.update(kwargs)

        try:
            if factor_def.computation_type == "function":
                return self._compute_function_factor(factor_def, data, params)
            elif factor_def.computation_type in ("formula", "ml_model"):
                # formula 与 ml_model（算法代理）统一由 factor_engine 处理
                raise NotImplementedError(
                    f"computation_type={factor_def.computation_type} "
                    f"由 factor_engine 处理，不应在 storage 层计算"
                )
            else:
                raise ValueError(
                    f"不支持的计算类型: {factor_def.computation_type}；"
                    f"v4 支持 function / formula / ml_model(算法代理)"
                )
        except Exception as e:
            logger.error(f"计算因子失败 {factor_id}: {e}")
            return None

    def _compute_function_factor(
        self, factor_def: FactorDefinition, data: 'pd.DataFrame', params: Dict
    ) -> 'pd.Series':
        comp_data = factor_def.computation_data
        func_rel = comp_data.get("function_file")
        if not func_rel:
            raise ValueError(f"因子 {factor_def.factor_id} 缺少 function_file")

        # 允许绝对路径或相对 storage_dir；找不到时遍历所有一级目录 functions/
        func_file = self.storage_dir / func_rel
        if not func_file.exists():
            fname = Path(func_rel).name
            for fdir in self._functions_dirs():
                candidate = fdir / fname
                if candidate.exists():
                    func_file = candidate
                    break
        if not func_file.exists():
            raise FileNotFoundError(f"找不到因子函数文件: {func_rel}")

        entry_point = comp_data.get("entry_point", "calculate")
        spec = importlib.util.spec_from_file_location(
            f"factor_{factor_def.factor_id}", func_file
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, entry_point):
            raise ValueError(f"函数中未找到入口点: {entry_point}")
        func = getattr(module, entry_point)
        return func(data, **params)

    # ------------------------------------------------------------------
    # 定义读写
    # ------------------------------------------------------------------
    def load_factor_definition(self, factor_id: str) -> Optional[FactorDefinition]:
        try:
            for d in self._definitions_dirs():
                def_file = d / f"{factor_id}.json"
                if def_file.exists():
                    with open(def_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    return FactorDefinition(**data)
            return None
        except Exception as e:
            logger.error(f"加载因子定义失败: {e}")
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
        try:
            removed = False
            for d in self._definitions_dirs():
                def_file = d / f"{factor_id}.json"
                if def_file.exists():
                    def_file.unlink()
                    removed = True
            for d in self._functions_dirs():
                fn_file = d / f"{factor_id}.py"
                if fn_file.exists():
                    fn_file.unlink()
                    removed = True
            return removed
        except Exception as e:
            logger.error(f"删除因子失败: {e}")
            return False

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
    # 便捷 API
    # ------------------------------------------------------------------
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
        """
        保存函数类因子；category 应为一级分类目录名（basic_kline/derivatives/funding）。
        """
        try:
            group = self._group_for_category(category)
            functions_dir = self.storage_dir / group / "functions"
            functions_dir.mkdir(parents=True, exist_ok=True)

            func_file = functions_dir / f"{factor_id}.py"
            with open(func_file, 'w', encoding='utf-8') as f:
                if imports:
                    for imp in imports:
                        f.write(f"{imp}\n")
                    f.write("\n")
                f.write(function_code)

            factor_def = FactorDefinition(
                factor_id=factor_id,
                name=name,
                description=description,
                category=group,
                subcategory=subcategory,
                computation_type="function",
                computation_data={
                    "function_file": str(func_file.relative_to(self.storage_dir)),
                    "function_code": function_code,
                    "entry_point": entry_point,
                    "imports": imports or [],
                },
                parameters=parameters or {},
            )
            return self._save_factor_definition(factor_def)
        except Exception as e:
            logger.error(f"保存因子失败: {e}")
            return False

    # 兼容老名字：save_technical_factor 现在就是 save_function_factor
    def save_technical_factor(self, *args, **kwargs) -> bool:
        kwargs.setdefault("category", "basic_kline")
        kwargs.setdefault("subcategory", "technical")
        return self.save_function_factor(*args, **kwargs)

    # 兼容老名字：挖掘因子
    def save_minactor_factor(
        self, factor_id: str, name: str,
        function_code: str,
        description: str = "",
        subcategory: str = "mined",
        entry_point: str = "calculate",
        imports: List[str] = None,
        parameters: Dict = None,
    ) -> bool:
        return self.save_function_factor(
            factor_id=factor_id,
            name=name,
            function_code=function_code,
            description=description,
            category="basic_kline",
            subcategory=subcategory,
            entry_point=entry_point,
            imports=imports,
            parameters=parameters,
        )

    # ------------------------------------------------------------------
    # 评估与挖掘历史
    # ------------------------------------------------------------------
    def save_evaluation(
        self, factor_id: str, evaluation_data: Dict, source: str = None
    ) -> bool:
        """
        保存评估结果到因子所在一级分类的 evaluations/ 目录。

        `source` 参数兼容旧实现（historically: 'technicals'/'minactors'），
        但现在以 factor 的实际分类目录为准。
        """
        try:
            factor_def = self.load_factor_definition(factor_id)
            group = self._group_for_category(
                factor_def.category if factor_def else ""
            )
            eval_dir = self.storage_dir / group / "evaluations"
            eval_dir.mkdir(parents=True, exist_ok=True)
            eval_file = eval_dir / f"{factor_id}.json"

            existing = {}
            if eval_file.exists():
                with open(eval_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            evaluations = existing.get('evaluations', [])
            evaluations.append(
                {
                    'evaluated_at': datetime.now().isoformat(),
                    'results': evaluation_data,
                }
            )
            existing['evaluations'] = evaluations

            with open(eval_file, 'w', encoding='utf-8') as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
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
