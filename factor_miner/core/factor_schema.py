"""
Unified factor schema skeleton.

This module defines the data contracts used by the next-generation
factor library services. The initial version focuses on stable typing
and legacy normalization helpers without taking over existing runtime
paths yet.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


DEFAULT_SOURCE_GROUP = "basic_kline"
KNOWN_COMPUTATION_TYPES = ("function", "formula", "algorithm_proxy")
LEGACY_COMPUTATION_TYPE_ALIASES = {
    "ml_model": "algorithm_proxy",
}


@dataclass
class FactorArtifacts:
    """References to executable or descriptive factor artifacts."""

    function_file: Optional[str] = None
    formula_file: Optional[str] = None
    formula_inline: Optional[str] = None
    algorithm_name: Optional[str] = None
    proxy_key: Optional[str] = None
    factor_name: Optional[str] = None
    entry_point: str = "calculate"
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy(cls, computation_data: Optional[Dict[str, Any]]) -> "FactorArtifacts":
        payload = computation_data or {}
        known_keys = {
            "function_file",
            "formula_file",
            "formula",
            "algorithm_name",
            "proxy_key",
            "factor_name",
            "entry_point",
        }
        extra = {k: v for k, v in payload.items() if k not in known_keys}
        return cls(
            function_file=payload.get("function_file"),
            formula_file=payload.get("formula_file"),
            formula_inline=payload.get("formula"),
            algorithm_name=payload.get("algorithm_name"),
            proxy_key=payload.get("proxy_key"),
            factor_name=payload.get("factor_name"),
            entry_point=payload.get("entry_point", "calculate"),
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        extra = payload.pop("extra", {}) or {}
        payload.update(extra)
        return {k: v for k, v in payload.items() if v not in (None, {}, [])}


@dataclass
class FactorTraits:
    """Non-structural traits used for filtering and execution hints."""

    is_event: bool = False
    is_mined: bool = False
    is_cross_sectional: bool = False
    is_window: bool = False
    requires_extra_data: bool = False
    realtime_supported: bool = False
    min_warmup_bars: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        extra = payload.pop("extra", {}) or {}
        payload.update(extra)
        return {k: v for k, v in payload.items() if v not in (None, {}, [])}


@dataclass
class FactorDefinition:
    """Canonical definition used by catalog, lifecycle and executor."""

    factor_id: str
    name: str
    description: str = ""
    source_group: str = DEFAULT_SOURCE_GROUP
    factor_kind: str = "technical"
    computation_type: str = "function"
    artifacts: FactorArtifacts = field(default_factory=FactorArtifacts)
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    traits: FactorTraits = field(default_factory=FactorTraits)
    tags: List[str] = field(default_factory=list)
    output_type: str = "series"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.computation_type in LEGACY_COMPUTATION_TYPE_ALIASES:
            self.computation_type = LEGACY_COMPUTATION_TYPE_ALIASES[self.computation_type]
        if self.computation_type not in KNOWN_COMPUTATION_TYPES:
            # Keep data readable during migration, but normalize unknown values later.
            self.metadata.setdefault("legacy_computation_type", self.computation_type)
        self.metadata.setdefault("created_at", datetime.now().isoformat())
        self.metadata.setdefault("schema_version", "1.0")

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "FactorDefinition":
        artifacts_raw = raw.get("artifacts") or {}
        traits_raw = raw.get("traits") or {}
        return cls(
            factor_id=raw["factor_id"],
            name=raw.get("name", raw["factor_id"]),
            description=raw.get("description", ""),
            source_group=raw.get("source_group", DEFAULT_SOURCE_GROUP),
            factor_kind=raw.get("factor_kind", "technical"),
            computation_type=raw.get("computation_type", "function"),
            artifacts=FactorArtifacts(
                function_file=artifacts_raw.get("function_file"),
                formula_file=artifacts_raw.get("formula_file"),
                formula_inline=artifacts_raw.get("formula_inline"),
                algorithm_name=artifacts_raw.get("algorithm_name"),
                proxy_key=artifacts_raw.get("proxy_key"),
                factor_name=artifacts_raw.get("factor_name"),
                entry_point=artifacts_raw.get("entry_point", "calculate"),
                extra={k: v for k, v in artifacts_raw.items() if k not in {
                    "function_file",
                    "formula_file",
                    "formula_inline",
                    "algorithm_name",
                    "proxy_key",
                    "factor_name",
                    "entry_point",
                }},
            ),
            parameters=raw.get("parameters") or {},
            dependencies=raw.get("dependencies") or [],
            traits=FactorTraits(
                is_event=traits_raw.get("is_event", False),
                is_mined=traits_raw.get("is_mined", False),
                is_cross_sectional=traits_raw.get("is_cross_sectional", False),
                is_window=traits_raw.get("is_window", False),
                requires_extra_data=traits_raw.get("requires_extra_data", False),
                realtime_supported=traits_raw.get("realtime_supported", False),
                min_warmup_bars=traits_raw.get("min_warmup_bars"),
                extra={k: v for k, v in traits_raw.items() if k not in {
                    "is_event",
                    "is_mined",
                    "is_cross_sectional",
                    "is_window",
                    "requires_extra_data",
                    "realtime_supported",
                    "min_warmup_bars",
                }},
            ),
            tags=raw.get("tags") or [],
            output_type=raw.get("output_type", "series"),
            metadata=raw.get("metadata") or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "name": self.name,
            "description": self.description,
            "source_group": self.source_group,
            "factor_kind": self.factor_kind,
            "computation_type": self.computation_type,
            "artifacts": self.artifacts.to_dict(),
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "traits": self.traits.to_dict(),
            "tags": self.tags,
            "output_type": self.output_type,
            "metadata": self.metadata,
        }


_AGGREGATION_METRIC_KEYS = (
    "ic_pearson",
    "ic_spearman",
    "icir",
    "win_rate",
    "sharpe_ratio",
    "long_short_return",
    "ic_positive_ratio",
)

_IC_KEY_MAP = {
    "ic_pearson": "ic_mean",
    "ic_spearman": "rank_ic_mean",
    "icir": "icir",
    "ic_positive_ratio": "ic_positive_ratio",
}

_RETURNS_KEYS = ("win_rate", "sharpe_ratio", "long_short_return")


@dataclass
class EvaluationAggregation:
    """Aggregated evaluation metrics across all evaluation records for a factor."""

    evaluated: bool = False
    eval_count: int = 0
    avg_metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    last_evaluated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FactorSummary:
    """Compact representation used by list and search APIs."""

    factor_id: str
    name: str
    source_group: str
    factor_kind: str
    computation_type: str
    traits: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    latest_evaluation_summary: Dict[str, Any] = field(default_factory=dict)
    evaluation_aggregation: EvaluationAggregation = field(default_factory=EvaluationAggregation)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FactorQuery:
    """Query parameters accepted by the future catalog service."""

    keyword: str = ""
    source_group: str = ""
    factor_kind: str = ""
    computation_type: str = ""
    include_tags: List[str] = field(default_factory=list)
    include_traits: Dict[str, Any] = field(default_factory=dict)
    limit: Optional[int] = None
    offset: int = 0


@dataclass
class SaveResult:
    success: bool
    factor_id: str
    message: str = ""
    written_files: List[str] = field(default_factory=list)


@dataclass
class DeleteResult:
    success: bool
    factor_id: str
    message: str = ""
    deleted_files: List[str] = field(default_factory=list)


@dataclass
class MigrationResult:
    success: bool
    factor_id: str
    message: str = ""
    before: Dict[str, Any] = field(default_factory=dict)
    after: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthIssue:
    factor_id: str
    issue_type: str
    detail: str = ""


@dataclass
class HealthReport:
    ok: bool = True
    total_factors: int = 0
    missing_definitions: List[str] = field(default_factory=list)
    missing_artifacts: List[str] = field(default_factory=list)
    orphan_files: List[str] = field(default_factory=list)
    index_ok: bool = True
    index_entry_count: int = 0
    issues: List[HealthIssue] = field(default_factory=list)


@dataclass
class CleanupResult:
    cleaned_count: int = 0
    cleaned_files: List[str] = field(default_factory=list)
    kept_files: List[str] = field(default_factory=list)
    orphan_files: List[str] = field(default_factory=list)


class LegacyDefinitionAdapter:
    """Adapter that normalizes existing on-disk definitions to the new schema."""

    def __init__(self, known_source_groups: Optional[List[str]] = None):
        self.known_source_groups = set(known_source_groups or [
            "basic_kline",
            "derivatives",
            "funding",
            "onchain",
        ])

    def adapt(self, raw: Dict[str, Any]) -> FactorDefinition:
        category = (raw.get("category") or "").strip()
        subcategory = (raw.get("subcategory") or "").strip()
        source_group = category if category in self.known_source_groups else DEFAULT_SOURCE_GROUP
        factor_kind = subcategory or (
            category if category and category not in self.known_source_groups else "technical"
        )
        data_requirement = (raw.get("data_requirement") or "").strip()
        traits = FactorTraits(
            is_event=subcategory == "event" or data_requirement == "event_factor",
            is_mined=subcategory == "mined" or data_requirement == "mined_factor",
            is_cross_sectional=False,
            is_window=bool((raw.get("metadata") or {}).get("is_window", False)),
            min_warmup_bars=(raw.get("metadata") or {}).get("min_warmup_bars"),
        )
        tags = []
        if subcategory:
            tags.append(subcategory)
        if category and category not in self.known_source_groups:
            tags.append(category)

        normalized = FactorDefinition(
            factor_id=raw["factor_id"],
            name=raw.get("name", raw["factor_id"]),
            description=raw.get("description", ""),
            source_group=source_group,
            factor_kind=factor_kind,
            computation_type=raw.get("computation_type", "function"),
            artifacts=FactorArtifacts.from_legacy(raw.get("computation_data") or {}),
            parameters=raw.get("parameters") or {},
            dependencies=raw.get("dependencies") or [],
            traits=traits,
            tags=tags,
            output_type=raw.get("output_type", "series"),
            metadata=raw.get("metadata") or {},
        )
        normalized.metadata.setdefault("legacy_fields", {})
        normalized.metadata["legacy_fields"].update({
            "category": raw.get("category"),
            "subcategory": raw.get("subcategory"),
            "data_requirement": raw.get("data_requirement"),
            "type": raw.get("type"),
        })
        return normalized


def normalize_definition(
    raw: Dict[str, Any],
    known_source_groups: Optional[List[str]] = None,
) -> FactorDefinition:
    """Normalize both new and legacy payloads to the canonical schema."""

    if "source_group" in raw and "factor_kind" in raw and "artifacts" in raw:
        return FactorDefinition.from_dict(raw)
    return LegacyDefinitionAdapter(known_source_groups=known_source_groups).adapt(raw)
