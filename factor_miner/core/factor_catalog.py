"""
Unified factor catalog skeleton.

This service will become the single read path for factor listing,
searching, summary generation and index maintenance.
"""

import math
from datetime import datetime
from typing import Dict, List, Optional

from .factor_repository import FactorRepository
from .factor_schema import (
    EvaluationAggregation,
    FactorDefinition,
    FactorQuery,
    FactorSummary,
    _AGGREGATION_METRIC_KEYS,
    _IC_KEY_MAP,
    _RETURNS_KEYS,
)


class FactorCatalogService:
    """Catalog facade for normalized factor discovery and summaries."""

    def __init__(self, repository: Optional[FactorRepository] = None):
        self.repository = repository or FactorRepository()
        self._index_cache: Optional[Dict] = None

    def list_factors(
        self, query: Optional[FactorQuery] = None, use_index: bool = True
    ) -> List[FactorSummary]:
        query = query or FactorQuery()
        if use_index:
            summaries = self._list_from_index(query)
            if summaries is not None:
                return summaries
        return self._list_from_disk(query)

    def _list_from_index(self, query: FactorQuery) -> Optional[List[FactorSummary]]:
        index_data = self._get_index()
        if index_data is None:
            return None
        raw_factors = index_data.get("factors") or []
        summaries: List[FactorSummary] = []
        for raw in raw_factors:
            summary = self._summary_from_index_entry(raw)
            if summary is not None and self._matches(summary, query):
                summaries.append(summary)
        if query.offset:
            summaries = summaries[query.offset:]
        if query.limit is not None:
            summaries = summaries[: query.limit]
        return summaries

    def _list_from_disk(self, query: FactorQuery) -> List[FactorSummary]:
        summaries: List[FactorSummary] = []
        for definition_file in self.repository.iter_definition_files():
            raw = self.repository.load_json_file(definition_file)
            factor_def = FactorDefinition.from_dict(raw) if "source_group" in raw else None
            if factor_def is None:
                factor_def = self.repository.load_definition(definition_file.stem)
            if factor_def is None:
                continue
            summary = self.build_summary(factor_def)
            if not self._matches(summary, query):
                continue
            summaries.append(summary)

        if query.offset:
            summaries = summaries[query.offset:]
        if query.limit is not None:
            summaries = summaries[: query.limit]
        return summaries

    def _get_index(self) -> Optional[Dict]:
        if self._index_cache is not None:
            return self._index_cache
        index_file = self.repository.index_file()
        if not index_file.exists():
            return None
        try:
            self._index_cache = self.repository.load_json_file(index_file)
            return self._index_cache
        except Exception:
            return None

    def invalidate_index_cache(self) -> None:
        self._index_cache = None

    def _summary_from_index_entry(self, raw: Dict) -> Optional[FactorSummary]:
        if not raw or "factor_id" not in raw:
            return None
        agg_raw = raw.get("evaluation_aggregation") or {}
        aggregation = EvaluationAggregation(
            evaluated=agg_raw.get("evaluated", False),
            eval_count=agg_raw.get("eval_count", 0),
            avg_metrics=agg_raw.get("avg_metrics") or {},
            last_evaluated_at=agg_raw.get("last_evaluated_at"),
        )
        return FactorSummary(
            factor_id=raw.get("factor_id", ""),
            name=raw.get("name", ""),
            source_group=raw.get("source_group", ""),
            factor_kind=raw.get("factor_kind", ""),
            computation_type=raw.get("computation_type", ""),
            traits=raw.get("traits") or {},
            tags=raw.get("tags") or [],
            latest_evaluation_summary=raw.get("latest_evaluation_summary") or {},
            evaluation_aggregation=aggregation,
            metadata=raw.get("metadata") or {},
        )

    def get_factor(self, factor_id: str) -> Optional[FactorDefinition]:
        return self.repository.load_definition(factor_id)

    def get_summary(self, factor_id: str) -> Optional[FactorSummary]:
        factor_def = self.get_factor(factor_id)
        if factor_def is None:
            return None
        return self.build_summary(factor_def)

    def get_stats(self) -> Dict:
        stats = {
            "total_factors": 0,
            "source_groups": {},
            "factor_kinds": {},
            "computation_types": {},
        }
        for summary in self.list_factors():
            stats["total_factors"] += 1
            stats["source_groups"][summary.source_group] = stats["source_groups"].get(summary.source_group, 0) + 1
            stats["factor_kinds"][summary.factor_kind] = stats["factor_kinds"].get(summary.factor_kind, 0) + 1
            stats["computation_types"][summary.computation_type] = (
                stats["computation_types"].get(summary.computation_type, 0) + 1
            )
        return stats

    def rebuild_index(self) -> Dict:
        factors = []
        for summary in self.list_factors(use_index=False):
            factors.append(summary.to_dict())
        payload = {
            "schema_version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "source_groups": self.repository.list_source_groups(),
            "factors": factors,
        }
        self.repository.save_json_file(self.repository.index_file(), payload)
        self._index_cache = payload
        return payload

    def update_index_entry(self, factor_id: str) -> None:
        summary = self.get_summary(factor_id)
        if summary is None:
            return
        index_data = self._get_index()
        if index_data is None:
            self.rebuild_index()
            return
        factors = index_data.get("factors") or []
        updated = False
        for i, entry in enumerate(factors):
            if entry.get("factor_id") == factor_id:
                factors[i] = summary.to_dict()
                updated = True
                break
        if not updated:
            factors.append(summary.to_dict())
        index_data["factors"] = factors
        index_data["generated_at"] = datetime.now().isoformat()
        self.repository.save_json_file(self.repository.index_file(), index_data)
        self._index_cache = index_data

    def load_index(self) -> Dict:
        index_file = self.repository.index_file()
        if not index_file.exists():
            return self.rebuild_index()
        return self.repository.load_json_file(index_file)

    def build_summary(self, factor_def: FactorDefinition) -> FactorSummary:
        evaluations = self.repository.load_evaluations(factor_def.factor_id)
        latest_summary = self._extract_latest_evaluation_summary(evaluations)
        aggregation = self.aggregate_evaluations(factor_def.factor_id, evaluations)
        return FactorSummary(
            factor_id=factor_def.factor_id,
            name=factor_def.name,
            source_group=factor_def.source_group,
            factor_kind=factor_def.factor_kind,
            computation_type=factor_def.computation_type,
            traits=factor_def.traits.to_dict(),
            tags=factor_def.tags,
            latest_evaluation_summary=latest_summary,
            evaluation_aggregation=aggregation,
            metadata=factor_def.metadata,
        )

    def aggregate_evaluations(
        self, factor_id: str, evaluations_payload: Optional[Dict] = None
    ) -> EvaluationAggregation:
        if evaluations_payload is None:
            evaluations_payload = self.repository.load_evaluations(factor_id)

        evaluations = (evaluations_payload or {}).get("evaluations") or []
        eval_count = len(evaluations)
        if eval_count == 0:
            return EvaluationAggregation(
                evaluated=False, eval_count=0, avg_metrics={}, last_evaluated_at=None
            )

        sums: Dict[str, float] = {k: 0.0 for k in _AGGREGATION_METRIC_KEYS}
        counts: Dict[str, int] = {k: 0 for k in _AGGREGATION_METRIC_KEYS}
        last_evaluated_at: Optional[str] = None

        for ev in evaluations:
            if not isinstance(ev, dict):
                continue
            results = ev.get("results") or {}
            if not isinstance(results, dict):
                results = {}
            for k in _AGGREGATION_METRIC_KEYS:
                v = self._extract_metric_value(k, results)
                if v is not None:
                    sums[k] += float(v)
                    counts[k] += 1
            ev_ts = ev.get("evaluated_at")
            if ev_ts:
                last_evaluated_at = ev_ts

        avg_metrics: Dict[str, Optional[float]] = {}
        for k in _AGGREGATION_METRIC_KEYS:
            raw = (sums[k] / counts[k]) if counts[k] > 0 else None
            avg_metrics[k] = self._safe_num(raw)

        return EvaluationAggregation(
            evaluated=True,
            eval_count=eval_count,
            avg_metrics=avg_metrics,
            last_evaluated_at=last_evaluated_at,
        )

    def _extract_metric_value(self, key: str, results: Dict) -> Optional[float]:
        v = results.get(key)
        if isinstance(v, (int, float)):
            return float(v)

        summary = results.get("summary") if isinstance(results.get("summary"), dict) else None
        if summary is not None:
            v = summary.get(key)
            if isinstance(v, (int, float)):
                return float(v)

        ic_sub = results.get("ic") if isinstance(results.get("ic"), dict) else None
        if ic_sub is not None and key in _IC_KEY_MAP:
            v = ic_sub.get(_IC_KEY_MAP[key])
            if v is None:
                v = ic_sub.get(key)
            if isinstance(v, (int, float)):
                return float(v)

        ret_sub = results.get("returns") if isinstance(results.get("returns"), dict) else None
        if ret_sub is not None and key in _RETURNS_KEYS:
            v = ret_sub.get(key)
            if isinstance(v, (int, float)):
                return float(v)

        return None

    @staticmethod
    def _safe_num(x) -> Optional[float]:
        if isinstance(x, (int, float)) and math.isfinite(x):
            return float(x)
        return None

    def _extract_latest_evaluation_summary(self, payload: Dict) -> Dict:
        evaluations = (payload or {}).get("evaluations") or []
        if not evaluations:
            return {}
        latest = evaluations[-1] or {}
        results = latest.get("results") or {}
        summary = results.get("summary") if isinstance(results, dict) else None
        if isinstance(summary, dict) and summary:
            return summary
        return results if isinstance(results, dict) else {}

    def _matches(self, summary: FactorSummary, query: FactorQuery) -> bool:
        if query.keyword:
            haystack = " ".join([
                summary.factor_id.lower(),
                summary.name.lower(),
                summary.source_group.lower(),
                summary.factor_kind.lower(),
                " ".join(tag.lower() for tag in summary.tags),
            ])
            if query.keyword.lower() not in haystack:
                return False
        if query.source_group and summary.source_group != query.source_group:
            return False
        if query.factor_kind and summary.factor_kind != query.factor_kind:
            return False
        if query.computation_type and summary.computation_type != query.computation_type:
            return False
        for tag in query.include_tags:
            if tag not in summary.tags:
                return False
        for trait_name, trait_value in query.include_traits.items():
            if summary.traits.get(trait_name) != trait_value:
                return False
        return True
