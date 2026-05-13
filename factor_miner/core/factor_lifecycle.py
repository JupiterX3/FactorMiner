"""
Unified factor lifecycle skeleton.

This service owns write-path operations such as save, delete, migrate
and health checks. The initial version provides validation scaffolding
and repository wiring without changing legacy callers yet.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .factor_catalog import FactorCatalogService
from .factor_repository import FactorRepository
from .factor_schema import (
    CleanupResult,
    DeleteResult,
    FactorDefinition,
    HealthIssue,
    HealthReport,
    KNOWN_COMPUTATION_TYPES,
    MigrationResult,
    SaveResult,
)

logger = logging.getLogger(__name__)


class FactorLifecycleService:
    """Write-path service for factor library lifecycle operations."""

    def __init__(
        self,
        repository: Optional[FactorRepository] = None,
        catalog: Optional[FactorCatalogService] = None,
    ):
        self.repository = repository or FactorRepository()
        self.catalog = catalog or FactorCatalogService(self.repository)

    def save_factor(
        self,
        factor_def: FactorDefinition,
        artifacts: Optional[Dict[str, str]] = None,
        overwrite: bool = False,
        validate: bool = True,
    ) -> SaveResult:
        if validate:
            self.validate_definition(factor_def)
        existing = self.repository.find_definition_file(factor_def.factor_id)
        if existing is not None and not overwrite:
            return SaveResult(
                success=False,
                factor_id=factor_def.factor_id,
                message="factor already exists",
            )

        written_files: List[str] = []
        artifacts = artifacts or {}

        function_file = factor_def.artifacts.function_file
        if function_file and function_file in artifacts:
            written_files.append(str(self.repository.save_text_artifact(function_file, artifacts[function_file])))

        formula_file = factor_def.artifacts.formula_file
        if formula_file and formula_file in artifacts:
            written_files.append(str(self.repository.save_text_artifact(formula_file, artifacts[formula_file])))

        written_files.append(str(self.repository.save_definition(factor_def)))
        self.catalog.invalidate_index_cache()
        self.catalog.update_index_entry(factor_def.factor_id)
        return SaveResult(
            success=True,
            factor_id=factor_def.factor_id,
            message="factor saved",
            written_files=written_files,
        )

    def delete_factor(self, factor_id: str, cascade: bool = True) -> DeleteResult:
        deleted_files = [str(path) for path in self.repository.delete_factor_files(factor_id)]
        if cascade:
            self.catalog.invalidate_index_cache()
            self.catalog.rebuild_index()
        return DeleteResult(
            success=bool(deleted_files),
            factor_id=factor_id,
            message="factor deleted" if deleted_files else "factor not found",
            deleted_files=deleted_files,
        )

    def migrate_legacy_definition(self, factor_id: str) -> MigrationResult:
        factor_def = self.repository.load_definition(factor_id)
        if factor_def is None:
            return MigrationResult(
                success=False,
                factor_id=factor_id,
                message="factor not found",
            )
        before = {}
        definition_file = self.repository.find_definition_file(factor_id)
        if definition_file is not None:
            before = self.repository.load_json_file(definition_file)
        after = factor_def.to_dict()
        return MigrationResult(
            success=True,
            factor_id=factor_id,
            message="legacy definition normalized in memory",
            before=before,
            after=after,
        )

    def health_check(self) -> HealthReport:
        report = HealthReport()
        all_factor_ids: set = set()

        for definition_file in self.repository.iter_definition_files():
            factor_id = definition_file.stem
            all_factor_ids.add(factor_id)
            factor_def = self.repository.load_definition(factor_id)
            if factor_def is None:
                report.missing_definitions.append(str(definition_file))
                report.issues.append(HealthIssue(
                    factor_id=factor_id,
                    issue_type="missing_definition",
                    detail=f"definition file exists but cannot be loaded: {definition_file}",
                ))
                continue
            if factor_def.artifacts.function_file:
                candidate = self.repository.storage_dir / factor_def.artifacts.function_file
                if not candidate.exists():
                    report.missing_artifacts.append(str(candidate))
                    report.issues.append(HealthIssue(
                        factor_id=factor_id,
                        issue_type="missing_artifact",
                        detail=f"function_file not found: {candidate}",
                    ))
            if factor_def.artifacts.formula_file and not factor_def.artifacts.formula_inline:
                candidate = self.repository.storage_dir / factor_def.artifacts.formula_file
                if not candidate.exists():
                    report.missing_artifacts.append(str(candidate))
                    report.issues.append(HealthIssue(
                        factor_id=factor_id,
                        issue_type="missing_artifact",
                        detail=f"formula_file not found: {candidate}",
                    ))
            if factor_def.computation_type == "algorithm_proxy":
                algo_name = factor_def.artifacts.algorithm_name
                if algo_name:
                    algo_path = self.repository.storage_dir.parent / "user_algo" / f"{algo_name}.py"
                    if not algo_path.exists():
                        report.missing_artifacts.append(str(algo_path))
                        report.issues.append(HealthIssue(
                            factor_id=factor_id,
                            issue_type="missing_artifact",
                            detail=f"algorithm proxy not found: {algo_path}",
                        ))

        report.total_factors = len(all_factor_ids)

        index_path = self.repository.index_file()
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    index_data = json.load(f)
                report.index_entry_count = len(index_data) if isinstance(index_data, list) else len(index_data.get("factors", []))
                index_ids = set()
                if isinstance(index_data, list):
                    for entry in index_data:
                        if isinstance(entry, dict) and "factor_id" in entry:
                            index_ids.add(entry["factor_id"])
                elif isinstance(index_data, dict):
                    for entry in index_data.get("factors", []):
                        if isinstance(entry, dict) and "factor_id" in entry:
                            index_ids.add(entry["factor_id"])
                if index_ids != all_factor_ids:
                    report.index_ok = False
                    missing_in_index = all_factor_ids - index_ids
                    extra_in_index = index_ids - all_factor_ids
                    if missing_in_index:
                        report.issues.append(HealthIssue(
                            factor_id="",
                            issue_type="index_missing_entries",
                            detail=f"factors in disk but not in index: {sorted(missing_in_index)[:10]}",
                        ))
                    if extra_in_index:
                        report.issues.append(HealthIssue(
                            factor_id="",
                            issue_type="index_stale_entries",
                            detail=f"factors in index but not on disk: {sorted(extra_in_index)[:10]}",
                        ))
            except Exception as e:
                report.index_ok = False
                report.issues.append(HealthIssue(
                    factor_id="",
                    issue_type="index_corrupt",
                    detail=f"cannot parse index file: {e}",
                ))
        else:
            report.index_ok = False
            report.issues.append(HealthIssue(
                factor_id="",
                issue_type="index_missing",
                detail="factor_index.json does not exist",
            ))

        report.ok = (
            not report.missing_definitions
            and not report.missing_artifacts
            and report.index_ok
        )
        return report

    def cleanup_orphans(self, dry_run: bool = True) -> CleanupResult:
        result = CleanupResult()
        known_factor_ids: set = set()
        for definition_file in self.repository.iter_definition_files():
            known_factor_ids.add(definition_file.stem)

        for group in self.repository.list_source_groups():
            group_dir = self.repository.storage_dir / group
            if not group_dir.exists():
                continue
            for subdir_name in ("functions", "formulas", "evaluations"):
                subdir = group_dir / subdir_name
                if not subdir.exists():
                    continue
                for file_path in subdir.iterdir():
                    if not file_path.is_file():
                        continue
                    stem = file_path.stem
                    if stem.endswith("_evaluation"):
                        stem = stem[: -len("_evaluation")]
                    if stem not in known_factor_ids:
                        result.orphan_files.append(str(file_path))
                        if dry_run:
                            result.kept_files.append(str(file_path))
                        else:
                            try:
                                file_path.unlink()
                                result.cleaned_files.append(str(file_path))
                                logger.info(f"cleaned orphan: {file_path}")
                            except Exception as e:
                                logger.error(f"failed to clean orphan {file_path}: {e}")
                                result.kept_files.append(str(file_path))

        result.cleaned_count = len(result.cleaned_files)
        return result

    def validate_index(self) -> HealthReport:
        self.catalog.rebuild_index()
        return self.health_check()

    def validate_definition(self, factor_def: FactorDefinition) -> None:
        if not factor_def.factor_id:
            raise ValueError("factor_id is required")
        if not factor_def.source_group:
            raise ValueError("source_group is required")
        if factor_def.computation_type not in KNOWN_COMPUTATION_TYPES:
            raise ValueError(
                f"computation_type must be one of {KNOWN_COMPUTATION_TYPES}, "
                f"got '{factor_def.computation_type}'"
            )
        if factor_def.computation_type == "function" and not factor_def.artifacts.function_file:
            raise ValueError("function factors require artifacts.function_file")
        if factor_def.computation_type == "formula":
            if not factor_def.artifacts.formula_file and not factor_def.artifacts.formula_inline:
                raise ValueError("formula factors require formula_file or formula_inline")
        if factor_def.computation_type == "algorithm_proxy" and not factor_def.artifacts.algorithm_name:
            raise ValueError("algorithm_proxy factors require artifacts.algorithm_name")
