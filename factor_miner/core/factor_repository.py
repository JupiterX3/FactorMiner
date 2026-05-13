"""
Unified factor repository skeleton.

The repository encapsulates low-level file layout access and returns
normalized schema objects. The initial implementation is intentionally
lightweight so the project can adopt it incrementally.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from .factor_schema import FactorDefinition, normalize_definition


class FactorRepository:
    """Low-level file repository for factor definitions and artifacts."""

    EXCLUDED_NAME_TOKENS = ("_archived_", "_deprecated", "_backup")

    def __init__(self, storage_dir: Optional[str] = None):
        if storage_dir is None:
            storage_dir = Path(__file__).parent.parent.parent / "factorlib"
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def list_source_groups(self) -> List[str]:
        groups: List[str] = []
        if not self.storage_dir.exists():
            return groups
        for child in self.storage_dir.iterdir():
            if not child.is_dir():
                continue
            if child.name.startswith((".", "_")):
                continue
            if any(token in child.name for token in self.EXCLUDED_NAME_TOKENS):
                continue
            if (child / "definitions").is_dir():
                groups.append(child.name)
        return sorted(groups)

    def iter_definition_files(self) -> List[Path]:
        definition_files: List[Path] = []
        for group in self.list_source_groups():
            definition_files.extend((self.storage_dir / group / "definitions").glob("*.json"))
        return sorted(definition_files)

    def find_definition_file(self, factor_id: str) -> Optional[Path]:
        for group in self.list_source_groups():
            candidate = self.storage_dir / group / "definitions" / f"{factor_id}.json"
            if candidate.exists():
                return candidate
        return None

    def load_definition(self, factor_id: str) -> Optional[FactorDefinition]:
        definition_file = self.find_definition_file(factor_id)
        if definition_file is None:
            return None
        with open(definition_file, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return normalize_definition(raw, known_source_groups=self.list_source_groups())

    def save_definition(self, factor_def: FactorDefinition) -> Path:
        definition_dir = self.storage_dir / factor_def.source_group / "definitions"
        definition_dir.mkdir(parents=True, exist_ok=True)
        definition_file = definition_dir / f"{factor_def.factor_id}.json"
        with open(definition_file, "w", encoding="utf-8") as handle:
            json.dump(factor_def.to_dict(), handle, ensure_ascii=False, indent=2)
        return definition_file

    def load_json_file(self, file_path: Path) -> Dict:
        with open(file_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def save_json_file(self, file_path: Path, payload: Dict) -> Path:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return file_path

    def load_text_artifact(self, relative_path: str) -> str:
        artifact_path = self.storage_dir / relative_path
        with open(artifact_path, "r", encoding="utf-8") as handle:
            return handle.read()

    def save_text_artifact(self, relative_path: str, content: str) -> Path:
        artifact_path = self.storage_dir / relative_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with open(artifact_path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return artifact_path

    def load_evaluations(self, factor_id: str) -> Dict:
        for group in self.list_source_groups():
            for filename in (f"{factor_id}_evaluation.json", f"{factor_id}.json"):
                candidate = self.storage_dir / group / "evaluations" / filename
                if candidate.exists():
                    return self.load_json_file(candidate)
        return {"factor_id": factor_id, "evaluations": []}

    def save_evaluations(self, factor_id: str, payload: Dict, source_group: str) -> Path:
        eval_path = self.storage_dir / source_group / "evaluations" / f"{factor_id}.json"
        return self.save_json_file(eval_path, payload)

    def delete_factor_files(self, factor_id: str) -> List[Path]:
        deleted: List[Path] = []
        for group in self.list_source_groups():
            for relative in (
                ("definitions", f"{factor_id}.json"),
                ("functions", f"{factor_id}.py"),
                ("formulas", f"{factor_id}.txt"),
                ("evaluations", f"{factor_id}_evaluation.json"),
                ("evaluations", f"{factor_id}.json"),
            ):
                candidate = self.storage_dir / group / relative[0] / relative[1]
                if candidate.exists():
                    candidate.unlink()
                    deleted.append(candidate)
        return deleted

    def index_file(self) -> Path:
        return self.storage_dir / "factor_index.json"
