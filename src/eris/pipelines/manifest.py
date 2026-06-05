
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar


@dataclass
class BaseManifestRow(ABC):
    id: str

    @classmethod
    @abstractmethod
    def from_dict(d: Dict[str, Any]):
        raise NotImplementedError("Manifest from_dict not implemented.")


@dataclass
class StructureManifestRow(BaseManifestRow):
    path: str
    chain: Optional[str] = None
    model: int = 0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StructureManifestRow":
        path = str(d.get("path") or d.get("input") or "").strip()
        if not path:
            raise ValueError("Manifest row missing 'path' (or 'input').")

        sid = str(d.get("id") or d.get("sample_id") or "").strip()
        if not sid:
            sid = Path(path).stem

        chain = str(d.get("chain") or d.get("chain_id") or "").strip() or None

        model_raw = d.get("model") or d.get("model_index") or 0
        try:
            model = int(model_raw)
        except Exception:
            model = 0

        return cls(id=sid, path=path, chain=chain, model=model)
    

@dataclass
class ArtifactManifestRow(BaseManifestRow):
    artifact_dir: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ArtifactManifestRow":
        artifact_dir = str(d.get("path") or d.get("input") or "" or d.get("artifact_dir")).strip()
        if not artifact_dir:
            raise ValueError("Manifest row missing 'path' (or 'input').")

        sid = str(d.get("id") or d.get("sample_id") or "").strip()
        if not sid:
            sid = Path(artifact_dir).stem

        return cls(id=sid, artifact_dir=artifact_dir)



T = TypeVar("T", bound=BaseManifestRow)

def read_manifest(path: Path, row_cls: Type[T]) -> List[T]:
    """
    Supported formats:
      - .csv with headers: id,path,chain,model (chain/model optional)
      - .jsonl where each line is a dict with keys like: id/path/chain/model
      - .json list[dict]
    """
    suf = path.suffix.lower()

    if suf == ".csv":
        rows: List[T] = []
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for d in reader:
                rows.append(row_cls.from_dict(d))
        if not rows:
            raise ValueError(f"Empty CSV manifest: {path}")
        return rows

    if suf == ".jsonl":
        rows = []
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(row_cls.from_dict(json.loads(line)))
        if not rows:
            raise ValueError(f"Empty JSONL manifest: {path}")
        return rows

    if suf == ".json":
        obj = json.loads(path.read_text())
        if not isinstance(obj, list):
            raise ValueError("JSON manifest must be a list of objects.")
        rows = [row_cls.from_dict(d) for d in obj]
        if not rows:
            raise ValueError(f"Empty JSON manifest: {path}")
        return rows

    raise ValueError(f"Unsupported manifest format: {path.suffix} (use .csv, .jsonl, or .json)")