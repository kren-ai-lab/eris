from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import csv
import json

from eris.export.torch import load_torch


def _read_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(dict(r))
    if not rows:
        raise ValueError(f"Empty index file: {path}")
    return rows


def _is_compatible(meta: Dict[str, Any], prefer: str) -> bool:
    shapes = meta.get("shapes", {})
    if not isinstance(shapes, dict):
        shapes = {}

    has_grid = "grid" in shapes
    has_graph = "edge_index" in shapes
    has_cloud = "pos" in shapes
    has_matrix = any(k in shapes for k in ("dist", "dist_bins", "contact", "contact_bins"))

    if prefer == "grid":
        return has_grid
    if prefer == "graph":
        return has_graph
    if prefer == "cloud":
        # clouds “puros”: pos pero no edge_index ni grid
        return has_cloud and (not has_graph) and (not has_grid)
    if prefer == "matrix":
        return has_matrix
    if prefer == "grid_or_graph":
        return has_grid or has_graph
    return True


@dataclass
class ErisFolderDataset:
    """
    Dataset que consume `index.csv` generado por `eris pack`.
    Carga artifacts con `load_torch(...)`.

    Parámetros
    ----------
    index_csv:
        Path al index.csv.
    prefer:
        'grid', 'graph', 'cloud', 'matrix', o 'grid_or_graph'
    encoders:
        allowlist opcional de encoders.
    device:
        'cpu', 'cuda', etc.
    transform:
        función opcional que transforma el sample dict.
    auto_filter:
        filtra filas incompatibles usando meta.json["shapes"].
    matrix_as_channel:
        si True, matrices se entregan como (1,N,N)
    """
    index_csv: Union[str, Path]
    prefer: str = "grid_or_graph"
    encoders: Optional[Sequence[str]] = None
    device: Optional[str] = None
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    auto_filter: bool = True
    matrix_as_channel: bool = True

    def __post_init__(self) -> None:
        self.index_csv = Path(self.index_csv)
        self.rows = _read_csv(self.index_csv)

        if self.encoders is not None:
            allow = set(self.encoders)
            self.rows = [r for r in self.rows if r.get("encoder") in allow]

        if self.auto_filter:
            kept = []
            for r in self.rows:
                meta_path = Path(r.get("meta_path", ""))
                if not meta_path.exists():
                    continue
                try:
                    meta = json.loads(meta_path.read_text())
                except Exception:
                    continue
                if _is_compatible(meta, self.prefer):
                    kept.append(r)
            self.rows = kept

        if not self.rows:
            raise ValueError("No rows left after filtering (encoders/compatibility).")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        artifact_dir = Path(r["artifact_dir"])

        sample = load_torch(
            artifact_dir, prefer=self.prefer, device=self.device
        )

        # attach identifiers
        sample["id"] = r.get("id")
        sample["encoder"] = r.get("encoder")
        sample["artifact_dir"] = str(artifact_dir)

        # optional csv fields
        for k in ("n_residues", "n_edges", "grid_shape"):
            if k in r and r[k] != "":
                sample[k] = r[k]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


__all__ = ["ErisFolderDataset"]
