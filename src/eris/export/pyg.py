from __future__ import annotations

from pathlib import Path
from typing import Literal, Dict, Any, Optional

import numpy as np

from eris.core.types import StructureArtifact
from eris.export.torch import to_torch



def torch_dict_to_pyg(out: Dict[str, Any], *, device: Optional[str] = None):
    """
    Convert the canonical torch dict (from to_torch()) into torch_geometric.data.Data.

    Strategy:
      1) Infer representation kind (graph/grid/matrix/cloud) from keys.
      2) Build Data using the appropriate schema.
      3) Attach remaining tensors/metadata as extra attributes (serializable via torch.save).
    """
    try:
        import torch
        from torch_geometric.data import Data
    except Exception as e:
        raise ImportError("PyG export requires torch + torch_geometric installed.") from e

    # ---------
    # 1) Infer kind (prefer explicit signals)
    # ---------
    has_edge_index = "edge_index" in out
    has_pos = "pos" in out
    has_grid_signals = any(k in out for k in ("channels", "grid_size", "voxel_size", "origin"))  # grid hints
    has_matrix_signals = any(k in out for k in ("matrix_key", "bin_edges", "n"))                 # matrix hints

    if has_edge_index:
        kind = "graph"
    elif has_grid_signals:
        kind = "grid"
    elif has_matrix_signals:
        kind = "matrix"
    elif has_pos:
        kind = "cloud"
    else:
        raise KeyError("Can't convert to PyG: missing signals of graph/grid/matrix/cloud.")

    pyg = Data()

    # ---------
    # 2) Populate Data depending on kind
    # ---------
    if kind == "graph":
        pyg.edge_index = out["edge_index"]
        if "x" in out:
            pyg.x = out["x"]
        if "pos" in out:
            pyg.pos = out["pos"]
        if "edge_attr" in out:
            pyg.edge_attr = out["edge_attr"]
        if "edge_dist" in out:
            pyg.edge_dist = out["edge_dist"]
        if "graph_attr" in out:
            pyg.graph_attr = out["graph_attr"]
        if "graph_attr_keys" in out:
            pyg.graph_attr_keys = out["graph_attr_keys"]

        if "centroid" in out:
            pyg.centroid = out["centroid"]

    elif kind == "cloud":
        # Cloud: PyG can store pos and optional x
        pyg.pos = out["pos"]
        if "x" in out:
            pyg.x = out["x"]
        if "centroid" in out:
            pyg.centroid = out["centroid"]

    elif kind == "grid":
        # Grid has no natural PyG "x" meaning; store as pyg.grid instead
        # (out["x"] in grid mode is the grid tensor)
        if "x" in out:
            pyg.grid = out["x"]
        if "channels" in out:
            pyg.channels = out["channels"]
        if "grid_size" in out:
            pyg.grid_size = out["grid_size"]
        if "voxel_size" in out:
            pyg.voxel_size = out["voxel_size"]
        if "origin" in out:
            pyg.origin = out["origin"]
        if "centroid" in out:
            pyg.centroid = out["centroid"]

    elif kind == "matrix":
        # Matrix has no natural PyG "x" meaning; store as pyg.matrix instead
        # (out["x"] in matrix mode is the matrix tensor)
        if "x" in out:
            pyg.matrix = out["x"]
        if "matrix_key" in out:
            pyg.matrix_key = out["matrix_key"]
        if "n" in out:
            pyg.n = out["n"]
        if "bin_edges" in out:
            pyg.bin_edges = out["bin_edges"]
        if "pos" in out:
            pyg.pos = out["pos"]
        if "centroid" in out:
            pyg.centroid = out["centroid"]

    # ---------
    # 3) Attach metadata (python objects) safely
    # ---------
    if "extra" in out:
        pyg.extra = out["extra"]

    # Optionally attach any remaining tensor fields that were not consumed above.
    # This is useful for debugging, but keep it conservative to avoid collisions.
    consumed = set()
    if kind == "graph":
        consumed |= {"edge_index", "x", "pos", "edge_attr", "edge_dist", "graph_attr", "graph_attr_keys", "centroid", "extra"}
    elif kind == "cloud":
        consumed |= {"pos", "x", "centroid", "extra"}
    elif kind == "grid":
        consumed |= {"x", "channels", "grid_size", "voxel_size", "origin", "centroid", "extra"}
    elif kind == "matrix":
        consumed |= {"x", "matrix_key", "n", "bin_edges", "pos", "centroid", "extra"}

    for k, v in out.items():
        if k in consumed:
            continue
        if isinstance(v, torch.Tensor):
            # attach as attribute; avoid overwriting core PyG names
            setattr(pyg, k, v)

    # ---------
    # 4) Move to device
    # ---------
    if device is not None:
        pyg = pyg.to(device)

    return pyg


def to_pyg(
    artifact_dir: str | Path,
    *,
    prefer: str = "grid_or_graph",
    device: Optional[str] = None,
    matrix_as_channel: bool = True,
):
    out = to_torch(
        artifact_dir,
        prefer=prefer,
        device=device,
        matrix_as_channel=matrix_as_channel,
    )
    return torch_dict_to_pyg(out, device=device)


def load_pyg(
    artifact_dir: str | Path,
    *,
    prefer: str = "grid_or_graph",
    device: Optional[str] = None,
    matrix_as_channel: bool = True,
):
    artifact = StructureArtifact.load(Path(artifact_dir))
    return to_pyg(artifact, prefer=prefer, device=device, matrix_as_channel=matrix_as_channel)


def save_artifact_to_pyg(
    artifact_dir: Path,
    *,
    out_path: Path,
    prefer: str = "grid_or_graph",
    device: Optional[str] = None,
    matrix_as_channel: bool = True,
):
    import torch

    artifact = StructureArtifact.load(artifact_dir)

    pyg = to_pyg(
        artifact,
        prefer=prefer,
        device=device,
        matrix_as_channel=matrix_as_channel,
    )

    torch.save(pyg, out_path)

