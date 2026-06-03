from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from eris.core.types import StructureArtifact


def _to_tensor(x: Any, *, dtype: Optional[str] = None):
    import torch

    if isinstance(x, torch.Tensor):
        t = x
    else:
        arr = np.asarray(x)
        t = torch.from_numpy(arr)

    if dtype is not None:
        dt = getattr(torch, dtype, None)
        if dt is None:
            raise ValueError(
                f"Unknown torch dtype '{dtype}'. Example: 'float32', 'float16', 'int64'."
            )
        t = t.to(dt)
    return t


def _grid_to_torch(data: Dict[str, Any]):
    """
    """
    out: Dict[str, Any] = {}

    out["x"] = _to_tensor(data["grid"], dtype=None)
    out["channels"] = data.get("channels", None)
    out["grid_size"] = _to_tensor(data.get("grid_size", [out["x"].shape[-1]]), dtype="int32")
    
    if "voxel_size" in data:
        out["voxel_size"] = _to_tensor(data["voxel_size"], dtype="float32")
    
    if "origin" in data:
        out["origin"] = _to_tensor(data["origin"], dtype="float32")
    
    if "centroid" in data:
        out["centroid"] = _to_tensor(data["centroid"], dtype="float32")

    return out


def _graph_to_torch(data: Dict[str, Any]):
    """
    """
    out: Dict[str, Any] = {}
    
    if "node_attr" not in data:
        raise KeyError("Graph must have node_attr to export.")

    out["x"] = _to_tensor(data["node_attr"], dtype="float32")
    out["edge_index"] = _to_tensor(data["edge_index"], dtype="int64")

    if "edge_attr" in data:
        out["edge_attr"] = _to_tensor(data["edge_attr"], dtype="float32")

    if "edge_dist" in data:
        out["edge_dist"] = _to_tensor(data["edge_dist"], dtype="float32")

    if "pos" in data:
        out["pos"] = _to_tensor(data["pos"], dtype="float32")
    
    if "centroid" in data:
        out["centroid"] = _to_tensor(data["centroid"], dtype="float32")

    if "graph_attr" in data:
        out["graph_attr"] = _to_tensor(data["graph_attr"], dtype="float32").reshape(-1)

    return out


def _matrix_to_torch(data: Dict[str, Any], matrix_key: str|None, matrix_as_channel: bool):
    """
    """
    out: Dict[str, Any] = {}

    key = matrix_key  # type: ignore[assignment]
    mat = data[key]      # type: ignore[index]
    t = _to_tensor(mat, dtype=None)

    if t.ndim != 2:
        raise ValueError(f"Matrix key '{key}' must be 2D, got shape {tuple(t.shape)}")

    if matrix_as_channel:
        t = t.unsqueeze(0)  # (1, N, N)

    out["x"] = t
    out["matrix_key"] = key
    out["n"] = int(t.shape[-1])

    if "bin_edges" in data:
        out["bin_edges"] = _to_tensor(data["bin_edges"], dtype="float32")

    if "pos" in data:
        out["pos"] = _to_tensor(data["pos"], dtype="float32")
    
    if "centroid" in data:
        out["centroid"] = _to_tensor(data["centroid"], dtype="float32")

    return out


def _point_cloud_to_torch(data: Dict[str, Any]):
    """
    """
    out: Dict[str, Any] = {}

    out["pos"] = _to_tensor(data["pos"], dtype="float32")

    if "node_attr" in data:
        out["x"] = _to_tensor(data["node_attr"], dtype="float32")

    if "centroid" in data:
        out["centroid"] = _to_tensor(data["centroid"], dtype="float32")

    return out


def to_torch(
    artifact: StructureArtifact,
    *,
    #as_dict: bool = True,
    prefer: str = "grid_or_graph",
    device: Optional[str] = None,
    matrix_as_channel: bool = True,
) -> Dict[str, Any]:
    """
    Convert Eris artifact -> dict of torch tensors.

    prefer:
      - "grid_or_graph": grid if present, else graph if present, else matrix if present, else cloud if present
      - "grid": require 'grid'
      - "graph": require 'edge_index'
      - "matrix": require one of ('dist','dist_bins','contact','contact_bins')
      - "cloud": require 'pos'

    Auto features:
      - Graph: prefer 'node_attr' as x, then 'node_emb', else aa_index->onehot (+ optional plddt)
      - Graph: expose 'edge_attr' if present (keeps 'edge_dist' too)
      - Graph: expose 'graph_attr' if present
    """
    import torch

    d = artifact.data
    out: Dict[str, Any] = {}

    has_grid = "grid" in d
    has_graph = "edge_index" in d
    has_pos = "pos" in d

    # matrix family
    matrix_key = None
    for k in ("dist", "dist_bins", "contact", "contact_bins"):
        if k in d:
            matrix_key = k
            break
    has_matrix = matrix_key is not None

    prefer = (prefer or "grid_or_graph").strip().lower()
    if prefer not in {"grid_or_graph", "grid", "graph", "matrix", "cloud"}:
        raise ValueError("prefer must be one of: grid_or_graph, grid, graph, matrix, cloud")

    if prefer == "grid" and (not has_grid):
        raise KeyError("Artifact has no 'grid'.")
    if prefer == "graph" and (not has_graph):
        raise KeyError("Artifact has no 'edge_index'.")
    if prefer == "matrix" and (not has_matrix):
        raise KeyError("Artifact has no matrix key ('dist','dist_bins','contact','contact_bins').")
    if prefer == "cloud" and (not has_pos):
        raise KeyError("Artifact has no 'pos'.")

    # --- GRID
    if prefer in ("grid", "grid_or_graph") and has_grid:
        out = _grid_to_torch(d)

    # --- GRAPH
    elif prefer in ("graph", "grid_or_graph") and has_graph:
        out = _graph_to_torch(d)

    # --- MATRIX
    elif prefer == "matrix" or (prefer == "grid_or_graph" and has_matrix):
        out = _matrix_to_torch(d, matrix_key, matrix_as_channel)
        
    # --- CLOUD fallback
    else:
        if not has_pos:
            raise KeyError(
                "Artifact cannot be represented as torch input: missing 'grid', 'edge_index', 'matrix', and 'pos'."
            )
        out = _point_cloud_to_torch(d)
    
    # --- Attach optional debug metadata (python lists)
    extra = {}
    
    # Residue extra-metadata
    for k in ("residue_ids", "resname", "aa_index", "plddt", "bfactor_ca"):
        if k in d:
            try:
                extra[k] = np.asarray(d[k]).reshape(-1).tolist()
            except Exception:
                pass

    # Atom extra-metadata
    for k in ("atom_name", "elem_index", "atom_role", "bfactor", "occ"):
        if k in d:
            try:
                extra[k] = np.asarray(d[k]).reshape(-1).tolist()
            except Exception:
                pass

    # more optional atom-level debug (keep as list, not tensor)
    for k in ("seq_pos_residue", "aa_index_residue"):
        if k in d:
            try:
                extra[k] = np.asarray(d[k]).reshape(-1).tolist()
            except Exception:
                pass

    if extra:
        out["extra"] = extra

    if device is not None:
        for k, v in list(out.items()):
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device)

    return out


def load_torch(
    artifact_dir: str | Path,
    *,
    prefer: str = "grid_or_graph",
    device: Optional[str] = None,
    matrix_as_channel: bool = True,
) -> Dict[str, Any]:
    artifact = StructureArtifact.load(Path(artifact_dir))
    return to_torch(
        artifact,
        prefer=prefer,
        device=device,
        matrix_as_channel=matrix_as_channel,
    )


def save_artifact_to_torch(
    artifact_dir,
    *,
    out_path: Path,
    prefer="grid_or_graph",
    device=None,
    matrix_as_channel=True,
):
    import torch

    artifact = StructureArtifact.load(Path(artifact_dir))

    data = to_torch(
        artifact,
        prefer=prefer,
        device=device,
        matrix_as_channel=matrix_as_channel,
    )

    torch.save(data, out_path)



