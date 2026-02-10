from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from morpheo.core.types import StructureArtifact


def _to_tensor(x: Any, *, dtype: Optional[str] = None):
    import torch

    if isinstance(x, torch.Tensor):
        t = x
    else:
        arr = np.asarray(x)
        t = torch.from_numpy(arr)

    if dtype is not None:
        # dtype string -> torch dtype
        dt = getattr(torch, dtype, None)
        if dt is None:
            raise ValueError(f"Unknown torch dtype '{dtype}'. Example: 'float32', 'float16', 'int64'.")
        t = t.to(dt)
    return t


def to_torch(
    artifact: StructureArtifact,
    *,
    as_dict: bool = True,
    prefer: str = "grid_or_graph",
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convert Morpheo artifact -> dict of torch tensors.

    prefer:
      - "grid_or_graph": if 'grid' exists, expose it as x; else if 'edge_index' exists, treat as graph.
      - "grid": require 'grid'
      - "graph": require 'edge_index'
      - "cloud": require 'pos'
    """
    import torch

    d = artifact.data
    out: Dict[str, Any] = {}

    has_grid = "grid" in d
    has_graph = "edge_index" in d
    has_pos = "pos" in d

    if prefer == "grid":
        if not has_grid:
            raise KeyError("Artifact has no 'grid'.")
    elif prefer == "graph":
        if not has_graph:
            raise KeyError("Artifact has no 'edge_index'.")
    elif prefer == "cloud":
        if not has_pos:
            raise KeyError("Artifact has no 'pos'.")
    elif prefer != "grid_or_graph":
        raise ValueError("prefer must be one of: grid_or_graph, grid, graph, cloud")

    # --- X (main input)
    if prefer in ("grid", "grid_or_graph") and has_grid:
        out["x"] = _to_tensor(d["grid"], dtype=None)  # keep dtype from np (float16/float32)
        out["channels"] = d.get("channels", None)
        out["grid_size"] = _to_tensor(d.get("grid_size", [out["x"].shape[-1]]), dtype="int32")
        if "voxel_size" in d:
            out["voxel_size"] = _to_tensor(d["voxel_size"], dtype="float32")
        if "origin" in d:
            out["origin"] = _to_tensor(d["origin"], dtype="float32")
        if "centroid" in d:
            out["centroid"] = _to_tensor(d["centroid"], dtype="float32")

    elif prefer in ("graph", "grid_or_graph") and has_graph:
        # build node features minimally from available arrays
        # default: onehot+plddt if aa_index exists
        if "aa_index" in d:
            aa = np.asarray(d["aa_index"]).astype(np.int64).reshape(-1)
            n = aa.shape[0]
            onehot = np.zeros((n, 21), dtype=np.float32)
            valid = (aa >= 0) & (aa < 21)
            onehot[np.arange(n)[valid], aa[valid]] = 1.0

            feats = [onehot]
            if "plddt" in d:
                p = np.asarray(d["plddt"], dtype=np.float32).reshape(-1, 1)
                p = np.nan_to_num(p, nan=0.0)
                # normalize to [0,1] if it looks like 0..100
                if float(np.nanmax(p)) > 1.5:
                    p = p / 100.0
                feats.append(p)

            out["x"] = _to_tensor(np.concatenate(feats, axis=1), dtype="float32")
        elif "node_emb" in d:
            out["x"] = _to_tensor(d["node_emb"], dtype="float32")
        else:
            raise KeyError("Graph artifact lacks 'aa_index' or 'node_emb' to build x.")

        out["edge_index"] = _to_tensor(d["edge_index"], dtype="int64")
        if "edge_dist" in d:
            out["edge_dist"] = _to_tensor(d["edge_dist"], dtype="float32")
        if "pos" in d:
            out["pos"] = _to_tensor(d["pos"], dtype="float32")

    else:
        # cloud fallback
        if not has_pos:
            raise KeyError("Artifact cannot be represented as torch input: missing 'grid', 'edge_index', and 'pos'.")
        out["pos"] = _to_tensor(d["pos"], dtype="float32")
        if "aa_index" in d:
            out["aa_index"] = _to_tensor(d["aa_index"], dtype="int64")
        if "plddt" in d:
            out["plddt"] = _to_tensor(np.nan_to_num(np.asarray(d["plddt"], dtype=np.float32), nan=0.0), dtype="float32")

    # --- Attach optional debug metadata (kept as python objects)
    for k in ("residue_ids", "resname"):
        if k in d:
            out[k] = np.asarray(d[k]).astype(str).tolist()

    # Move to device if requested
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
) -> Dict[str, Any]:
    artifact = StructureArtifact.load(Path(artifact_dir))
    return to_torch(artifact, prefer=prefer, device=device)
