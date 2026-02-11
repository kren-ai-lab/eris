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
        dt = getattr(torch, dtype, None)
        if dt is None:
            raise ValueError(
                f"Unknown torch dtype '{dtype}'. Example: 'float32', 'float16', 'int64'."
            )
        t = t.to(dt)
    return t


def to_torch(
    artifact: StructureArtifact,
    *,
    as_dict: bool = True,
    prefer: str = "grid_or_graph",
    device: Optional[str] = None,
    matrix_as_channel: bool = True,
    cloud_x_mode: str = "none",  # "none" | "auto" | "onehot" | "onehot+plddt" | "onehot+plddt+bfactor"
) -> Dict[str, Any]:
    """
    Convert Morpheo artifact -> dict of torch tensors.

    prefer:
      - "grid_or_graph": grid if present, else graph if present, else matrix if present, else cloud if present
      - "grid": require 'grid'
      - "graph": require 'edge_index'
      - "matrix": require one of ('dist','dist_bins','contact','contact_bins')
      - "cloud": require 'pos'

    Additions (auto features):
      - Graph: prefer 'node_attr' as x, then 'node_emb', else aa_index->onehot (+ optional plddt)
      - Graph: expose 'edge_attr' if present (keeps 'edge_dist' too)
      - Cloud: if cloud_x_mode == "auto" and 'node_attr' exists, use it as x; else fall back to onehot modes.
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

    if prefer == "grid":
        if not has_grid:
            raise KeyError("Artifact has no 'grid'.")
    elif prefer == "graph":
        if not has_graph:
            raise KeyError("Artifact has no 'edge_index'.")
    elif prefer == "matrix":
        if not has_matrix:
            raise KeyError("Artifact has no matrix key ('dist','dist_bins','contact','contact_bins').")
    elif prefer == "cloud":
        if not has_pos:
            raise KeyError("Artifact has no 'pos'.")
    elif prefer != "grid_or_graph":
        raise ValueError("prefer must be one of: grid_or_graph, grid, graph, matrix, cloud")

    # --- GRID
    if prefer in ("grid", "grid_or_graph") and has_grid:
        out["x"] = _to_tensor(d["grid"], dtype=None)
        out["channels"] = d.get("channels", None)
        out["grid_size"] = _to_tensor(d.get("grid_size", [out["x"].shape[-1]]), dtype="int32")
        if "voxel_size" in d:
            out["voxel_size"] = _to_tensor(d["voxel_size"], dtype="float32")
        if "origin" in d:
            out["origin"] = _to_tensor(d["origin"], dtype="float32")
        if "centroid" in d:
            out["centroid"] = _to_tensor(d["centroid"], dtype="float32")

    # --- GRAPH
    elif prefer in ("graph", "grid_or_graph") and has_graph:
        # NEW: prefer deterministic node_attr if present (ns encoders)
        if "node_attr" in d:
            out["x"] = _to_tensor(d["node_attr"], dtype="float32")

        # passthrough embeddings if present
        elif "node_emb" in d:
            out["x"] = _to_tensor(d["node_emb"], dtype="float32")

        # fallback: build from aa_index (+ optional plddt)
        elif "aa_index" in d:
            aa = np.asarray(d["aa_index"]).astype(np.int64).reshape(-1)
            n = aa.shape[0]
            onehot = np.zeros((n, 21), dtype=np.float32)
            valid = (aa >= 0) & (aa < 21)
            onehot[np.arange(n)[valid], aa[valid]] = 1.0

            feats = [onehot]
            if "plddt" in d:
                p = np.asarray(d["plddt"], dtype=np.float32).reshape(-1, 1)
                p = np.nan_to_num(p, nan=0.0)
                if float(np.nanmax(p)) > 1.5:
                    p = p / 100.0
                feats.append(p)

            out["x"] = _to_tensor(np.concatenate(feats, axis=1), dtype="float32")
        else:
            raise KeyError("Graph artifact lacks 'node_attr', 'aa_index' or 'node_emb' to build x.")

        out["edge_index"] = _to_tensor(d["edge_index"], dtype="int64")

        # NEW: pass edge_attr if present
        if "edge_attr" in d:
            out["edge_attr"] = _to_tensor(d["edge_attr"], dtype="float32")

        if "edge_dist" in d:
            out["edge_dist"] = _to_tensor(d["edge_dist"], dtype="float32")
        if "pos" in d:
            out["pos"] = _to_tensor(d["pos"], dtype="float32")

    # --- MATRIX
    elif prefer == "matrix" or (prefer == "grid_or_graph" and has_matrix):
        mk = matrix_key  # type: ignore[assignment]
        mat = d[mk]      # type: ignore[index]
        t = _to_tensor(mat, dtype=None)

        if t.ndim != 2:
            raise ValueError(f"Matrix key '{mk}' must be 2D, got shape {tuple(t.shape)}")

        if matrix_as_channel:
            t = t.unsqueeze(0)  # (1, N, N)

        out["x"] = t
        out["matrix_key"] = mk
        out["n"] = int(t.shape[-1])

        if "bin_edges" in d:
            out["bin_edges"] = _to_tensor(d["bin_edges"], dtype="float32")
        if "pos" in d:
            out["pos"] = _to_tensor(d["pos"], dtype="float32")

    # --- CLOUD fallback
    else:
        if not has_pos:
            raise KeyError(
                "Artifact cannot be represented as torch input: missing 'grid', 'edge_index', 'matrix', and 'pos'."
            )

        out["pos"] = _to_tensor(d["pos"], dtype="float32")

        mode = (cloud_x_mode or "none").strip().lower()
        if mode not in {"none", "auto", "onehot", "onehot+plddt", "onehot+plddt+bfactor"}:
            raise ValueError("cloud_x_mode must be: none | auto | onehot | onehot+plddt | onehot+plddt+bfactor")

        # NEW: auto uses node_attr if available, else falls back to onehot
        if mode == "auto":
            if "node_attr" in d:
                out["x"] = _to_tensor(d["node_attr"], dtype="float32")
                mode = "none"  # already set x, skip onehot construction
            else:
                mode = "onehot+plddt" if "plddt" in d else "onehot"

        # Build point-wise x if requested and possible
        if mode != "none":
            if "aa_index" not in d:
                raise KeyError("cloud_x_mode requires 'aa_index' in artifact.")

            aa = np.asarray(d["aa_index"], dtype=np.int64).reshape(-1)
            n = aa.shape[0]

            onehot = np.zeros((n, 21), dtype=np.float32)
            valid = (aa >= 0) & (aa < 21)
            onehot[np.arange(n)[valid], aa[valid]] = 1.0

            feats = [onehot]

            if "plddt" in d and ("plddt" in mode):
                p = np.asarray(d["plddt"], dtype=np.float32).reshape(-1, 1)
                p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
                if float(np.nanmax(p)) > 1.5:
                    p = p / 100.0
                feats.append(p)

            if "bfactor_ca" in d and ("bfactor" in mode):
                b = np.asarray(d["bfactor_ca"], dtype=np.float32).reshape(-1, 1)
                b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
                if float(np.nanmax(b)) > 1.5:
                    b = b / 100.0
                feats.append(b)

            out["x"] = _to_tensor(np.concatenate(feats, axis=1), dtype="float32")

        # still attach raw scalars if present (useful for debugging)
        if "aa_index" in d:
            out["aa_index"] = _to_tensor(d["aa_index"], dtype="int64")
        if "plddt" in d:
            out["plddt"] = _to_tensor(
                np.nan_to_num(np.asarray(d["plddt"], dtype=np.float32), nan=0.0),
                dtype="float32",
            )
        if "bfactor_ca" in d:
            out["bfactor_ca"] = _to_tensor(
                np.nan_to_num(np.asarray(d["bfactor_ca"], dtype=np.float32), nan=0.0),
                dtype="float32",
            )

    # --- Attach optional debug metadata
    for k in ("residue_ids", "resname"):
        if k in d:
            out[k] = np.asarray(d[k]).astype(str).tolist()

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
    cloud_x_mode: str = "none",
) -> Dict[str, Any]:
    artifact = StructureArtifact.load(Path(artifact_dir))
    return to_torch(
        artifact,
        prefer=prefer,
        device=device,
        matrix_as_channel=matrix_as_channel,
        cloud_x_mode=cloud_x_mode,
    )
