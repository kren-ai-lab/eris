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
      - Graph: expose 'graph_attr' and 'graph_attr_keys' if present
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

        # Graph-level attributes (must be available in prefer="graph")
        if "graph_attr" in d:
            out["graph_attr"] = _to_tensor(d["graph_attr"], dtype="float32").reshape(-1)

        if "graph_attr_keys" in d:
            out["graph_attr_keys"] = np.asarray(d["graph_attr_keys"]).astype(str).tolist()

        # Prefer deterministic node_attr (ns encoders)
        if "node_attr" in d:
            out["x"] = _to_tensor(d["node_attr"], dtype="float32")

        # passthrough learned embeddings if present
        elif "node_emb" in d:
            out["x"] = _to_tensor(d["node_emb"], dtype="float32")

        # fallback: residue aa_index -> onehot (+ optional plddt)
        elif "aa_index" in d:
            aa = np.asarray(d["aa_index"]).astype(np.int64).reshape(-1)
            n = aa.shape[0]
            onehot = np.zeros((n, 21), dtype=np.float32)
            valid = (aa >= 0) & (aa < 21)
            onehot[np.arange(n)[valid], aa[valid]] = 1.0

            feats = [onehot]
            if "plddt" in d:
                p = np.asarray(d["plddt"], dtype=np.float32).reshape(-1, 1)
                p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
                if float(np.nanmax(p)) > 1.5:
                    p = p / 100.0
                feats.append(p)

            out["x"] = _to_tensor(np.concatenate(feats, axis=1), dtype="float32")
        else:
            raise KeyError(
                "Graph artifact lacks 'node_attr' and 'node_emb'. "
                "Residue legacy fallback requires 'aa_index'."
            )

        out["edge_index"] = _to_tensor(d["edge_index"], dtype="int64")

        # Edge attributes (preferred)
        if "edge_attr" in d:
            out["edge_attr"] = _to_tensor(d["edge_attr"], dtype="float32")

        # Keep raw distances too (useful)
        if "edge_dist" in d:
            out["edge_dist"] = _to_tensor(d["edge_dist"], dtype="float32")

        if "pos" in d:
            out["pos"] = _to_tensor(d["pos"], dtype="float32")

        # Graph-level attributes (important for your tests)
        if "graph_attr" in d:
            out["graph_attr"] = _to_tensor(d["graph_attr"], dtype="float32").reshape(-1)
        if "graph_attr_keys" in d:
            out["graph_attr_keys"] = np.asarray(d["graph_attr_keys"]).astype(str).tolist()

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

        # auto uses node_attr if available, else falls back to onehot
        if mode == "auto":
            if "node_attr" in d:
                out["x"] = _to_tensor(d["node_attr"], dtype="float32")
                mode = "none"
            else:
                mode = "onehot+plddt" if "plddt" in d else "onehot"

        # Build point-wise x if requested
        if mode != "none":
            if "aa_index" not in d:
                raise KeyError(
                    "cloud_x_mode requested onehot features but artifact has no 'aa_index'. "
                    "Use cloud_x_mode='auto' with 'node_attr', or set cloud_x_mode='none'."
                )

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
                np.nan_to_num(np.asarray(d["plddt"], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
                dtype="float32",
            )
        if "bfactor_ca" in d:
            out["bfactor_ca"] = _to_tensor(
                np.nan_to_num(np.asarray(d["bfactor_ca"], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
                dtype="float32",
            )

        # Graph-level attrs are still useful on cloud artifacts too
        if "graph_attr" in d:
            out["graph_attr"] = _to_tensor(d["graph_attr"], dtype="float32").reshape(-1)
        if "graph_attr_keys" in d:
            out["graph_attr_keys"] = np.asarray(d["graph_attr_keys"]).astype(str).tolist()

    # --- Attach optional debug metadata (python lists)
    for k in ("residue_ids", "resname", "atom_name"):
        if k in d:
            out[k] = np.asarray(d[k]).astype(str).tolist()

    # more optional atom-level debug (keep as list, not tensor)
    for k in ("seq_pos_residue", "aa_index_residue"):
        if k in d:
            try:
                out[k] = np.asarray(d[k]).reshape(-1).tolist()
            except Exception:
                pass

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
