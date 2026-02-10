from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np

from morpheo.core.types import StructureArtifact


XMode = Literal[
    "aa_index",
    "onehot",
    "aa_index+plddt",
    "onehot+plddt",
    "aa_index+bfactor",
    "onehot+bfactor",
    "aa_index+plddt+emb",
    "onehot+plddt+emb",
]


def _onehot_from_aa_index(aa_index: np.ndarray, num_classes: int = 21) -> np.ndarray:
    aa_index = np.asarray(aa_index).astype(np.int64).reshape(-1)
    out = np.zeros((aa_index.shape[0], num_classes), dtype=np.float32)
    valid = (aa_index >= 0) & (aa_index < num_classes)
    out[np.arange(aa_index.shape[0])[valid], aa_index[valid]] = 1.0
    return out


def _rbf(dist: np.ndarray, num_kernels: int = 16, dmin: float = 0.0, dmax: float = 20.0) -> np.ndarray:
    """
    RBF encoding for distances.

    dist: (E,)
    returns: (E, K)
    """
    dist = np.asarray(dist, dtype=np.float32).reshape(-1, 1)
    centers = np.linspace(dmin, dmax, num_kernels, dtype=np.float32).reshape(1, -1)
    # stable gamma
    step = float(centers[0, 1] - centers[0, 0]) if num_kernels > 1 else 1.0
    gamma = 1.0 / (step * step + 1e-8)
    return np.exp(-gamma * (dist - centers) ** 2).astype(np.float32)


def _get_vector(d: dict, key: str, n: int, fill_value: float = np.nan) -> np.ndarray:
    """
    Fetch vector (N,) from artifact; if missing, returns fill.
    """
    if key not in d:
        return np.full((n,), fill_value, dtype=np.float32)
    v = np.asarray(d[key])
    if v.ndim != 1 or v.shape[0] != n:
        raise ValueError(f"'{key}' must have shape (N,), got {v.shape}")
    return v.astype(np.float32)


def _get_matrix(d: dict, key: str, n: int) -> np.ndarray:
    """
    Fetch matrix (N,D) from artifact; raises if missing or mismatch.
    """
    if key not in d:
        raise KeyError(f"Artifact missing '{key}'.")
    m = np.asarray(d[key])
    if m.ndim != 2 or m.shape[0] != n:
        raise ValueError(f"'{key}' must have shape (N,D) with N={n}, got {m.shape}")
    return m.astype(np.float32)


def to_pyg(
    artifact: StructureArtifact,
    x_mode: XMode = "onehot+plddt",
    rbf_k: int = 16,
    rbf_dmax: float = 20.0,
):
    """
    Convert Morpheo artifact -> torch_geometric.data.Data.

    Requires torch + torch_geometric installed.
    """
    try:
        import torch
        from torch_geometric.data import Data
    except Exception as e:
        raise ImportError("PyG export requires torch + torch_geometric installed.") from e

    d = artifact.data

    # --- Required basics
    if "pos" not in d:
        raise KeyError("Artifact missing 'pos'.")
    if "edge_index" not in d:
        raise KeyError("Artifact missing 'edge_index'.")

    pos = np.asarray(d["pos"], dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"'pos' must have shape (N,3), got {pos.shape}")
    n = pos.shape[0]

    edge_index = np.asarray(d["edge_index"], dtype=np.int64)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"'edge_index' must have shape (2,E), got {edge_index.shape}")

    # --- Node features
    if "aa_index" not in d:
        raise KeyError("Artifact missing 'aa_index' required to build x.")
    aa_index = np.asarray(d["aa_index"]).reshape(-1)
    if aa_index.shape[0] != n:
        raise ValueError(f"'aa_index' must have shape (N,), N={n}, got {aa_index.shape}")
    aa_index = aa_index.astype(np.int64)

    x_parts: list[np.ndarray] = []

    if x_mode.startswith("onehot"):
        x_parts.append(_onehot_from_aa_index(aa_index, num_classes=21))
    else:
        x_parts.append(aa_index.astype(np.float32).reshape(-1, 1))

    if "+plddt" in x_mode:
        plddt = _get_vector(d, "plddt", n, fill_value=np.nan).reshape(-1, 1)
        # si viene NaN (PDB experimental), queda NaN y es responsabilidad del modelo/loader
        x_parts.append(plddt)

    if "+bfactor" in x_mode:
        bfac = _get_vector(d, "bfactor_ca", n, fill_value=np.nan).reshape(-1, 1)
        x_parts.append(bfac)

    if x_mode.endswith("+emb"):
        node_emb = _get_matrix(d, "node_emb", n)
        x_parts.append(node_emb)

    x = np.concatenate(x_parts, axis=1).astype(np.float32)

    # --- Edge attributes (RBF on edge_dist if available)
    edge_attr = None
    if "edge_dist" in d:
        edge_dist = np.asarray(d["edge_dist"], dtype=np.float32).reshape(-1)
        if edge_dist.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"'edge_dist' must have length E={edge_index.shape[1]}, got {edge_dist.shape}"
            )
        edge_attr = _rbf(edge_dist, num_kernels=int(rbf_k), dmax=float(rbf_dmax))
        edge_attr = torch.from_numpy(edge_attr).float()

    data = Data(
        x=torch.from_numpy(x).float(),
        pos=torch.from_numpy(pos).float(),
        edge_index=torch.from_numpy(edge_index).long(),
        edge_attr=edge_attr,
    )

    # Optional: keep ids if you want (handy for debugging)
    # NOTE: PyG expects tensors; store as python attr.
    if "residue_ids" in d:
        data.residue_ids = np.asarray(d["residue_ids"]).astype(str).tolist()
    if "resname" in d:
        data.resname = np.asarray(d["resname"]).astype(str).tolist()

    return data


def load_pyg(
    artifact_dir: str | Path,
    x_mode: XMode = "onehot+plddt",
    rbf_k: int = 16,
    rbf_dmax: float = 20.0,
):
    artifact = StructureArtifact.load(Path(artifact_dir))
    return to_pyg(artifact, x_mode=x_mode, rbf_k=rbf_k, rbf_dmax=rbf_dmax)
