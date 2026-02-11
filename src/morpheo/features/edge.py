from __future__ import annotations

from typing import Optional, Literal
import numpy as np

EdgeMode = Literal[
    "dist",
    "dist+rbf",
    "dist+rbf+seq",
    "dist+rbf+seq+local",
]

def rbf(dist: np.ndarray, *, k: int = 16, dmax: float = 20.0) -> np.ndarray:
    """
    dist: (E,) -> (E,k)
    Simple Gaussian RBF centers uniformly spaced in [0, dmax].
    """
    d = np.asarray(dist, dtype=np.float32).reshape(-1, 1)
    centers = np.linspace(0.0, float(dmax), int(k), dtype=np.float32).reshape(1, -1)
    # width so that basis overlap moderately
    step = float(dmax) / max(int(k) - 1, 1)
    gamma = 1.0 / max(step * step, 1e-6)
    return np.exp(-gamma * (d - centers) ** 2).astype(np.float32)

def build_edge_attr(
    edge_index: np.ndarray,
    edge_dist: np.ndarray,
    *,
    mode: EdgeMode = "dist+rbf+seq+local",
    rbf_k: int = 16,
    rbf_dmax: float = 20.0,
    seq_pos: Optional[np.ndarray] = None,
    local_cutoff: int = 4,
) -> np.ndarray:
    """
    Deterministic edge features for residue graphs.

    seq_pos:
      (N,) integer position along the chain (0..N-1). If not provided, seq features become zeros.
    """
    ei = np.asarray(edge_index, dtype=np.int64)
    if ei.shape[0] != 2:
        raise ValueError(f"edge_index must be (2,E), got {ei.shape}")
    dist = np.asarray(edge_dist, dtype=np.float32).reshape(-1)

    feats = [dist.reshape(-1, 1).astype(np.float32)]

    if "rbf" in mode:
        feats.append(rbf(dist, k=rbf_k, dmax=rbf_dmax))

    if "seq" in mode:
        if seq_pos is None:
            seq_sep = np.zeros((dist.shape[0], 1), dtype=np.float32)
        else:
            sp = np.asarray(seq_pos, dtype=np.int64).reshape(-1)
            i = sp[ei[0]]
            j = sp[ei[1]]
            sep = np.abs(i - j).astype(np.float32)
            seq_sep = sep.reshape(-1, 1)
        feats.append(seq_sep)

    if "local" in mode:
        if seq_pos is None:
            local = np.zeros((dist.shape[0], 1), dtype=np.float32)
        else:
            sp = np.asarray(seq_pos, dtype=np.int64).reshape(-1)
            i = sp[ei[0]]
            j = sp[ei[1]]
            sep = np.abs(i - j)
            local = (sep <= int(local_cutoff)).astype(np.float32).reshape(-1, 1)
        feats.append(local)

    return np.concatenate(feats, axis=1).astype(np.float32)
