from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np

from morpheo.features.amino_acids import N_AA, get_physchem


NodeMode = Literal[
    "aa_index",
    "onehot",
    "physchem",
    "onehot+physchem",
    "onehot+physchem+plddt",
    "onehot+physchem+plddt+bfactor",
]

def onehot_aa(aa_index: np.ndarray) -> np.ndarray:
    aa = np.asarray(aa_index, dtype=np.int64).reshape(-1)
    aa = np.clip(aa, 0, N_AA - 1)
    n = aa.shape[0]
    x = np.zeros((n, N_AA), dtype=np.float32)
    x[np.arange(n), aa] = 1.0
    return x

def _norm01(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1, 1)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    # if looks like 0..100, scale
    if float(np.max(v)) > 1.5:
        v = v / 100.0
    # clamp
    v = np.clip(v, 0.0, 1.0)
    return v

def build_node_attr(
    aa_index: np.ndarray,
    *,
    plddt: Optional[np.ndarray] = None,
    bfactor_ca: Optional[np.ndarray] = None,
    mode: NodeMode = "onehot+physchem+plddt",
) -> np.ndarray:
    """
    Deterministic node features from residue identity + optional confidence signals.

    Returns float32 array (N,F). Always deterministic, no external models.
    """
    aa_index = np.asarray(aa_index, dtype=np.int64).reshape(-1)

    feats = []

    if mode == "aa_index":
        return aa_index.astype(np.int64)

    if "onehot" in mode:
        feats.append(onehot_aa(aa_index))

    if "physchem" in mode:
        feats.append(get_physchem(aa_index))

    if "plddt" in mode:
        if plddt is None:
            # keep deterministic shape by adding zeros
            feats.append(np.zeros((aa_index.shape[0], 1), dtype=np.float32))
        else:
            feats.append(_norm01(plddt))

    if "bfactor" in mode:
        if bfactor_ca is None:
            feats.append(np.zeros((aa_index.shape[0], 1), dtype=np.float32))
        else:
            feats.append(_norm01(bfactor_ca))

    if not feats:
        return np.zeros((aa_index.shape[0], 0), dtype=np.float32)

    return np.concatenate(feats, axis=1).astype(np.float32)
