from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from morpheo.features.constants import (
    AA_VOCAB,
    UNKNOWN_INDEX,
    PHYSchem,
    PHYSchem_KEYS,
    AAINDEX_BASIC_KEYS,
)
from morpheo.features.utils import parse_mode, onehot, safe_float, ensure_1d


def build_node_attr(
    aa_index: np.ndarray,
    *,
    plddt_ca: Optional[np.ndarray] = None,
    bfactor_ca: Optional[np.ndarray] = None,
    has_ca: Optional[np.ndarray] = None,
    mode: str = "aa_onehot+aa_index+physchem+aaindex_basic+plddt_ca+bfactor_ca+flags",
    missing: str = "zero",
) -> Tuple[np.ndarray, List[str]]:
    """
    Build deterministic residue node features.

    Parameters
    ----------
    aa_index : (N,) int
        0..20 where 20 is unknown (X). (Your pipeline already uses this.)
    plddt_ca : (N,) float, optional
    bfactor_ca : (N,) float, optional
    has_ca : (N,) bool, optional
        If None, assumes True for all residues in bundle (since missing CA are skipped upstream).
    mode : str
        '+'-separated tokens. Order does not matter; output order is canonical.
        Supported: aa_onehot, aa_index, physchem, aaindex_basic, plddt_ca, bfactor_ca, flags
    missing : str
        'zero' (default) -> NaNs replaced by 0.0
        'nan' -> keep NaNs

    Returns
    -------
    node_attr : (N,F) float32
    keys : List[str]
        Feature keys aligned to columns in node_attr.
    """
    aa_index = np.asarray(aa_index, dtype=np.int64).reshape(-1)
    n = int(aa_index.shape[0])
    if n == 0:
        raise ValueError("aa_index is empty")

    tokens = parse_mode(mode)

    if has_ca is None:
        has_ca_arr = np.ones((n,), dtype=np.float32)
    else:
        hc = ensure_1d(has_ca, n, "has_ca").astype(bool)
        has_ca_arr = hc.astype(np.float32)

    feats: List[np.ndarray] = []
    keys: List[str] = []

    # --- canonical output order (do NOT change after release)
    # 1) aa_onehot
    if "aa_onehot" in tokens:
        oh = onehot(aa_index, num_classes=len(AA_VOCAB))
        feats.append(oh.astype(np.float32))
        keys.extend([f"aa_onehot.{aa}" for aa in AA_VOCAB])

    # 2) aa_index
    if "aa_index" in tokens:
        feats.append(aa_index.astype(np.float32).reshape(n, 1))
        keys.append("aa_index")

    # 3) physchem (8 fixed dims)
    if "physchem" in tokens:
        # map aa_index -> AA1 -> physchem vector
        # aa_index is assumed aligned to AA_VOCAB ordering elsewhere; if not, unknowns still safe.
        # If your AA1 mapping differs, you can pass aa1 array instead later.
        aa1 = np.array([AA_VOCAB[i] if 0 <= i < len(AA_VOCAB) else "X" for i in aa_index], dtype=np.str_)
        mat = np.zeros((n, len(PHYSchem_KEYS)), dtype=np.float32)
        for i, a in enumerate(aa1):
            mat[i, :] = np.asarray(PHYSchem.get(str(a), PHYSchem["X"]), dtype=np.float32)
        feats.append(mat)
        keys.extend([f"physchem.{k}" for k in PHYSchem_KEYS])

    # 4) aaindex_basic (6 fixed dims: subset)
    if "aaindex_basic" in tokens:
        aa1 = np.array([AA_VOCAB[i] if 0 <= i < len(AA_VOCAB) else "X" for i in aa_index], dtype=np.str_)
        # reuse PHYSchem source: [hydrophobicity_kd, volume, polarity, charge, aromatic, aliphatic]
        mat = np.zeros((n, 6), dtype=np.float32)
        for i, a in enumerate(aa1):
            v = np.asarray(PHYSchem.get(str(a), PHYSchem["X"]), dtype=np.float32)
            mat[i, :] = v[[0, 1, 2, 3, 4, 5]]
        feats.append(mat)
        keys.extend(AAINDEX_BASIC_KEYS)

    # 5) plddt_ca
    if "plddt_ca" in tokens:
        if plddt_ca is None:
            col = np.zeros((n, 1), dtype=np.float32)
        else:
            col = safe_float(ensure_1d(plddt_ca, n, "plddt_ca"), missing=missing).reshape(n, 1)
        feats.append(col.astype(np.float32))
        keys.append("plddt_ca")

    # 6) bfactor_ca
    if "bfactor_ca" in tokens:
        if bfactor_ca is None:
            col = np.zeros((n, 1), dtype=np.float32)
        else:
            col = safe_float(ensure_1d(bfactor_ca, n, "bfactor_ca"), missing=missing).reshape(n, 1)
        feats.append(col.astype(np.float32))
        keys.append("bfactor_ca")

    # 7) flags
    if "flags" in tokens:
        is_gly = (aa_index == _aa_idx("G")).astype(np.float32).reshape(n, 1)
        is_pro = (aa_index == _aa_idx("P")).astype(np.float32).reshape(n, 1)
        is_unk = (aa_index == UNKNOWN_INDEX).astype(np.float32).reshape(n, 1)
        feats.extend([is_gly, is_pro, is_unk, has_ca_arr.reshape(n, 1)])
        keys.extend(["is_gly", "is_pro", "is_unk", "has_ca"])

    if not feats:
        # return empty but valid
        return np.zeros((n, 0), dtype=np.float32), []

    out = np.concatenate(feats, axis=1).astype(np.float32)
    return out, keys


def _aa_idx(aa1: str) -> int:
    aa1 = (aa1 or "X").strip().upper()
    try:
        return AA_VOCAB.index(aa1)
    except ValueError:
        return UNKNOWN_INDEX
