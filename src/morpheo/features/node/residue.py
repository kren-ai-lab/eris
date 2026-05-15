from __future__ import annotations

from typing import List, Optional, Tuple, Union
import numpy as np

from morpheo.features.node.constants import (
    AA_VOCAB,
    UNKNOWN_INDEX,
    PHYSchem,
    PHYSchem_KEYS,
    AAINDEX_BASIC_KEYS,
)
from morpheo.features.utils import parse_mode, onehot, safe_float, ensure_1d


# --- canonical ordering (NO CAMBIAR en releases)
_AA_ONEHOT_KEYS = [f"aa_onehot.{aa}" for aa in AA_VOCAB]
_AA_INDEX_KEY = ["aa_index"]
_AAINDEX_BASIC_SUBIDX = [0, 1, 2, 3, 4, 5]  # maps to PHYSchem first 6 dims (kd, vol, pol, charge, arom, aliph)


def _physchem_from_index(aa_index: np.ndarray) -> np.ndarray:
    aa_index = np.asarray(aa_index, dtype=np.int64).reshape(-1)
    out = np.zeros((aa_index.shape[0], len(PHYSchem_KEYS)), dtype=np.float32)
    # build mapping by vocab
    for i, idx in enumerate(aa_index):
        if idx < 0 or idx >= len(AA_VOCAB):
            idx = UNKNOWN_INDEX
        aa = AA_VOCAB[int(idx)]
        out[i] = np.asarray(PHYSchem.get(aa, PHYSchem["X"]), dtype=np.float32)
    return out


def build_node_attr(
    aa_index: np.ndarray,
    *,
    # new names (contract)
    plddt_ca: Optional[np.ndarray] = None,
    bfactor_ca: Optional[np.ndarray] = None,
    has_ca: Optional[np.ndarray] = None,
    # backwards-compatible aliases
    plddt: Optional[np.ndarray] = None,
    bfactor: Optional[np.ndarray] = None,
    # contract
    mode: str = "aa_onehot+aa_index+physchem+aaindex_basic+plddt_ca+bfactor_ca+flags",
    missing: str = "zero",
    return_keys: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
    """
    Deterministic residue node features with stable keys.

    Supported tokens (order in output is canonical):
      - aa_onehot
      - aa_index
      - physchem
      - aaindex_basic
      - plddt_ca
      - bfactor_ca
      - flags   (is_gly, is_pro, is_unk, has_ca)

    Notes
    -----
    - If plddt_ca/bfactor_ca are missing, we emit zeros (or NaN if missing="nan") but keep shape stable.
    - has_ca defaults to True for all residues (your IO skip-missing-CA policy).
    """
    aa_index = np.asarray(aa_index, dtype=np.int64).reshape(-1)
    n = int(aa_index.shape[0])
    if n == 0:
        x = np.zeros((0, 0), dtype=np.float32)
        return (x, []) if return_keys else x

    # aliases
    if plddt_ca is None and plddt is not None:
        plddt_ca = plddt
    if bfactor_ca is None and bfactor is not None:
        bfactor_ca = bfactor

    toks = parse_mode(mode)

    if has_ca is None:
        has_ca = np.ones((n,), dtype=bool)
    else:
        has_ca = ensure_1d(has_ca, n, "has_ca").astype(bool)

    feats: List[np.ndarray] = []
    keys: List[str] = []

    # aa_onehot
    if "aa_onehot" in toks:
        oh = onehot(np.clip(aa_index, 0, len(AA_VOCAB) - 1), num_classes=len(AA_VOCAB))
        feats.append(oh.astype(np.float32))
        keys.extend(_AA_ONEHOT_KEYS)

    # aa_index (float column, stable)
    if "aa_index" in toks:
        feats.append(aa_index.astype(np.float32).reshape(-1, 1))
        keys.extend(_AA_INDEX_KEY)

    # physchem (8)
    if "physchem" in toks:
        pc = _physchem_from_index(aa_index)
        feats.append(pc.astype(np.float32))
        keys.extend([f"physchem.{k}" for k in PHYSchem_KEYS])

    # aaindex_basic (fixed subset)
    if "aaindex_basic" in toks:
        pc = _physchem_from_index(aa_index)  # reuse source of truth
        basic = pc[:, _AAINDEX_BASIC_SUBIDX]
        feats.append(basic.astype(np.float32))
        keys.extend(AAINDEX_BASIC_KEYS)

    # plddt_ca
    if "plddt_ca" in toks:
        if plddt_ca is None:
            v = np.full((n, 1), np.nan if missing == "nan" else 0.0, dtype=np.float32)
        else:
            v = safe_float(plddt_ca, missing=missing).reshape(-1, 1)
            # normalize heuristic (0..100 -> 0..1)
            if missing != "nan" and float(np.nanmax(v)) > 1.5:
                v = v / 100.0
            if missing != "nan":
                v = np.clip(v, 0.0, 1.0)
        feats.append(v.astype(np.float32))
        keys.append("plddt_ca")

    # bfactor_ca
    if "bfactor_ca" in toks:
        if bfactor_ca is None:
            v = np.full((n, 1), np.nan if missing == "nan" else 0.0, dtype=np.float32)
        else:
            v = safe_float(bfactor_ca, missing=missing).reshape(-1, 1)
            if missing != "nan" and float(np.nanmax(v)) > 1.5:
                v = v / 100.0
            if missing != "nan":
                v = np.clip(v, 0.0, 1.0)
        feats.append(v.astype(np.float32))
        keys.append("bfactor_ca")

    # flags
    if "flags" in toks:
        is_gly = (aa_index == int(list(AA_VOCAB).index("G"))).astype(np.float32).reshape(-1, 1)
        is_pro = (aa_index == int(list(AA_VOCAB).index("P"))).astype(np.float32).reshape(-1, 1)
        is_unk = (aa_index == UNKNOWN_INDEX).astype(np.float32).reshape(-1, 1)
        has_ca_f = has_ca.astype(np.float32).reshape(-1, 1)
        feats.append(np.concatenate([is_gly, is_pro, is_unk, has_ca_f], axis=1).astype(np.float32))
        keys.extend(["flag.is_gly", "flag.is_pro", "flag.is_unk", "flag.has_ca"])

    if feats:
        x = np.concatenate(feats, axis=1).astype(np.float32)
    else:
        x = np.zeros((n, 0), dtype=np.float32)

    return (x, keys) if return_keys else x
