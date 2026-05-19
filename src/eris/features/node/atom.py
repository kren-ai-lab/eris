import numpy as np

from typing import List, Optional, Tuple, Union

from eris.features.utils import parse_mode, onehot, safe_float, ensure_1d

# -----------------------------
# Fixed dims (must be stable!)
# -----------------------------
ELEM_DIM = 17   # must match io.py _ELEM_VOCAB length
ROLE_DIM = 6    # roles 0..5 in io.py _atom_role


def _onehot_fixed(idx: np.ndarray, dim: int) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    out = np.zeros((idx.shape[0], dim), dtype=np.float32)
    m = (idx >= 0) & (idx < dim)
    out[np.arange(idx.shape[0])[m], idx[m]] = 1.0
    return out

def build_node_attr(
    *,
    elem_index: np.ndarray,
    atom_role: np.ndarray,
    bfactor: np.ndarray,
    occ: np.ndarray,
    bfactor_norm: bool = True,
    mode: str = "elem+role+bfactor+occ",
    return_keys: bool = False,
    missing: str = "zero",
) -> np.ndarray:
    """
    node_attr: [onehot(elem,17), onehot(role,6), bfactor(1), occ(1)] => (N,25)
    """
    elem_index = np.asarray(elem_index, dtype=np.int64).reshape(-1)
    atom_role = np.asarray(atom_role, dtype=np.int64).reshape(-1)
    n = int(elem_index.shape[0])
    if n == 0:
        x = np.zeros((0, 0), dtype=np.float32)
        return (x, []) if return_keys else x

    toks = parse_mode(mode)
    feats: List[np.ndarray] = []
    keys: List[str] = []

    if "elem" in toks:
        e = _onehot_fixed(elem_index, ELEM_DIM)            # (N,17)
        feats.append(e.astype(np.float32))
        keys.extend([f"elem.{i}" for i in range(ELEM_DIM)])
    
    if "role" in toks:
        r = _onehot_fixed(atom_role, ROLE_DIM)            # (N,6)
        feats.append(r.astype(np.float32))
        keys.extend([f"role.{i}" for i in range(ROLE_DIM)])

    if "bfactor" in toks:
        if bfactor is None:
            v = np.full((n, 1), np.nan if missing == "nan" else 0.0, dtype=np.float32)
        else:
            v = safe_float(bfactor, missing=missing).reshape(-1, 1)
            if missing != "nan" and bfactor_norm and float(np.nanmax(v)) > 1.5:
                v = v / 100.0
            if missing != "nan":
                v = np.clip(v, 0.0, 1.0)
        feats.append(v.astype(np.float32))
        keys.extend(["bfactor"])

    if "occ" in toks:
        oc = np.nan_to_num(np.asarray(occ, dtype=np.float32), nan=0.0).reshape(-1, 1)
        feats.append(oc.astype(np.float32))
        keys.extend(["occ"])

    if feats:
        x = np.concatenate(feats, axis=1).astype(np.float32)
    else:
        x = np.zeros((n, 0), dtype=np.float32)

    return (x, keys) if return_keys else x