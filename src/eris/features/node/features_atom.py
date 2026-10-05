import numpy as np

from typing import Dict, Any, List

from eris.features.utils import parse_mode, onehot, safe_float, ensure_1d
from eris.features.feature_registry import register_atom_node_feature, FeatureSpec

# -----------------------------
# Fixed dims (must be stable!)
# -----------------------------
ELEM_DIM = 17  # must match io.py _ELEM_VOCAB length
ROLE_DIM = 6  # roles 0..5 in io.py _atom_role


def _onehot_fixed(idx: np.ndarray, dim: int) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    out = np.zeros((idx.shape[0], dim), dtype=np.float32)
    m = (idx >= 0) & (idx < dim)
    out[np.arange(idx.shape[0])[m], idx[m]] = 1.0
    return out


# -------------------------
# elem_index
# -------------------------
def elem_index_fn(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
    elem_index = np.asarray(data["elem_index"], dtype=np.int64).reshape(-1)
    e = _onehot_fixed(elem_index, ELEM_DIM)

    return e.astype(np.float32)


def elem_index_keys(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
    return [f"elem.{i}" for i in range(ELEM_DIM)]


register_atom_node_feature(
    FeatureSpec(
        token="elem",
        order=10,
        requires={"elem_index"},
        fn=elem_index_fn,
        keys_fn=elem_index_keys,
    )
)


# -------------------------
# role
# -------------------------
def role_fn(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
    atom_role = np.asarray(data["atom_role"], dtype=np.int64).reshape(-1)
    r = _onehot_fixed(atom_role, ROLE_DIM)

    return r.astype(np.float32)


def role_keys(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
    return [f"role.{i}" for i in range(ROLE_DIM)]


register_atom_node_feature(
    FeatureSpec(
        token="role",
        order=20,
        requires={"atom_role"},
        fn=role_fn,
        keys_fn=role_keys,
    )
)


# -------------------------
# bfactor
# -------------------------


def bfactor_fn(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
    n = data["n"]
    missing = ctx["missing"]

    if data["bfactor"] is None:
        v = np.full((n, 1), np.nan if missing == "nan" else 0.0, dtype=np.float32)
    else:
        v = safe_float(data["bfactor"], missing=missing).reshape(-1, 1)

    return v.astype(np.float32)


def bfactor_keys(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
    return ["bfactor"]


register_atom_node_feature(
    FeatureSpec(
        token="bfactor",
        order=30,
        requires={"missing", "bfactor", "n"},
        fn=bfactor_fn,
        keys_fn=bfactor_keys,
    )
)


# -------------------------
# occ
# -------------------------


def occ_fn(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:

    oc = np.nan_to_num(np.asarray(data["occ"], dtype=np.float32), nan=0.0).reshape(-1, 1)

    return oc.astype(np.float32)


def occ_keys(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
    return ["occ"]


register_atom_node_feature(
    FeatureSpec(
        token="occ",
        order=40,
        requires={"occ"},
        fn=occ_fn,
        keys_fn=occ_keys,
    )
)
