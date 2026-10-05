import numpy as np

from typing import List, Optional, Tuple, Union

from eris.features.feature_builders import build_from_registry, validate_feature_mode
from eris.features.feature_registry import ATOM_NODE_REGISTRY


def build_node_attr(
    *,
    elem_index: np.ndarray,
    atom_role: np.ndarray,
    bfactor: np.ndarray,
    occ: np.ndarray,
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
        validate_feature_mode(mode, ATOM_NODE_REGISTRY)
        x = np.zeros((0, 0), dtype=np.float32)
        return (x, []) if return_keys else x

    data = {
        "elem_index": elem_index,
        "atom_role": atom_role,
        "bfactor": bfactor,
        "occ": occ,
        "n": n,
    }

    feat_ctx = {"missing": missing}

    return build_from_registry(
        data=data,
        mode=mode,
        registry=ATOM_NODE_REGISTRY,
        feat_ctx=feat_ctx,
        return_keys=return_keys,
    )
