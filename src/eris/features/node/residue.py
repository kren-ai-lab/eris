from __future__ import annotations

from typing import List, Optional, Tuple, Union
import numpy as np

from eris.features.utils import ensure_1d
from eris.features.feature_builders import build_from_registry
from eris.features.feature_registry import RESIDUE_NODE_REGISTRY


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
    mode: str = "onehot+aa_index+physchem+aaindex_basic+plddt_ca+bfactor_ca+flags",
    missing: str = "zero",
    return_keys: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
        
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

    if has_ca is None:
        has_ca = np.ones((n,), dtype=bool)
    else:
        has_ca = ensure_1d(has_ca, n, "has_ca").astype(bool)

    data = {
        "aa_index": aa_index,
        "plddt_ca": plddt_ca,
        "bfactor_ca": bfactor_ca,
        "has_ca": has_ca,
        "n": n
    }

    feat_ctx = {
        "missing": missing,
    }

    return build_from_registry(data=data,
                               mode = mode,
                               registry = RESIDUE_NODE_REGISTRY,
                               feat_ctx = feat_ctx,
                               return_keys = return_keys
                               )


