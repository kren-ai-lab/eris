from __future__ import annotations

from typing import List, Optional, Tuple, Union
import numpy as np

from eris.features.utils import parse_mode
from eris.features.feature_registry import build_from_registry, GRAPH_EDGE_REGISTRY


def build_edge_attr(
    edge_index: np.ndarray,
    edge_dist: np.ndarray,
    *,
    mode: str = "dist+rbf+seq+local",
    rbf_k: int = 16,
    rbf_dmax: float = 20.0,
    seq_pos: Optional[np.ndarray] = None,
    local_cutoff: int = 4,
    return_keys: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
    """
    Deterministic edge features with stable keys.
    Tokens:
      - dist
      - rbf
      - seq
      - local   (is_local flag based on seq_sep <= local_cutoff)
    """
    ei = np.asarray(edge_index, dtype=np.int64)
    if ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError(f"edge_index must be (2,E), got {ei.shape}")

    dist = np.asarray(edge_dist, dtype=np.float32).reshape(-1)
    E = int(ei.shape[1])
    if dist.shape[0] != E:
        raise ValueError(f"edge_dist must be (E,), got {dist.shape} for E={E}")

    toks = parse_mode(mode)
    token_names = [tok[0] for tok in toks]
    
    if ("seq_sep" in token_names) or ("local" in token_names):
        if seq_pos is None:
            seq_sep = np.zeros((E,), dtype=np.float32)
        else:
            sp = np.asarray(seq_pos, dtype=np.int64).reshape(-1)
            src = ei[0]
            dst = ei[1]
            seq_sep = np.abs(sp[src] - sp[dst]).astype(np.float32)

    data = {
        "ei": ei,
        "dist": dist,
        "E": E
    }

    ctx = {
        "rbf_k": rbf_k,
        "rbf_dmax": rbf_dmax,
        "local_cutoff": local_cutoff,
        "seq_pos": seq_pos,
        "seq_sep": seq_sep,
    }

    return build_from_registry(data=data,
                               mode = mode,
                               registry = GRAPH_EDGE_REGISTRY,
                               ctx = ctx,
                               return_keys = return_keys
                               )
