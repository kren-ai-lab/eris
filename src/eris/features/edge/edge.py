from __future__ import annotations

from typing import List, Optional, Tuple, Union
import numpy as np

from eris.features.utils import parse_mode
from eris.features.rbf import compute_rbf as rbf_encode


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
      - seq_sep
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

    feats: List[np.ndarray] = []
    keys: List[str] = []

    # dist
    if "dist" in toks or "rbf" in toks or "seq_sep" in toks or "local" in toks:
        feats.append(dist.reshape(-1, 1).astype(np.float32))
        keys.append("dist")

    # rbf
    if "rbf" in toks:
        if E == 0:
            r = np.zeros((0, int(rbf_k)), dtype=np.float32)
        else:
            r = rbf_encode(dist, num_kernels=int(rbf_k), dmin=0.0, dmax=float(rbf_dmax))
        feats.append(r.astype(np.float32))
        keys.extend([f"rbf.{i}" for i in range(int(rbf_k))])

    # seq_sep + local flag
    if ("seq_sep" in toks) or ("local" in toks):
        if seq_pos is None:
            seq_sep = np.zeros((E,), dtype=np.float32)
        else:
            sp = np.asarray(seq_pos, dtype=np.int64).reshape(-1)
            src = ei[0]
            dst = ei[1]
            seq_sep = np.abs(sp[src] - sp[dst]).astype(np.float32)

        if "seq_sep" in toks:
            feats.append(seq_sep.reshape(-1, 1).astype(np.float32))
            keys.append("seq_sep")

        if "local" in toks:
            is_local = (seq_sep <= float(local_cutoff)).astype(np.float32).reshape(-1, 1)
            feats.append(is_local.astype(np.float32))
            keys.append(f"flag.is_local_le_{int(local_cutoff)}")

    x = np.concatenate(feats, axis=1).astype(np.float32) if feats else np.zeros((E, 0), dtype=np.float32)
    return (x, keys) if return_keys else x
