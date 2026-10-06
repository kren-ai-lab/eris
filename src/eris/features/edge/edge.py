from __future__ import annotations

from typing import List, Optional, Tuple, Union
import numpy as np

from eris.features.utils import parse_mode
from eris.features.feature_builders import build_from_registry
from eris.features.feature_registry import GRAPH_EDGE_REGISTRY


def build_edge_attr(
    edge_index: np.ndarray,
    edge_dist: np.ndarray,
    *,
    mode: str = "dist+rbf+seq+local",
    rbf_k: int = 16,
    rbf_dmax: float = 20.0,
    seq_pos: Optional[np.ndarray] = None,
    chain_id: Optional[np.ndarray] = None,
    backbone_n: Optional[np.ndarray] = None,
    backbone_ca: Optional[np.ndarray] = None,
    backbone_c: Optional[np.ndarray] = None,
    backbone_n_mask: Optional[np.ndarray] = None,
    backbone_ca_mask: Optional[np.ndarray] = None,
    backbone_c_mask: Optional[np.ndarray] = None,
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
      - same_chain / inter_chain
      - relative_direction (residue graphs only)
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
    src = ei[0]
    dst = ei[1]

    needs_seq_sep = ("seq" in token_names) or ("local" in token_names)
    needs_chain_relation = ("same_chain" in token_names) or ("inter_chain" in token_names)
    chains = None
    if chain_id is not None:
        chains = np.asarray(chain_id).reshape(-1)
        if ei.size and ei.max() >= chains.shape[0]:
            raise ValueError("chain_id must align with edge_index node indices")
    if needs_chain_relation and chains is None:
        raise ValueError("same_chain and inter_chain features require aligned chain_id")

    if needs_seq_sep:
        if seq_pos is None:
            seq_sep = np.zeros((E,), dtype=np.float32)
        else:
            sp = np.asarray(seq_pos, dtype=np.int64).reshape(-1)
            seq_sep = np.abs(sp[src] - sp[dst]).astype(np.float32)
            if chains is not None:
                if chains.shape[0] != sp.shape[0]:
                    raise ValueError("chain_id must align with seq_pos")
                seq_sep[chains[src] != chains[dst]] = -1.0

    data = {"ei": ei, "dist": dist, "E": E}
    for name, value in (
        ("backbone_n", backbone_n),
        ("backbone_ca", backbone_ca),
        ("backbone_c", backbone_c),
        ("backbone_n_mask", backbone_n_mask),
        ("backbone_ca_mask", backbone_ca_mask),
        ("backbone_c_mask", backbone_c_mask),
    ):
        if value is not None:
            data[name] = np.asarray(value)

    ctx = {
        "rbf_k": rbf_k,
        "rbf_dmax": rbf_dmax,
        "local_cutoff": local_cutoff,
        "seq_pos": seq_pos,
    }
    if needs_seq_sep:
        ctx["seq_sep"] = seq_sep
    if needs_chain_relation:
        ctx["same_chain"] = chains[src] == chains[dst]

    return build_from_registry(
        data=data, mode=mode, registry=GRAPH_EDGE_REGISTRY, feat_ctx=ctx, return_keys=return_keys
    )
