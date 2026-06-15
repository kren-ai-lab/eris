from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Set, Optional, Tuple
import numpy as np

from eris.features.utils import parse_mode

@dataclass(frozen=True)
class FeatureSpec:
    token: str
    order: int
    requires: Set[str]
    fn: Callable[[Dict[str, Any], Dict[str, Any]], np.ndarray]      # (data, ctx) -> (N,F)
    keys_fn: Callable[[Dict[str, Any], Dict[str, Any]], List[str]]  # (data, ctx) -> keys


RESIDUE_NODE_REGISTRY: Dict[str, FeatureSpec] = {}
def register_residue_node_feature(spec: FeatureSpec):
    if spec.token in RESIDUE_NODE_REGISTRY:
        raise ValueError(f"Duplicate feature token: {spec.token}")
    RESIDUE_NODE_REGISTRY[spec.token] = spec
    return spec


ATOM_NODE_REGISTRY: Dict[str, FeatureSpec] = {}
def register_atom_node_feature(spec: FeatureSpec):
    if spec.token in ATOM_NODE_REGISTRY:
        raise ValueError(f"Duplicate feature token: {spec.token}")
    ATOM_NODE_REGISTRY[spec.token] = spec
    return spec


GRAPH_EDGE_REGISTRY: Dict[str, FeatureSpec] = {}
def register_graph_edge_feature(spec: FeatureSpec):
    if spec.token in GRAPH_EDGE_REGISTRY:
        raise ValueError(f"Duplicate feature token: {spec.token}")
    GRAPH_EDGE_REGISTRY[spec.token] = spec
    return spec


# Can be used for any feature.
# Implement GRAPH_STATS/etc if necessary.
#   Remember to change the corresponding encoder and core/represesentation
#   to pass keys to ctx (in encoder) and then use ctx to
#   build the feature keys (in corresponding representation)
#
# Example:
# GRAPH_STATS: Dict[str, FeatureSpec] = {}
# def register_edge_graph_feature(spec: FeatureSpec):
#     if spec.token in GRAPH_STATS:
#         raise ValueError(f"Duplicate feature token: {spec.token}")
#     GRAPH_STATS[spec.token] = spec
#     return spec


def build_from_registry(
    data: Dict[str, Any],
    *,
    mode: str,
    registry: Dict[str, FeatureSpec],
    ctx: Optional[Dict[str, Any]] = None,
    return_keys: bool = False,
):
    ctx = ctx or {}

    # -------------------------
    # Parse tokens
    # -------------------------
    toks = parse_mode(mode)

    # ---- detect unknown tokens
    unknown = [t[0] for t in toks if t[0] not in registry]
    if unknown:
        raise ValueError(f"Unknown feature tokens: {unknown}")

    # ---- select + sort
    selected = [(registry[t[0]], t[1]) for t in toks]
    selected.sort(key=lambda x: x[0].order)

    feats: List[np.ndarray] = []
    keys: List[str] = []

    N: Optional[int] = None  # number of nodes

    # -------------------------
    # Validate requirements
    # -------------------------
    for spec, params in selected:
        missing = [k for k in spec.requires if k not in data and k not in ctx]
        if missing:
            raise KeyError(f"Feature '{spec.token}' requires missing fields: {missing}")

    # -------------------------
    # Compute features
    # -------------------------
    for spec, params in selected:
        f = spec.fn(data, ctx, params)

        # ---- validate output
        if not isinstance(f, np.ndarray):
            raise TypeError(f"Feature '{spec.token}' must return np.ndarray")

        # if f.ndim != 2:
        #     raise ValueError(f"Feature '{spec.token}' must return 2D array, got shape {f.shape}")

        if N is None:
            N = f.shape[0]
        else:
            if f.shape[0] != N:
                raise ValueError(
                    f"Feature '{spec.token}' has mismatched rows: expected {N}, got {f.shape[0]}"
                )

        feats.append(f.astype(np.float32))

        k = spec.keys_fn(data, ctx, params)
        if len(k) != f.shape[1]:
            raise ValueError(
                f"Feature '{spec.token}' keys mismatch: "
                f"{len(k)} keys for {f.shape[1]} columns"
            )
        keys.extend(k)

    # -------------------------
    # Build final tensor
    # -------------------------
    if feats:
        x = np.concatenate(feats, axis=1).astype(np.float32)
    else:
        # no features selected → still need correct N
        if N is None:
            # fallback: try infer N from any array in data
            for v in data.values():
                if isinstance(v, np.ndarray) and v.ndim > 0:
                    N = v.shape[0]
                    break

        if N is None:
            raise ValueError("Cannot infer number of nodes (N) from data.")

        x = np.zeros((N, 0), dtype=np.float32)

    return (x, keys) if return_keys else x


