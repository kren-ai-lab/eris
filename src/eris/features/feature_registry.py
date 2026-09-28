from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Set, Optional
import numpy as np


ParamsFn = Callable[[Dict[str, Any]], Dict[str, Any]]


def _normalize_feature_params(
    token: str,
    params: Dict[str, Any],
    params_fn: Optional[ParamsFn],
    *,
    allow_unvalidated: bool = False,
) -> Dict[str, Any]:
    if params_fn is None:
        if params and not allow_unvalidated:
            raise ValueError(
                f"Feature '{token}' does not support parameters: "
                f"{sorted(str(param) for param in params)}"
            )
        return dict(params)

    normalized = params_fn(dict(params))
    if not isinstance(normalized, dict):
        raise TypeError(f"Parameter normalizer for feature '{token}' must return a dict")
    return normalized


@dataclass(frozen=True)
class FeatureSpec:
    token: str
    order: int
    requires: Set[str]
    fn: Callable[
        [Dict[str, Any], Dict[str, Any], Dict[str, Any]], np.ndarray
    ]  # (data, ctx, params) -> (N,F)
    keys_fn: Callable[
        [Dict[str, Any], Dict[str, Any], Dict[str, Any]], List[str]
    ]  # (data, ctx, params) -> keys
    params_fn: Optional[ParamsFn] = None

    def normalize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return _normalize_feature_params(self.token, params, self.params_fn)


@dataclass(frozen=True)
class VoxelFeatureSpec:
    token: str
    order: int
    requires: Set[str]
    n_channels_fn: Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any]], int]
    fill_fn: Callable[
        [
            np.ndarray,
            Optional[np.ndarray],
            int,
            Dict[str, Any],
            Dict[str, Any],
            Dict[str, Any],
            str,
        ],
        int,
    ]
    keys_fn: Callable[[Dict[str, Any], Dict[str, Any], Dict[str, Any]], List[str]]
    params_fn: Optional[ParamsFn] = None

    def normalize_params(
        self,
        params: Dict[str, Any],
        *,
        allow_unvalidated: bool = False,
    ) -> Dict[str, Any]:
        return _normalize_feature_params(
            self.token,
            params,
            self.params_fn,
            allow_unvalidated=allow_unvalidated,
        )


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


GRID_VOXEL_REGISTRY: Dict[str, VoxelFeatureSpec] = {}


def register_grid_voxel_feature(spec: VoxelFeatureSpec):
    if spec.token in GRID_VOXEL_REGISTRY:
        raise ValueError(f"Duplicate feature token: {spec.token}")
    GRID_VOXEL_REGISTRY[spec.token] = spec
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
