from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

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
    fingerprint_params_fn: Optional[ParamsFn] = None

    def normalize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return _normalize_feature_params(self.token, params, self.params_fn)

    def project_fingerprint_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.fingerprint_params_fn is None:
            return dict(params)
        projected = self.fingerprint_params_fn(dict(params))
        if not isinstance(projected, dict):
            raise TypeError(f"Fingerprint projection for feature '{self.token}' must return a dict")
        return projected


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
ATOM_NODE_REGISTRY: Dict[str, FeatureSpec] = {}
GRAPH_EDGE_REGISTRY: Dict[str, FeatureSpec] = {}
GRID_VOXEL_REGISTRY: Dict[str, VoxelFeatureSpec] = {}


def _register(registry: Dict[str, Any], spec: Any):
    if spec.token in registry:
        raise ValueError(f"Duplicate feature token: {spec.token}")
    registry[spec.token] = spec
    return spec


def register_residue_node_feature(spec: FeatureSpec):
    return _register(RESIDUE_NODE_REGISTRY, spec)


def register_atom_node_feature(spec: FeatureSpec):
    return _register(ATOM_NODE_REGISTRY, spec)


def register_graph_edge_feature(spec: FeatureSpec):
    return _register(GRAPH_EDGE_REGISTRY, spec)


def register_grid_voxel_feature(spec: VoxelFeatureSpec):
    return _register(GRID_VOXEL_REGISTRY, spec)
