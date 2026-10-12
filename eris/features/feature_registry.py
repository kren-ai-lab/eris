"""Feature specification dataclasses and the global feature registries."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

ParamsFn = Callable[[dict[str, Any]], dict[str, Any]]


class VoxelFillFn(Protocol):
    """Voxel fill callback, called with keyword arguments."""

    def __call__(
        self,
        *,
        grid: np.ndarray,
        counts: np.ndarray | None,
        ch_start: int,
        data: dict[str, Any],
        ctx: dict[str, Any],
        params: dict[str, Any],
    ) -> int:
        """Fill channels from ``ch_start`` and return the next channel index."""
        ...


def _normalize_feature_params(
    token: str,
    params: dict[str, Any],
    params_fn: ParamsFn | None,
    *,
    allow_unvalidated: bool = False,
) -> dict[str, Any]:
    if params_fn is None:
        if params and not allow_unvalidated:
            raise ValueError(
                f"Feature '{token}' does not support parameters: {sorted(str(param) for param in params)}"
            )
        return dict(params)

    normalized = params_fn(dict(params))
    if not isinstance(normalized, dict):
        raise TypeError(f"Parameter normalizer for feature '{token}' must return a dict")
    return normalized


@dataclass(frozen=True)
class FeatureSpec:
    """Registered node or edge feature and its callbacks."""

    token: str
    order: int
    requires: set[str]
    fn: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], np.ndarray]  # (data, ctx, params) -> (N,F)
    keys_fn: Callable[
        [dict[str, Any], dict[str, Any], dict[str, Any]], list[str]
    ]  # (data, ctx, params) -> keys
    params_fn: ParamsFn | None = None
    fingerprint_params_fn: ParamsFn | None = None

    def normalize_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize user parameters."""
        return _normalize_feature_params(self.token, params, self.params_fn)

    def project_fingerprint_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Project normalized parameters onto the fields that affect cache identity."""
        if self.fingerprint_params_fn is None:
            return dict(params)
        projected = self.fingerprint_params_fn(dict(params))
        if not isinstance(projected, dict):
            raise TypeError(f"Fingerprint projection for feature '{self.token}' must return a dict")
        return projected


@dataclass(frozen=True)
class VoxelFeatureSpec:
    """Registered voxel-grid feature and its callbacks."""

    token: str
    order: int
    requires: set[str]
    n_channels_fn: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], int]
    fill_fn: VoxelFillFn
    keys_fn: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], list[str]]
    params_fn: ParamsFn | None = None

    def normalize_params(
        self,
        params: dict[str, Any],
        *,
        allow_unvalidated: bool = False,
    ) -> dict[str, Any]:
        """Validate and normalize user parameters."""
        return _normalize_feature_params(
            self.token,
            params,
            self.params_fn,
            allow_unvalidated=allow_unvalidated,
        )


RESIDUE_NODE_REGISTRY: dict[str, FeatureSpec] = {}
ATOM_NODE_REGISTRY: dict[str, FeatureSpec] = {}
GRAPH_EDGE_REGISTRY: dict[str, FeatureSpec] = {}
GRID_VOXEL_REGISTRY: dict[str, VoxelFeatureSpec] = {}


def _register(registry: dict[str, Any], spec: Any) -> Any:
    if spec.token in registry:
        raise ValueError(f"Duplicate feature token: {spec.token}")
    registry[spec.token] = spec
    return spec


def register_residue_node_feature(spec: FeatureSpec) -> FeatureSpec:
    """Register a residue node feature."""
    return _register(RESIDUE_NODE_REGISTRY, spec)


def register_atom_node_feature(spec: FeatureSpec) -> FeatureSpec:
    """Register an atom node feature."""
    return _register(ATOM_NODE_REGISTRY, spec)


def register_graph_edge_feature(spec: FeatureSpec) -> FeatureSpec:
    """Register a graph edge feature."""
    return _register(GRAPH_EDGE_REGISTRY, spec)


def register_grid_voxel_feature(spec: VoxelFeatureSpec) -> VoxelFeatureSpec:
    """Register a grid voxel feature."""
    return _register(GRID_VOXEL_REGISTRY, spec)
