"""Registry-driven feature assembly for node, edge and voxel features."""

from typing import Any

import numpy as np

from eris.features.feature_registry import FeatureSpec
from eris.features.utils import parse_mode


def _select_features(
    mode: Any,
    registry: dict[str, Any],
    *,
    kind: str,
    allow_unvalidated_params: bool,
) -> list[tuple[Any, dict[str, Any]]]:
    toks = parse_mode(mode)
    token_names = [token for token, _ in toks]

    unknown = [token for token in token_names if token not in registry]
    if unknown:
        raise ValueError(f"Unknown {kind} tokens: {unknown}")

    seen = set()
    duplicates = []
    for token in token_names:
        if token in seen and token not in duplicates:
            duplicates.append(token)
        seen.add(token)
    if duplicates:
        raise ValueError(f"Duplicate {kind} tokens: {duplicates}")

    selected = []
    for token, params in toks:
        spec = registry[token]
        if allow_unvalidated_params and getattr(spec, "params_fn", None) is None:
            normalized_params = spec.normalize_params(params, allow_unvalidated=True)
        else:
            normalized_params = spec.normalize_params(params)
        selected.append((spec, normalized_params))
    selected.sort(key=lambda item: item[0].order)
    return selected


def validate_feature_mode(mode: Any, registry: dict[str, FeatureSpec]) -> None:
    """Raise if ``mode`` names unknown, duplicate or invalid-parameter tokens."""
    _select_features(
        mode,
        registry,
        kind="feature",
        allow_unvalidated_params=False,
    )


# -------------------------
# Normal registry
# -------------------------


def build_from_registry(
    data: dict[str, Any],
    *,
    mode: Any,
    registry: dict[str, FeatureSpec],
    feat_ctx: dict[str, Any] | None = None,
    return_keys: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[str]]:
    """Build a float32 ``(N, F)`` matrix from registered features, optionally with keys."""
    feat_ctx = feat_ctx or {}

    selected = _select_features(
        mode,
        registry,
        kind="feature",
        allow_unvalidated_params=False,
    )

    feats: list[np.ndarray] = []
    keys: list[str] = []

    if "n" in data:
        expected_rows: int | None = int(data["n"])
    elif "E" in data:
        expected_rows = int(data["E"])
    else:
        expected_rows = None

    N = expected_rows

    # -------------------------
    # Validate requirements
    # -------------------------
    for spec, _params in selected:
        missing = [k for k in spec.requires if k not in data and k not in feat_ctx]
        if missing:
            raise KeyError(f"Feature '{spec.token}' requires missing fields: {missing}")

    # -------------------------
    # Compute features
    # -------------------------
    for spec, params in selected:
        f = spec.fn(data, feat_ctx, params)

        # ---- validate output
        if not isinstance(f, np.ndarray):
            raise TypeError(f"Feature '{spec.token}' must return np.ndarray")

        if f.ndim != 2:
            raise ValueError(f"Feature '{spec.token}' must return a 2D array, got shape {f.shape}")

        if N is None:
            N = f.shape[0]
        elif f.shape[0] != N:
            raise ValueError(f"Feature '{spec.token}' has mismatched rows: expected {N}, got {f.shape[0]}")

        feats.append(f.astype(np.float32))

        k = spec.keys_fn(data, feat_ctx, params)
        if len(k) != f.shape[1]:
            raise ValueError(f"Feature '{spec.token}' keys mismatch: {len(k)} keys for {f.shape[1]} columns")
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


# -------------------------
# Voxel registry
# -------------------------


def _pool_mode(pool: str) -> str:
    p = (pool or "max").strip().lower()
    if p not in ("max", "sum", "mean"):
        raise ValueError("pool must be one of: 'max', 'sum', 'mean'")
    return p


def build_from_registry_voxel(
    data: dict[str, Any],
    *,
    mode: Any,
    registry: dict[str, Any],
    feat_ctx: dict[str, Any] | None = None,
    grid_size: int,
    pool: str = "sum",
    return_keys: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[str]]:
    """Build a float32 ``(C, G, G, G)`` voxel grid from registered features, optionally with keys."""
    feat_ctx = dict(feat_ctx or {})
    pool_mode = _pool_mode(pool)
    feat_ctx["pool"] = pool_mode

    selected = _select_features(
        mode,
        registry,
        kind="voxel",
        allow_unvalidated_params=True,
    )

    G = int(grid_size)

    # -------- validate requirements and channel declarations --------
    channel_specs = []
    for spec, params in selected:
        missing = [k for k in spec.requires if k not in data and k not in feat_ctx]
        if missing:
            raise KeyError(f"Voxel feature '{spec.token}' missing fields: {missing}")

        declared = int(spec.n_channels_fn(data, feat_ctx, params))
        feature_keys = spec.keys_fn(data, feat_ctx, params)
        if declared < 0:
            raise ValueError(f"Voxel feature '{spec.token}' declared a negative channel count: {declared}")
        if len(feature_keys) != declared:
            raise ValueError(
                f"Voxel feature '{spec.token}' channel/key mismatch: "
                f"declared {declared} channels but emitted {len(feature_keys)} keys"
            )
        channel_specs.append((spec, params, declared, feature_keys))

    C = sum(item[2] for item in channel_specs)

    grid = np.zeros((C, G, G, G), dtype=np.float32)

    counts = None
    if pool_mode == "mean":
        counts = np.zeros_like(grid)

    # -------- fill --------
    ch = 0
    keys = []

    for spec, params, declared, feature_keys in channel_specs:
        ch_start = ch
        ch = spec.fill_fn(
            grid=grid,
            counts=counts,
            ch_start=ch_start,
            data=data,
            ctx=feat_ctx,
            params=params,
        )
        emitted = ch - ch_start
        if emitted != declared:
            raise ValueError(
                f"Voxel feature '{spec.token}' channel mismatch: declared {declared} but emitted {emitted}"
            )

        keys.extend(feature_keys)

    if ch != C:
        raise ValueError(f"Voxel channel accounting mismatch: expected {C}, emitted {ch}")
    if len(keys) != C:
        raise ValueError(f"Voxel channel/key mismatch: {C} channels for {len(keys)} keys")

    # -------- finalize --------
    if counts is not None:  # allocated above exactly when pool_mode == "mean"
        denom = np.maximum(counts, 1.0)
        grid /= denom

    return (grid, keys) if return_keys else grid
