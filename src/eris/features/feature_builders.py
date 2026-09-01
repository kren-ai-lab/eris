from typing import Callable, Dict, Any, List, Set, Optional, Tuple
import numpy as np

from eris.features.utils import parse_mode
from eris.features.feature_registry import FeatureSpec


def _select_features(
    mode: Any,
    registry: Dict[str, Any],
    *,
    kind: str,
    allow_params: bool,
):
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

    if not allow_params:
        for token, params in toks:
            if params:
                raise ValueError(
                    f"Feature '{token}' does not support parameters: "
                    f"{sorted(str(param) for param in params)}"
                )

    selected = [(registry[token], params) for token, params in toks]
    selected.sort(key=lambda item: item[0].order)
    return selected


def validate_feature_mode(mode: Any, registry: Dict[str, FeatureSpec]) -> None:
    _select_features(
        mode,
        registry,
        kind="feature",
        allow_params=False,
    )


# -------------------------
# Normal registry
# -------------------------


def build_from_registry(
    data: Dict[str, Any],
    *,
    mode: str,
    registry: Dict[str, FeatureSpec],
    feat_ctx: Optional[Dict[str, Any]] = None,
    return_keys: bool = False,
):
    feat_ctx = feat_ctx or {}

    selected = _select_features(
        mode,
        registry,
        kind="feature",
        allow_params=False,
    )

    feats: List[np.ndarray] = []
    keys: List[str] = []

    if "n" in data:
        expected_rows: Optional[int] = int(data["n"])
    elif "E" in data:
        expected_rows = int(data["E"])
    else:
        expected_rows = None

    N = expected_rows

    # -------------------------
    # Validate requirements
    # -------------------------
    for spec, params in selected:
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
            raise ValueError(
                f"Feature '{spec.token}' has mismatched rows: expected {N}, got {f.shape[0]}"
            )

        feats.append(f.astype(np.float32))

        k = spec.keys_fn(data, feat_ctx, params)
        if len(k) != f.shape[1]:
            raise ValueError(
                f"Feature '{spec.token}' keys mismatch: {len(k)} keys for {f.shape[1]} columns"
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


# -------------------------
# Voxel registry
# -------------------------


def _pool_mode(pool: str) -> str:
    p = (pool or "max").strip().lower()
    if p not in ("max", "sum", "mean"):
        raise ValueError("pool must be one of: 'max', 'sum', 'mean'")
    return p


def build_from_registry_voxel(
    data: Dict[str, Any],
    *,
    mode: Any,
    registry: Dict[str, Any],
    feat_ctx: Optional[Dict[str, Any]] = None,
    grid_size: int,
    pool: str = "sum",
    return_keys: bool = False,
):
    feat_ctx = dict(feat_ctx or {})
    pool_mode = _pool_mode(pool)
    feat_ctx["pool"] = pool_mode

    selected = _select_features(
        mode,
        registry,
        kind="voxel",
        allow_params=True,
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
            raise ValueError(
                f"Voxel feature '{spec.token}' declared a negative channel count: {declared}"
            )
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
                f"Voxel feature '{spec.token}' channel mismatch: "
                f"declared {declared} but emitted {emitted}"
            )

        keys.extend(feature_keys)

    if ch != C:
        raise ValueError(f"Voxel channel accounting mismatch: expected {C}, emitted {ch}")
    if len(keys) != C:
        raise ValueError(f"Voxel channel/key mismatch: {C} channels for {len(keys)} keys")

    # -------- finalize --------
    if pool_mode == "mean":
        denom = np.maximum(counts, 1.0)
        grid /= denom

    return (grid, keys) if return_keys else grid
