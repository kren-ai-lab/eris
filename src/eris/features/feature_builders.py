from typing import Callable, Dict, Any, List, Set, Optional, Tuple
import numpy as np

from eris.features.utils import parse_mode
from eris.features.feature_registry import FeatureSpec

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

        k = spec.keys_fn(data, feat_ctx, params)
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
    feat_ctx = feat_ctx or {}

    toks = parse_mode(mode)

    unknown = [t[0] for t in toks if t[0] not in registry]
    if unknown:
        raise ValueError(f"Unknown voxel tokens: {unknown}")

    selected = [(registry[t[0]], t[1]) for t in toks]
    selected.sort(key=lambda x: x[0].order)

    G = int(grid_size)

    # -------- count channels --------
    C = sum(spec.n_channels_fn(data, feat_ctx, params) for spec, params in selected)

    grid = np.zeros((C, G, G, G), dtype=np.float32)

    pool_mode = _pool_mode(pool)
    counts = None
    if pool_mode == "mean":
        counts = np.zeros_like(grid)

    # -------- validate --------
    for spec, params in selected:
        missing = [k for k in spec.requires if k not in data and k not in feat_ctx]
        if missing:
            raise KeyError(f"Voxel feature '{spec.token}' missing fields: {missing}")

    # -------- fill --------
    ch = 0
    keys = []

    for spec, params in selected:
        ch = spec.fill_fn(
            grid=grid,
            counts=counts,
            ch_start=ch,
            data=data,
            ctx=feat_ctx,
            params=params,
        )

        keys.extend(spec.keys_fn(data, feat_ctx, params))

    # -------- finalize --------
    if pool_mode == "mean":
        denom = np.maximum(counts, 1.0)
        grid /= denom

    return (grid, keys) if return_keys else grid
