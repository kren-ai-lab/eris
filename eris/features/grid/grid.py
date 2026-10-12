"""Voxel grid attribute builder."""

from __future__ import annotations

from typing import Any

import numpy as np

from eris.features.feature_builders import build_from_registry_voxel
from eris.features.feature_registry import GRID_VOXEL_REGISTRY


def _bbox(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mn = pos.min(axis=0)
    mx = pos.max(axis=0)
    return mn, mx


def build_grid_attr(
    *,
    pos: np.ndarray,
    centroid: np.ndarray,
    strc: dict[str, np.ndarray],
    channels: Any,
    grid_size: int,
    voxel_size: float | None,
    fit: bool,
    margin: float,
    center: bool,
    pool: str,
    sigma: float,
    return_keys: bool = False,
) -> tuple[np.ndarray, list[str], dict[str, float | np.ndarray]]:
    """Voxelize atom positions and return the grid, channel keys and grid geometry."""
    G = int(grid_size)

    pos_in = pos.copy()

    if center:
        centroid = np.asarray(centroid).reshape(-1)
        pos_in = pos_in - centroid

    # -------- voxel size --------
    if voxel_size is not None:
        vs = float(voxel_size)
    elif fit:
        mn, mx = _bbox(pos_in)
        mn -= margin
        mx += margin
        span = (mx - mn).max()
        vs = float(span / max(1, (G - 1)))
    else:
        vs = 1.0

    # -------- origin --------
    half = (G - 1) * vs / 2.0
    origin = np.array([-half, -half, -half], dtype=np.float32)

    # -------- voxel mapping --------
    ijk = np.floor((pos_in - origin[None, :]) / vs).astype(np.int64)

    valid = (
        (ijk[:, 0] >= 0)
        & (ijk[:, 0] < G)
        & (ijk[:, 1] >= 0)
        & (ijk[:, 1] < G)
        & (ijk[:, 2] >= 0)
        & (ijk[:, 2] < G)
    )

    ijk = ijk[valid]

    data = {
        "ijk": ijk,
        "aa_index": strc["aa_index"][valid],
        "plddt": strc["plddt"][valid],
        "bfactor_ca": strc["bfactor_ca"][valid],
    }

    feat_ctx = {
        "pool": pool,
        "sigma": sigma,
    }

    result = build_from_registry_voxel(
        data=data,
        mode=channels,
        registry=GRID_VOXEL_REGISTRY,
        feat_ctx=feat_ctx,
        grid_size=G,
        pool=pool,
        return_keys=return_keys,
    )

    if isinstance(result, tuple):
        grid, keys = result
    else:
        grid = result
        keys = []

    return grid, keys, {"voxel_size": vs, "origin": origin}
