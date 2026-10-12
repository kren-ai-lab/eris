from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from eris.features.feature_registry import VoxelFeatureSpec, register_grid_voxel_feature
from eris.features.node.constants import AA_VOCAB

# --- canonical ordering (NO CAMBIAR en releases)
_AA_ONEHOT_KEYS = [f"aa_onehot.{aa}" for aa in AA_VOCAB]
_AA_INDEX_KEY = ["aa_index"]


def _normalize_scalar_params(token: str, params: Dict[str, Any]) -> Dict[str, Any]:
    unknown = sorted(set(params) - {"normalization"})
    if unknown:
        raise ValueError(f"Unknown {token} voxel parameters: {unknown}")
    normalization = params.get("normalization", "none")
    if not isinstance(normalization, str):
        raise TypeError(f"{token} voxel parameter 'normalization' must be a string")
    normalization = normalization.strip().lower()
    if normalization not in {"none", "unit_interval"}:
        raise ValueError(
            f"{token} voxel parameter 'normalization' must be one of: none, unit_interval"
        )
    return {"normalization": normalization}


def normalize_plddt_params(params: Dict[str, Any]) -> Dict[str, Any]:
    return _normalize_scalar_params("plddt", params)


def normalize_bfactor_params(params: Dict[str, Any]) -> Dict[str, Any]:
    return _normalize_scalar_params("bfactor", params)


# -------------------------
# occupancy
# -------------------------
def add_occupancy_fn(
    *,
    grid: np.ndarray,
    counts: Optional[np.ndarray],
    ch_start: int,
    data: Dict[str, Any],
    ctx: Dict[str, Any],
    params: Dict[str, Any],
) -> int:
    ijk = data["ijk"]
    pool = ctx["pool"]

    if pool == "max":
        grid[ch_start, ijk[:, 0], ijk[:, 1], ijk[:, 2]] = 1.0
    else:
        for i, j, k in ijk:
            grid[ch_start, i, j, k] += 1.0
            if counts is not None:
                counts[ch_start, i, j, k] += 1.0
    return ch_start + 1


def occupancy_keys(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
    return ["occupancy"]


def occupancy_n_channels(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> int:
    return 1


register_grid_voxel_feature(
    VoxelFeatureSpec(
        token="occupancy",
        order=10,
        requires={"ijk", "pool"},
        fill_fn=add_occupancy_fn,
        keys_fn=occupancy_keys,
        n_channels_fn=occupancy_n_channels,
    )
)


# -------------------------
# density
# -------------------------
def _gaussian_splat(
    grid: np.ndarray, center_ijk: Tuple[int, int, int], sigma: float, value: float = 1.0
) -> None:
    if sigma <= 0:
        return
    gx, gy, gz = grid.shape
    ci, cj, ck = center_ijk
    rad = int(np.ceil(3.0 * sigma))

    i0, i1 = max(0, ci - rad), min(gx - 1, ci + rad)
    j0, j1 = max(0, cj - rad), min(gy - 1, cj + rad)
    k0, k1 = max(0, ck - rad), min(gz - 1, ck + rad)

    ii = np.arange(i0, i1 + 1) - ci
    jj = np.arange(j0, j1 + 1) - cj
    kk = np.arange(k0, k1 + 1) - ck

    gi = np.exp(-(ii.astype(np.float32) ** 2) / (2.0 * sigma * sigma))
    gj = np.exp(-(jj.astype(np.float32) ** 2) / (2.0 * sigma * sigma))
    gk = np.exp(-(kk.astype(np.float32) ** 2) / (2.0 * sigma * sigma))

    blob = (gi[:, None, None] * gj[None, :, None] * gk[None, None, :]) * float(value)
    grid[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1] += blob


def add_density_fn(
    *,
    grid: np.ndarray,
    counts: Optional[np.ndarray],
    ch_start: int,
    data: Dict[str, Any],
    ctx: Dict[str, Any],
    params: Dict[str, Any],
) -> int:
    ijk = data["ijk"]
    pool = ctx["pool"]

    sigma = params.get("sigma", ctx["sigma"])

    sig = float(sigma)
    for i, j, k in ijk:
        _gaussian_splat(grid[ch_start], (int(i), int(j), int(k)), sigma=sig, value=1.0)
        if pool != "max" and counts is not None:
            counts[ch_start, int(i), int(j), int(k)] += 1.0
    return ch_start + 1


def density_keys(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
    return ["density"]


def density_n_channels(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> int:
    return 1


register_grid_voxel_feature(
    VoxelFeatureSpec(
        token="density",
        order=20,
        requires={"ijk", "pool", "sigma"},
        fill_fn=add_density_fn,
        keys_fn=density_keys,
        n_channels_fn=density_n_channels,
    )
)


# -------------------------
# plddt
# -------------------------
def _accumulate_scalar(
    grid_ch: np.ndarray,
    counts_ch: Optional[np.ndarray],
    ijk: np.ndarray,
    values: np.ndarray,
    pool: str,
) -> None:
    """
    Accumulate scalar values into a single channel (in-place).
    - max: max pooling
    - sum: sum pooling
    - mean: sum pooling + counts (counts_ch required)
    """
    if pool == "max":
        maxima: Dict[Tuple[int, int, int], float] = {}
        for (i, j, k), val in zip(ijk, values):
            coordinate = (int(i), int(j), int(k))
            maxima[coordinate] = max(maxima.get(coordinate, float("-inf")), float(val))
        for (i, j, k), val in maxima.items():
            grid_ch[i, j, k] = val
    else:
        for (i, j, k), val in zip(ijk, values):
            grid_ch[i, j, k] += val
            if counts_ch is not None:
                counts_ch[i, j, k] += 1.0


def add_scalar_pool(
    *,
    grid: np.ndarray,
    counts: Optional[np.ndarray],
    ch: int,
    ijk: np.ndarray,
    values: np.ndarray,
    pool: str,
    normalize: Optional[Tuple[float, float]] = None,
) -> int:
    v = values.copy()

    if normalize is not None:
        lo, hi = normalize
        v = (v - lo) / (hi - lo + 1e-8)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    counts_ch = None if counts is None else counts[ch]

    _accumulate_scalar(grid[ch], counts_ch, ijk, v, pool)

    return ch + 1


def add_plddt_fn(
    *,
    grid: np.ndarray,
    counts: Optional[np.ndarray],
    ch_start: int,
    data: Dict[str, Any],
    ctx: Dict[str, Any],
    params: Dict[str, Any],
) -> int:

    return add_scalar_pool(
        grid=grid,
        counts=counts,
        ch=ch_start,
        ijk=data["ijk"],
        values=data["plddt"],
        pool=ctx["pool"],
        normalize=(0.0, 100.0) if params["normalization"] == "unit_interval" else None,
    )


def plddt_keys(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
    return ["plddt"]


def plddt_n_channels(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> int:
    return 1


register_grid_voxel_feature(
    VoxelFeatureSpec(
        token="plddt",
        order=30,
        requires={"ijk", "plddt", "pool"},
        fill_fn=add_plddt_fn,
        keys_fn=plddt_keys,
        n_channels_fn=plddt_n_channels,
        params_fn=normalize_plddt_params,
    )
)


# -------------------------
# bfactor
# -------------------------
def add_bfactor_fn(
    *,
    grid: np.ndarray,
    counts: Optional[np.ndarray],
    ch_start: int,
    data: Dict[str, Any],
    ctx: Dict[str, Any],
    params: Dict[str, Any],
) -> int:

    return add_scalar_pool(
        grid=grid,
        counts=counts,
        ch=ch_start,
        ijk=data["ijk"],
        values=data["bfactor_ca"],
        pool=ctx["pool"],
        normalize=(0.0, 100.0) if params["normalization"] == "unit_interval" else None,
    )


def bfactor_keys(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
    return ["bfactor"]


def bfactor_n_channels(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> int:
    return 1


register_grid_voxel_feature(
    VoxelFeatureSpec(
        token="bfactor",
        order=40,
        requires={"ijk", "bfactor_ca", "pool"},
        fill_fn=add_bfactor_fn,
        keys_fn=bfactor_keys,
        n_channels_fn=bfactor_n_channels,
        params_fn=normalize_bfactor_params,
    )
)


# -------------------------
# aa_onehot
# -------------------------
def _accumulate_onehot(
    grid: np.ndarray,
    counts: Optional[np.ndarray],
    ijk: np.ndarray,
    aa: np.ndarray,
    base_ch: int,
    pool: str,
) -> None:
    """
    Accumulate AA onehot into channels base_ch..base_ch+20.
    - max: set presence (0/1)
    - sum/mean: add 1 per hit (mean divides by counts later)
    """
    if pool == "max":
        for (i, j, k), a in zip(ijk, aa):
            a = int(a)
            if 0 <= a < 21:
                grid[base_ch + a, i, j, k] = 1.0
    else:
        for (i, j, k), a in zip(ijk, aa):
            a = int(a)
            if 0 <= a < 21:
                grid[base_ch + a, i, j, k] += 1.0
                if counts is not None:
                    counts[base_ch + a, i, j, k] += 1.0


def add_aa_onehot_fn(
    *,
    grid: np.ndarray,
    counts: Optional[np.ndarray],
    ch_start: int,
    data: Dict[str, Any],
    ctx: Dict[str, Any],
    params: Dict[str, Any],
) -> int:
    ijk = data["ijk"]
    aa_v = data["aa_index"]
    pool = ctx["pool"]

    _accumulate_onehot(grid, counts, ijk, aa_v, base_ch=ch_start, pool=pool)
    return ch_start + 21


def aa_onehot_keys(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
    return _AA_ONEHOT_KEYS


def aa_onehot_n_channels(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> int:
    return params.get("num_classes", 21)


register_grid_voxel_feature(
    VoxelFeatureSpec(
        token="aa_onehot",
        order=50,
        requires={"ijk", "aa_index", "pool"},
        fill_fn=add_aa_onehot_fn,
        keys_fn=aa_onehot_keys,
        n_channels_fn=aa_onehot_n_channels,
    )
)
