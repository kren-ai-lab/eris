# src/morpheo/encoders/grids/voxel_grid_ca.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

from morpheo.core.cache import DiskCache, make_cache_key
from morpheo.core.hashing import sha256_file, sha256_text
from morpheo.core.io import load_and_extract
from morpheo.core.registry import register_encoder
from morpheo.core.types import StructureArtifact, StructureInput


def _bbox(pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mn = pos.min(axis=0)
    mx = pos.max(axis=0)
    return mn, mx


def _gaussian_splat(grid: np.ndarray, center_ijk: Tuple[int, int, int], sigma: float, value: float = 1.0) -> None:
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


def _parse_channels(channels: Any) -> List[str]:
    """
    Accept:
      - "occupancy+plddt+aa_onehot"
      - ["occupancy","plddt","aa_onehot"]
    """
    if isinstance(channels, str):
        s = channels.strip()
        if not s:
            return ["occupancy"]
        return [c.strip() for c in s.split("+") if c.strip()]
    if isinstance(channels, list):
        out = []
        for c in channels:
            if not isinstance(c, str):
                raise ValueError("channels list must contain strings")
            c = c.strip()
            if c:
                out.append(c)
        return out if out else ["occupancy"]
    raise ValueError("channels must be a string (e.g. 'occupancy+plddt') or a list of strings")


def _pool_mode(pool: str) -> str:
    p = (pool or "max").strip().lower()
    if p not in ("max", "sum", "mean"):
        raise ValueError("pool must be one of: 'max', 'sum', 'mean'")
    return p


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
        for (i, j, k), val in zip(ijk, values):
            if val > grid_ch[i, j, k]:
                grid_ch[i, j, k] = val
    else:
        for (i, j, k), val in zip(ijk, values):
            grid_ch[i, j, k] += val
            if counts_ch is not None:
                counts_ch[i, j, k] += 1.0


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


@dataclass
class VoxelGridCA:
    name: str = "voxel_grid_ca"

    grid_size: int = 64
    dtype: str = "float32"

    # multi-channel spec
    channels: Any = "occupancy"  # e.g. "density+aa_onehot+plddt"

    # spatial normalization
    center: bool = True
    fit: bool = True
    margin: float = 4.0
    voxel_size: Optional[float] = None

    # density params (voxel units)
    sigma: float = 1.0

    # pooling
    pool: str = "max"  # "max" | "sum" | "mean"

    # value handling
    clamp: bool = True  # clip final grid to [0,1] (recommended for occupancy/density)

    use_cache: bool = True
    cache: Optional[DiskCache] = None

    def fingerprint(self) -> str:
        ch = "+".join(_parse_channels(self.channels))
        return sha256_text(
            f"{self.name}|G={self.grid_size}|dtype={self.dtype}|channels={ch}|center={self.center}|fit={self.fit}|margin={self.margin}|voxel_size={self.voxel_size}|sigma={self.sigma}|pool={self.pool}|clamp={self.clamp}|use_cache={self.use_cache}"
        )

    def encode(self, inp: StructureInput) -> StructureArtifact:
        cache = self.cache or DiskCache()
        file_hash = sha256_file(Path(inp.source))
        key = make_cache_key(self.name, self.fingerprint(), file_hash, inp.to_string())

        if self.use_cache:
            hit = cache.get(key)
            if hit is not None:
                data, meta = hit
                return StructureArtifact(self.name, data=data, meta=meta)

        pos, residue_ids, resname, aa_index, plddt, bfactor_ca = load_and_extract(inp)
        n = int(pos.shape[0])
        if n == 0:
            raise ValueError("No residues found to voxelize.")

        pos0 = pos.astype(np.float32)

        # center
        if self.center:
            centroid = pos0.mean(axis=0, keepdims=True)
            pos_c = pos0 - centroid
        else:
            centroid = np.zeros((1, 3), dtype=np.float32)
            pos_c = pos0

        G = int(self.grid_size)
        if G < 8:
            raise ValueError("grid_size too small; use >= 8.")

        # determine voxel_size
        if self.voxel_size is not None:
            vs = float(self.voxel_size)
            if vs <= 0:
                raise ValueError("voxel_size must be > 0.")
        else:
            if self.fit:
                mn, mx = _bbox(pos_c)
                mn = mn - float(self.margin)
                mx = mx + float(self.margin)
                span = (mx - mn).max()
                vs = float(span / max(1, (G - 1)))
                if vs <= 0:
                    vs = 1.0
            else:
                vs = 1.0

        half = (G - 1) * vs / 2.0
        origin = np.array([-half, -half, -half], dtype=np.float32)

        # map points to voxels
        ijk = np.floor((pos_c - origin[None, :]) / vs).astype(np.int64)
        valid = (
            (ijk[:, 0] >= 0) & (ijk[:, 0] < G) &
            (ijk[:, 1] >= 0) & (ijk[:, 1] < G) &
            (ijk[:, 2] >= 0) & (ijk[:, 2] < G)
        )
        ijk = ijk[valid]
        aa_v = np.asarray(aa_index, dtype=np.int64).reshape(-1)[valid]
        plddt_v = np.asarray(plddt, dtype=np.float32).reshape(-1)[valid]
        bfac_v = np.asarray(bfactor_ca, dtype=np.float32).reshape(-1)[valid]

        ch_list = _parse_channels(self.channels)

        # count channels
        C = 0
        for c in ch_list:
            if c == "aa_onehot":
                C += 21
            else:
                C += 1

        grid = np.zeros((C, G, G, G), dtype=np.float32)

        pool = _pool_mode(self.pool)
        counts = None
        if pool == "mean":
            counts = np.zeros((C, G, G, G), dtype=np.float32)

        # channel fillers
        def add_occupancy(ch: int) -> int:
            if pool == "max":
                grid[ch, ijk[:, 0], ijk[:, 1], ijk[:, 2]] = 1.0
            else:
                for i, j, k in ijk:
                    grid[ch, i, j, k] += 1.0
                    if counts is not None:
                        counts[ch, i, j, k] += 1.0
            return ch + 1

        def add_density(ch: int) -> int:
            sig = float(self.sigma)
            for i, j, k in ijk:
                _gaussian_splat(grid[ch], (int(i), int(j), int(k)), sigma=sig, value=1.0)
                if pool != "max" and counts is not None:
                    # simple proxy count: center voxel hit
                    counts[ch, int(i), int(j), int(k)] += 1.0
            return ch + 1

        def add_scalar_pool(ch: int, values: np.ndarray, normalize: Optional[Tuple[float, float]] = None) -> int:
            v = values.copy()
            if normalize is not None:
                lo, hi = normalize
                v = (v - lo) / (hi - lo + 1e-8)
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            cch = None if counts is None else counts[ch]
            _accumulate_scalar(grid[ch], cch, ijk, v, pool=pool)
            return ch + 1

        def add_aa_onehot(ch: int) -> int:
            _accumulate_onehot(grid, counts, ijk, aa_v, base_ch=ch, pool=pool)
            return ch + 21

        # fill channels in order
        cidx = 0
        for c in ch_list:
            if c == "occupancy":
                cidx = add_occupancy(cidx)
            elif c == "density":
                cidx = add_density(cidx)
            elif c == "plddt":
                cidx = add_scalar_pool(cidx, plddt_v, normalize=(0.0, 100.0))
            elif c == "bfactor":
                cidx = add_scalar_pool(cidx, bfac_v, normalize=(0.0, 100.0))
            elif c == "aa_onehot":
                cidx = add_aa_onehot(cidx)
            else:
                raise ValueError(
                    f"Unknown channel '{c}'. Supported: occupancy, density, plddt, bfactor, aa_onehot"
                )

        if pool == "mean":
            denom = np.maximum(counts, 1.0)
            grid = grid / denom

        if self.clamp:
            grid = np.clip(grid, 0.0, 1.0)

        if self.dtype == "float16":
            grid_out = grid.astype(np.float16)
        else:
            grid_out = grid.astype(np.float32)

        data = {
            "grid": grid_out,
            "channels": np.asarray(ch_list, dtype=np.str_),
            "grid_size": np.asarray([G], dtype=np.int32),
            "voxel_size": np.asarray([vs], dtype=np.float32),
            "origin": origin.astype(np.float32),
            "centroid": centroid.astype(np.float32),

            "residue_ids": residue_ids.astype(np.str_),
            "resname": resname.astype(np.str_),
            "aa_index": np.asarray(aa_index, dtype=np.int16),
            "plddt": np.asarray(plddt, dtype=np.float32),
            "bfactor_ca": np.asarray(bfactor_ca, dtype=np.float32),
        }

        meta: Dict[str, Any] = {
            "encoder_name": self.name,
            "input": inp.to_string(),
            "grid_size": G,
            "dtype": self.dtype,
            "channels": ch_list,
            "center": bool(self.center),
            "fit": bool(self.fit),
            "margin": float(self.margin),
            "voxel_size": float(vs),
            "sigma": float(self.sigma),
            "pool": pool,
            "clamp": bool(self.clamp),
            "n_residues": int(n),
            "n_points_in_grid": int(ijk.shape[0]),
            "shapes": {k: list(v.shape) for k, v in data.items() if hasattr(v, "shape")},
        }

        if self.use_cache:
            cache.set(key, data=data, meta=meta)

        return StructureArtifact(self.name, data=data, meta=meta)


register_encoder("voxel_grid_ca", lambda: VoxelGridCA())
