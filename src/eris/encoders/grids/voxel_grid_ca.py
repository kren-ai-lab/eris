from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, ClassVar
from typing import TYPE_CHECKING

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

from eris.core.base_encoder import BaseResidueEncoder, EncodingContext
from eris.core.representations import VoxelGridRepresentation

from eris.features import parse_mode, build_grid_attr

if TYPE_CHECKING:
    from eris.core.representations.base_rep import BaseRepresentation


@dataclass
class VoxelGridCA(BaseResidueEncoder):
    name: str = "voxel_grid_ca"
    representation: ClassVar[BaseRepresentation] = VoxelGridRepresentation()

    grid_size: int = 64
    dtype: str = "float32"

    # multi-channel spec
    channels: Any = "occupancy+density+aa_onehot+plddt+bfactor"  # e.g. "occupancy+density+aa_onehot+plddt"

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


    def _create_data(self, ctx: EncodingContext) -> Dict[str, Any]:
        strc = ctx.raw_structure
        pos = ctx.pos
        centroid = ctx.centroid

        grid, keys, extra = build_grid_attr(
            pos=pos,
            centroid = centroid,
            strc=strc,
            channels=self.channels,

            grid_size=self.grid_size,
            voxel_size=self.voxel_size,
            fit=self.fit,
            margin=self.margin,
            center=self.center,

            pool=self.pool,
            sigma=self.sigma,
            return_keys=True,
        )

        
        if self.clamp:
            grid = np.clip(grid, 0.0, 1.0)

        if self.dtype == "float16":
            grid = grid.astype(np.float16)
        else:
            grid = grid.astype(np.float32)

        # Update ctx
        ctx.runtime["channels"] = keys
        ctx.runtime["voxel_size_effective"] = float(extra["voxel_size"])

        data: Dict[str, Any] = {
            "grid": grid,

            "grid_size": np.asarray([self.grid_size], dtype=np.int32),
            "voxel_size": np.asarray([extra["voxel_size"]], dtype=np.float32),
            "origin": extra["origin"].astype(np.float32),
            "centroid": centroid.astype(np.float32),

            "residue_ids": strc["residue_ids"],
            "resname": strc["resname"],
            "aa_index": strc["aa_index"],
            "plddt": strc["plddt"],
            "bfactor_ca": strc["bfactor_ca"],
        }

        return data


    def _update_params_metadata(self, meta: Dict[str, Any], ctx: EncodingContext) -> Dict[str, Any]:
        runtime = ctx.runtime
        requested_channels = [
            token if not params else {token: dict(params)}
            for token, params in parse_mode(self.channels)
        ]
        
        meta["params"].update({
                "grid_size": int(self.grid_size),
                "dtype": str(self.dtype),
                "channels": requested_channels,
                "center": bool(self.center),
                "fit": bool(self.fit),
                "margin": float(self.margin),
                "voxel_size": float(runtime["voxel_size_effective"]),
                "sigma": float(self.sigma),
                "pool": str(self.pool),
                "clamp": bool(self.clamp),
            },)
        
        return meta
