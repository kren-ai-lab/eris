"""Residue voxel-grid encoder."""

from __future__ import annotations

from typing import Any, ClassVar, cast

import numpy as np

from eris.core.base_encoder import BaseResidueEncoder, EncodingContext
from eris.core.config import ResidueVoxelGridConfig
from eris.core.representations import VoxelGridRepresentation
from eris.core.representations.output_declarations import VOXEL_OUTPUTS
from eris.encoders._context import prepared_arrays
from eris.features.grid.grid import build_grid_attr


class VoxelGridCA(BaseResidueEncoder):
    """Residue voxel grid around CA coordinates."""

    # Identity
    name: ClassVar[str] = "residue_voxel_grid"
    version: ClassVar[str] = "0.1.2"
    representation: ClassVar = VoxelGridRepresentation()
    output_declarations: ClassVar = VOXEL_OUTPUTS

    config_type: ClassVar = ResidueVoxelGridConfig

    def _create_data(self, ctx: EncodingContext) -> dict[str, Any]:
        config = cast("ResidueVoxelGridConfig", self.config)
        strc, centroid, pos = prepared_arrays(ctx)

        grid, keys, extra = build_grid_attr(
            pos=pos,
            centroid=centroid,
            strc=strc,
            channels=config.channels.to_list(),
            grid_size=config.grid_size,
            voxel_size=config.voxel_size,
            fit=config.fit,
            margin=config.margin,
            center=False,
            pool=config.pool,
            sigma=config.sigma,
            return_keys=True,
        )

        if config.clamp:
            grid = np.clip(grid, 0.0, 1.0)

        grid = grid.astype(config.dtype)

        ctx.runtime["channels"] = keys
        ctx.runtime["voxel_size_effective"] = float(extra["voxel_size"])

        return {
            "grid": grid,
            "grid_size": np.asarray([config.grid_size], dtype=np.int32),
            "voxel_size": np.asarray([extra["voxel_size"]], dtype=np.float32),
            "origin": cast("np.ndarray", extra["origin"]).astype(np.float32),
            "centroid": centroid.astype(np.float32),
            "residue_ids": strc["residue_ids"],
            "chain_id": strc["chain_id"],
            "omitted_residue_ids": np.asarray(strc.get("omitted_residue_ids", []), dtype=np.str_),
            "resname": strc["resname"],
            "aa_index": strc["aa_index"],
            "plddt": strc["plddt"],
            "bfactor_ca": strc["bfactor_ca"],
        }
