from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from eris.core.base_encoder import BaseResidueEncoder, EncodingContext
from eris.core.config import ResidueVoxelGridConfig
from eris.core.representations import VoxelGridRepresentation
from eris.core.representations.output_declarations import VOXEL_OUTPUTS
from eris.features.grid.grid import build_grid_attr


class VoxelGridCA(BaseResidueEncoder):
    # Identity
    name: ClassVar[str] = "residue_voxel_grid"
    version: ClassVar[str] = "0.1.2"
    representation: ClassVar = VoxelGridRepresentation()
    output_declarations_v2: ClassVar = VOXEL_OUTPUTS

    config_type: ClassVar = ResidueVoxelGridConfig
    config: ResidueVoxelGridConfig

    def _create_data(self, ctx: EncodingContext) -> dict[str, Any]:
        strc = ctx.raw_structure
        pos = ctx.pos
        centroid = ctx.centroid

        grid, keys, extra = build_grid_attr(
            pos=pos,
            centroid=centroid,
            strc=strc,
            channels=self.config.channels.to_list(),
            grid_size=self.config.grid_size,
            voxel_size=self.config.voxel_size,
            fit=self.config.fit,
            margin=self.config.margin,
            center=False,
            pool=self.config.pool,
            sigma=self.config.sigma,
            return_keys=True,
        )

        if self.config.clamp:
            grid = np.clip(grid, 0.0, 1.0)

        grid = grid.astype(self.config.dtype)

        ctx.runtime["channels"] = keys
        ctx.runtime["voxel_size_effective"] = float(extra["voxel_size"])

        return {
            "grid": grid,
            "grid_size": np.asarray([self.config.grid_size], dtype=np.int32),
            "voxel_size": np.asarray([extra["voxel_size"]], dtype=np.float32),
            "origin": extra["origin"].astype(np.float32),
            "centroid": centroid.astype(np.float32),
            "residue_ids": strc["residue_ids"],
            "chain_id": strc["chain_id"],
            "omitted_residue_ids": np.asarray(strc.get("omitted_residue_ids", []), dtype=np.str_),
            "resname": strc["resname"],
            "aa_index": strc["aa_index"],
            "plddt": strc["plddt"],
            "bfactor_ca": strc["bfactor_ca"],
        }
