from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from eris.core.base_encoder import BaseResidueEncoder, EncodingContext
from eris.core.config import ResidueVoxelGridConfig
from eris.core.io import load_and_extract_residue
from eris.core.representations import VoxelGridRepresentation
from eris.core.representations.output_declarations import VOXEL_OUTPUTS
from eris.features.grid.grid import build_grid_attr

if TYPE_CHECKING:
    from eris.core.cache import DiskCache
    from eris.core.representations.base_rep import BaseRepresentation


@dataclass(init=False)
class VoxelGridCA(BaseResidueEncoder):
    # Identity
    name: ClassVar[str] = "residue_voxel_grid"
    version: ClassVar[str] = "0.1.1"
    representation: ClassVar[BaseRepresentation] = VoxelGridRepresentation()
    output_declarations_v2: ClassVar = VOXEL_OUTPUTS

    config: ResidueVoxelGridConfig = field(init=False)

    def __init__(
        self,
        *,
        config: ResidueVoxelGridConfig | None = None,
        use_cache: bool = True,
        cache: DiskCache | None = None,
        **representation_kwargs: Any,
    ) -> None:
        if config is not None:
            if not isinstance(config, ResidueVoxelGridConfig):
                raise TypeError("config must be a ResidueVoxelGridConfig")
            if representation_kwargs:
                names = ", ".join(sorted(representation_kwargs))
                raise TypeError(f"config cannot be combined with representation kwargs: {names}")
            resolved_config = config
        else:
            resolved_config = ResidueVoxelGridConfig(**representation_kwargs)

        self.config = resolved_config
        self.use_cache = use_cache
        self.cache = cache
        super().__post_init__()

    @property
    def center(self) -> bool:
        return self.config.center

    @property
    def scale(self) -> float | None:
        return self.config.scale

    @property
    def residue_center(self) -> str:
        return self.config.residue_center

    @property
    def amino_acids_only(self) -> bool:
        return self.config.amino_acids_only

    @property
    def altloc(self) -> str:
        return self.config.altloc

    @property
    def missing_residue_center(self) -> str:
        return self.config.missing_residue_center

    @property
    def grid_size(self) -> int:
        return self.config.grid_size

    @property
    def voxel_size(self) -> float | None:
        return self.config.voxel_size

    @property
    def fit(self) -> bool:
        return self.config.fit

    @property
    def margin(self) -> float:
        return self.config.margin

    @property
    def channels(self) -> str:
        return "+".join(token.name for token in self.config.channels.tokens)

    @property
    def sigma(self) -> float:
        return self.config.sigma

    @property
    def pool(self) -> str:
        return self.config.pool

    @property
    def clamp(self) -> bool:
        return self.config.clamp

    @property
    def dtype(self) -> str:
        return self.config.dtype

    def get_params(self) -> dict[str, Any]:
        return self.config.to_dict()

    def _load_structure(self, ctx: EncodingContext) -> dict[str, Any]:
        return load_and_extract_residue(
            ctx.inp,
            amino_acids_only=self.config.amino_acids_only,
            mode=self.config.residue_center,
            altloc=self.config.altloc,
            missing_ca=self.config.missing_residue_center,
            plddt_source=self.config.plddt_source,
        )

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
            "omitted_residue_ids": np.asarray(strc.get("omitted_residue_ids", []), dtype=np.str_),
            "resname": strc["resname"],
            "aa_index": strc["aa_index"],
            "plddt": strc["plddt"],
            "bfactor_ca": strc["bfactor_ca"],
        }
