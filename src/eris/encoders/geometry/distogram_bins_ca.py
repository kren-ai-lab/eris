from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from eris.core.base_encoder import BaseResidueEncoder, EncodingContext
from eris.core.config import ResidueDistogramBinsConfig
from eris.core.io import load_and_extract_residue
from eris.core.representations import DistogramBinsRepresentation
from eris.core.representations.output_declarations import RESIDUE_DISTOGRAM_BINS_OUTPUTS
from eris.features.utils import validate_bin_edges

if TYPE_CHECKING:
    from eris.core.cache import DiskCache
    from eris.core.representations.base_rep import BaseRepresentation


def _validate_class_capacity(config: ResidueDistogramBinsConfig) -> None:
    num_bins = config.num_bins if config.edges is None else len(config.edges) - 1
    max_class_id = num_bins if config.add_no_contact_class else num_bins - 1
    capacity = np.iinfo(np.dtype(config.dtype)).max
    if max_class_id > capacity:
        raise ValueError(
            f"dtype {config.dtype} cannot represent {num_bins} bins with maximum "
            f"class ID {max_class_id}; maximum capacity is {capacity}"
        )


@dataclass(init=False)
class DistogramBinsCA(BaseResidueEncoder):
    # Identity
    name: ClassVar[str] = "residue_distogram_bins"
    representation: ClassVar[BaseRepresentation] = DistogramBinsRepresentation()
    output_declarations_v2: ClassVar = RESIDUE_DISTOGRAM_BINS_OUTPUTS

    config: ResidueDistogramBinsConfig = field(init=False)

    def __init__(
        self,
        *,
        config: ResidueDistogramBinsConfig | None = None,
        use_cache: bool = True,
        cache: DiskCache | None = None,
        **representation_kwargs: Any,
    ) -> None:
        if config is not None:
            if not isinstance(config, ResidueDistogramBinsConfig):
                raise TypeError("config must be a ResidueDistogramBinsConfig")
            if representation_kwargs:
                names = ", ".join(sorted(representation_kwargs))
                raise TypeError(
                    f"config cannot be combined with representation kwargs: {names}"
                )
            resolved_config = config
        else:
            resolved_config = ResidueDistogramBinsConfig(**representation_kwargs)

        _validate_class_capacity(resolved_config)
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
    def edges(self) -> tuple[float, ...] | None:
        return self.config.edges

    @property
    def dmin(self) -> float | None:
        return self.config.dmin

    @property
    def dmax(self) -> float | None:
        return self.config.dmax

    @property
    def num_bins(self) -> int | None:
        return self.config.num_bins

    @property
    def include_self(self) -> bool:
        return self.config.include_self

    @property
    def add_no_contact_class(self) -> bool:
        return self.config.add_no_contact_class

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
        )

    def _create_data(self, ctx: EncodingContext) -> dict[str, Any]:
        strc = ctx.raw_structure
        centroid = ctx.centroid
        pos_out = ctx.pos

        diff = pos_out[:, None, :] - pos_out[None, :, :]
        dist = np.sqrt((diff**2).sum(axis=-1)).astype(np.float32)

        if not self.config.include_self:
            np.fill_diagonal(dist, np.nan)

        if self.config.edges is not None:
            edges = validate_bin_edges(
                np.asarray(self.config.edges, dtype=np.float32)
            )
        else:
            edges = np.linspace(
                self.config.dmin,
                self.config.dmax,
                self.config.num_bins + 1,
                dtype=np.float32,
            )

        num_bins = int(edges.size - 1)
        raw_bins = np.digitize(dist, edges, right=False) - 1
        raw_bins = np.clip(raw_bins, 0, num_bins - 1)

        if self.config.add_no_contact_class:
            out = (raw_bins + 1).astype(np.int32)
            out[dist > float(edges[-1])] = 0
            out[np.isnan(dist)] = 0
        else:
            out = raw_bins.astype(np.int32)
            out[np.isnan(dist)] = 0

        dist_bins = out.astype(np.dtype(self.config.dtype))

        return {
            "dist_bins": dist_bins,
            "bin_edges": edges.astype(np.float32),
            "pos": pos_out.astype(np.float32),
            "centroid": centroid.astype(np.float32),
            "residue_ids": strc["residue_ids"].astype(np.str_),
            "resname": strc["resname"].astype(np.str_),
            "aa_index": strc["aa_index"].astype(np.int16),
            "plddt": strc["plddt"].astype(np.float32),
            "bfactor_ca": strc["bfactor_ca"].astype(np.float32),
        }

    def _update_params_metadata(
        self,
        meta: dict[str, Any],
        ctx: EncodingContext,
    ) -> dict[str, Any]:
        meta["params"] = self.config.to_dict()
        return meta

    def _finalize_metadata(self, meta: dict[str, Any]) -> dict[str, Any]:
        meta = super()._finalize_metadata(meta)
        meta["params"] = self.config.to_dict()
        return meta
