from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from eris.core.base_encoder import BaseResidueEncoder, EncodingContext
from eris.core.config import ResiduePointCloudConfig
from eris.core.io import load_and_extract_residue
from eris.core.representations import PointCloudRepresentation
from eris.core.representations.output_declarations import RESIDUE_POINT_CLOUD_OUTPUTS
from eris.features.node.residue import build_node_attr

if TYPE_CHECKING:
    from eris.core.cache import DiskCache
    from eris.core.representations.base_rep import BaseRepresentation


@dataclass(init=False)
class ResiduePointCloudCA(BaseResidueEncoder):
    #Identity
    name: ClassVar[str] = "residue_point_cloud"
    representation: ClassVar[BaseRepresentation] = PointCloudRepresentation()
    output_declarations_v2: ClassVar = RESIDUE_POINT_CLOUD_OUTPUTS

    config: ResiduePointCloudConfig = field(init=False)

    def __init__(
        self,
        *,
        config: ResiduePointCloudConfig | None = None,
        use_cache: bool = True,
        cache: DiskCache | None = None,
        **representation_kwargs: Any,
    ) -> None:
        if config is not None:
            if not isinstance(config, ResiduePointCloudConfig):
                raise TypeError("config must be a ResiduePointCloudConfig")
            if representation_kwargs:
                names = ", ".join(sorted(representation_kwargs))
                raise TypeError(
                    f"config cannot be combined with representation kwargs: {names}"
                )
            resolved_config = config
        else:
            resolved_config = ResiduePointCloudConfig(**representation_kwargs)

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
    def node_attr_mode(self) -> str:
        return "+".join(token.name for token in self.config.node_attr_mode.tokens)

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

        node_attr, node_attr_keys = build_node_attr(
            strc["aa_index"],
            plddt=strc["plddt"],
            bfactor_ca=strc["bfactor_ca"],
            mode=self.config.node_attr_mode.to_list(),
            return_keys=True,
        )

        ctx.runtime["node_attr_keys"] = node_attr_keys

        data: dict[str, Any] = {
            "pos": pos_out.astype(np.float32),
            "centroid": centroid.astype(np.float32),
            "node_attr": node_attr.astype(np.float32),
            "residue_ids": strc["residue_ids"].astype(np.str_),
            "resname": strc["resname"].astype(np.str_),
            "aa_index": strc["aa_index"].astype(np.int16),
            "plddt": strc["plddt"].astype(np.float32),
            "bfactor_ca": strc["bfactor_ca"].astype(np.float32),
        }

        return data


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
