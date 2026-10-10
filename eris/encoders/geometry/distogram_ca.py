from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from eris.core.base_encoder import BaseResidueEncoder, EncodingContext
from eris.core.config import ResidueDistogramConfig
from eris.core.representations import DistogramRepresentation
from eris.core.representations.output_declarations import RESIDUE_DISTOGRAM_OUTPUTS

if TYPE_CHECKING:
    from eris.core.representations.base_rep import BaseRepresentation


class DistogramCA(BaseResidueEncoder):
    # Identity
    name: ClassVar[str] = "residue_distogram"
    version: ClassVar[str] = "0.1.2"
    representation: ClassVar[BaseRepresentation] = DistogramRepresentation()
    output_declarations_v2: ClassVar = RESIDUE_DISTOGRAM_OUTPUTS

    config_type: ClassVar = ResidueDistogramConfig
    config: ResidueDistogramConfig

    def _create_data(self, ctx: EncodingContext) -> dict[str, Any]:
        strc = ctx.raw_structure
        centroid = ctx.centroid
        pos_out = ctx.pos

        diff = pos_out[:, None, :] - pos_out[None, :, :]
        dist = np.sqrt((diff**2).sum(axis=-1)).astype(np.float32)

        if not self.config.include_self:
            np.fill_diagonal(dist, np.nan)

        return {
            "dist": dist.astype(np.float32),
            "pos": pos_out.astype(np.float32),
            "centroid": centroid.astype(np.float32),
            "residue_ids": strc["residue_ids"].astype(np.str_),
            "chain_id": strc["chain_id"].astype(np.str_),
            "omitted_residue_ids": np.asarray(strc.get("omitted_residue_ids", []), dtype=np.str_),
            "resname": strc["resname"].astype(np.str_),
            "aa_index": strc["aa_index"].astype(np.int16),
            "plddt": strc["plddt"].astype(np.float32),
            "bfactor_ca": strc["bfactor_ca"].astype(np.float32),
        }
