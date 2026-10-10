from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from eris.core.base_encoder import BaseResidueEncoder, EncodingContext
from eris.core.config import ResidueDistogramConfig
from eris.core.representations import MatrixRepresentation
from eris.core.representations.output_declarations import RESIDUE_DISTOGRAM_OUTPUTS

if TYPE_CHECKING:
    from eris.core.representations.base_rep import BaseRepresentation


def pairwise_distances(pos: np.ndarray) -> np.ndarray:
    diff = pos[:, None, :] - pos[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1)).astype(np.float32)


def residue_fields(ctx: EncodingContext) -> dict[str, Any]:
    """Per-residue outputs shared by the CA matrix encoders."""
    strc = ctx.raw_structure
    return {
        "pos": ctx.pos.astype(np.float32),
        "centroid": ctx.centroid.astype(np.float32),
        "residue_ids": strc["residue_ids"].astype(np.str_),
        "chain_id": strc["chain_id"].astype(np.str_),
        "omitted_residue_ids": np.asarray(strc.get("omitted_residue_ids", []), dtype=np.str_),
        "resname": strc["resname"].astype(np.str_),
        "aa_index": strc["aa_index"].astype(np.int16),
        "plddt": strc["plddt"].astype(np.float32),
        "bfactor_ca": strc["bfactor_ca"].astype(np.float32),
    }


class DistogramCA(BaseResidueEncoder):
    # Identity
    name: ClassVar[str] = "residue_distogram"
    version: ClassVar[str] = "0.1.2"
    representation: ClassVar[BaseRepresentation] = MatrixRepresentation("dist")
    output_declarations_v2: ClassVar = RESIDUE_DISTOGRAM_OUTPUTS

    config_type: ClassVar = ResidueDistogramConfig
    config: ResidueDistogramConfig

    def _create_data(self, ctx: EncodingContext) -> dict[str, Any]:
        dist = pairwise_distances(ctx.pos)
        if not self.config.include_self:
            np.fill_diagonal(dist, np.nan)
        return {"dist": dist, **residue_fields(ctx)}
