from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from eris.core.base_encoder import BaseResidueEncoder, EncodingContext
from eris.core.config import ResidueContactMapConfig
from eris.core.representations import MatrixRepresentation
from eris.core.representations.output_declarations import RESIDUE_CONTACT_MAP_OUTPUTS
from eris.encoders.geometry.distogram_ca import pairwise_distances, residue_fields

if TYPE_CHECKING:
    from eris.core.representations.base_rep import BaseRepresentation


class ContactMapCA(BaseResidueEncoder):
    # Identity
    name: ClassVar[str] = "residue_contact_map"
    version: ClassVar[str] = "0.1.2"
    representation: ClassVar[BaseRepresentation] = MatrixRepresentation("contact")
    output_declarations_v2: ClassVar = RESIDUE_CONTACT_MAP_OUTPUTS

    config_type: ClassVar = ResidueContactMapConfig
    config: ResidueContactMapConfig

    def _create_data(self, ctx: EncodingContext) -> dict[str, Any]:
        contact = pairwise_distances(ctx.pos) <= self.config.cutoff
        if not self.config.include_self:
            np.fill_diagonal(contact, False)
        if self.config.dtype != "bool":
            contact = contact.astype(np.uint8)
        return {"contact": contact, **residue_fields(ctx)}
