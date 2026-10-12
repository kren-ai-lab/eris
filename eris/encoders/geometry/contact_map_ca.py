"""Binary CA contact-map encoder."""

from __future__ import annotations

from typing import Any, ClassVar, cast

import numpy as np

from eris.core.base_encoder import BaseResidueEncoder, EncodingContext
from eris.core.config import ResidueContactMapConfig
from eris.core.representations import MatrixRepresentation
from eris.core.representations.output_declarations import RESIDUE_CONTACT_MAP_OUTPUTS
from eris.encoders._context import prepared_arrays
from eris.encoders.geometry.distogram_ca import pairwise_distances, residue_fields


class ContactMapCA(BaseResidueEncoder):
    """Binary CA contact map under a distance cutoff."""

    # Identity
    name: ClassVar[str] = "residue_contact_map"
    version: ClassVar[str] = "0.1.2"
    representation: ClassVar = MatrixRepresentation("contact")
    output_declarations: ClassVar = RESIDUE_CONTACT_MAP_OUTPUTS

    config_type: ClassVar = ResidueContactMapConfig

    def _create_data(self, ctx: EncodingContext) -> dict[str, Any]:
        config = cast("ResidueContactMapConfig", self.config)
        contact = pairwise_distances(prepared_arrays(ctx)[2]) <= config.cutoff
        if not config.include_self:
            np.fill_diagonal(contact, val=False)
        if config.dtype != "bool":
            contact = contact.astype(np.uint8)
        return {"contact": contact, **residue_fields(ctx)}
