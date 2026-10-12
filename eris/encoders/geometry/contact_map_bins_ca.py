from __future__ import annotations

from typing import Any, ClassVar

from eris.core.base_encoder import BaseResidueEncoder, EncodingContext
from eris.core.config import ResidueContactMapBinsConfig
from eris.core.representations import MatrixRepresentation
from eris.core.representations.output_declarations import RESIDUE_CONTACT_MAP_BINS_OUTPUTS
from eris.encoders.geometry.distogram_bins_ca import (
    binned_distance_data,
    validate_class_capacity,
)


class ContactMapBinsCA(BaseResidueEncoder):
    # Identity
    name: ClassVar[str] = "residue_contact_map_bins"
    version: ClassVar[str] = "0.1.2"
    representation: ClassVar = MatrixRepresentation("contact_bins")
    output_declarations: ClassVar = RESIDUE_CONTACT_MAP_BINS_OUTPUTS

    config_type: ClassVar = ResidueContactMapBinsConfig
    config: ResidueContactMapBinsConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        validate_class_capacity(self.config)

    def _create_data(self, ctx: EncodingContext) -> dict[str, Any]:
        return binned_distance_data(ctx, self.config, "contact_bins")
