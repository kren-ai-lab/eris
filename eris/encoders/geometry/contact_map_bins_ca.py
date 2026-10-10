from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from eris.core.base_encoder import BaseResidueEncoder, EncodingContext
from eris.core.config import ResidueContactMapBinsConfig
from eris.core.representations import ContactMapBinsRepresentation
from eris.core.representations.output_declarations import RESIDUE_CONTACT_MAP_BINS_OUTPUTS
from eris.features.utils import validate_bin_edges

if TYPE_CHECKING:
    from eris.core.representations.base_rep import BaseRepresentation


def _validate_class_capacity(config: ResidueContactMapBinsConfig) -> None:
    num_bins = config.num_bins if config.edges is None else len(config.edges) - 1
    max_class_id = num_bins if config.add_no_contact_class else num_bins - 1
    capacity = np.iinfo(np.dtype(config.dtype)).max
    if max_class_id > capacity:
        raise ValueError(
            f"dtype {config.dtype} cannot represent {num_bins} bins with maximum "
            f"class ID {max_class_id}; maximum capacity is {capacity}"
        )


class ContactMapBinsCA(BaseResidueEncoder):
    # Identity
    name: ClassVar[str] = "residue_contact_map_bins"
    version: ClassVar[str] = "0.1.2"
    representation: ClassVar[BaseRepresentation] = ContactMapBinsRepresentation()
    output_declarations_v2: ClassVar = RESIDUE_CONTACT_MAP_BINS_OUTPUTS

    config_type: ClassVar = ResidueContactMapBinsConfig
    config: ResidueContactMapBinsConfig

    def _create_data(self, ctx: EncodingContext) -> dict[str, Any]:
        strc = ctx.raw_structure
        centroid = ctx.centroid
        pos_out = ctx.pos

        diff = pos_out[:, None, :] - pos_out[None, :, :]
        dist = np.sqrt((diff**2).sum(axis=-1)).astype(np.float32)

        if not self.config.include_self:
            np.fill_diagonal(dist, np.nan)

        if self.config.edges is not None:
            edges = validate_bin_edges(np.asarray(self.config.edges, dtype=np.float32))
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

        contact_bins = out.astype(np.dtype(self.config.dtype))

        return {
            "contact_bins": contact_bins,
            "bin_edges": edges.astype(np.float32),
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
