from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from eris.core.base_encoder import BaseResidueEncoder, EncodingContext
from eris.core.config import ResidueContactMapBinsConfig, ResidueDistogramBinsConfig
from eris.core.representations import DistogramBinsRepresentation
from eris.core.representations.output_declarations import RESIDUE_DISTOGRAM_BINS_OUTPUTS
from eris.encoders.geometry.distogram_ca import pairwise_distances, residue_fields
from eris.features.utils import validate_bin_edges

if TYPE_CHECKING:
    from eris.core.representations.base_rep import BaseRepresentation


def binned_distance_data(
    ctx: EncodingContext,
    config: ResidueDistogramBinsConfig | ResidueContactMapBinsConfig,
    key: str,
) -> dict[str, Any]:
    """Shared binning for the binned distogram and binned contact map encoders."""
    dist = pairwise_distances(ctx.pos)

    if not config.include_self:
        np.fill_diagonal(dist, np.nan)

    if config.edges is not None:
        edges = validate_bin_edges(np.asarray(config.edges, dtype=np.float32))
    else:
        edges = np.linspace(config.dmin, config.dmax, config.num_bins + 1, dtype=np.float32)

    num_bins = int(edges.size - 1)
    raw_bins = np.digitize(dist, edges, right=False) - 1
    raw_bins = np.clip(raw_bins, 0, num_bins - 1)

    if config.add_no_contact_class:
        out = (raw_bins + 1).astype(np.int32)
        out[dist > float(edges[-1])] = 0
        out[np.isnan(dist)] = 0
    else:
        out = raw_bins.astype(np.int32)
        out[np.isnan(dist)] = 0

    return {
        key: out.astype(np.dtype(config.dtype)),
        "bin_edges": edges.astype(np.float32),
        **residue_fields(ctx),
    }


class DistogramBinsCA(BaseResidueEncoder):
    # Identity
    name: ClassVar[str] = "residue_distogram_bins"
    version: ClassVar[str] = "0.1.2"
    representation: ClassVar[BaseRepresentation] = DistogramBinsRepresentation()
    output_declarations_v2: ClassVar = RESIDUE_DISTOGRAM_BINS_OUTPUTS

    config_type: ClassVar = ResidueDistogramBinsConfig
    config: ResidueDistogramBinsConfig

    def _create_data(self, ctx: EncodingContext) -> dict[str, Any]:
        return binned_distance_data(ctx, self.config, "dist_bins")
