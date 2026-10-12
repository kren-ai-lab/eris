"""Residue distogram and contact-map encoder registrations."""

from eris.core.config import (
    ResidueContactMapBinsConfig,
    ResidueContactMapConfig,
    ResidueDistogramBinsConfig,
    ResidueDistogramConfig,
)
from eris.core.registry import register_encoder
from eris.encoders.geometry.contact_map_bins_ca import ContactMapBinsCA
from eris.encoders.geometry.contact_map_ca import ContactMapCA
from eris.encoders.geometry.distogram_bins_ca import DistogramBinsCA
from eris.encoders.geometry.distogram_ca import DistogramCA

register_encoder(
    ResidueDistogramConfig,
    DistogramCA,
    description="CA distogram (float NxN pairwise distances).",
)

register_encoder(
    ResidueDistogramBinsConfig,
    DistogramBinsCA,
    description="Binned CA distogram (int NxN classes with configurable bin edges).",
)

register_encoder(
    ResidueContactMapConfig,
    ContactMapCA,
    description="Binary CA contact map (NxN) under a distance cutoff.",
)

register_encoder(
    ResidueContactMapBinsConfig,
    ContactMapBinsCA,
    description="Binned CA contact map (NxN classes; contact/no-contact + distance bins).",
)
