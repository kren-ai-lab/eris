from morpheo.core.registry import register_encoder

from morpheo.encoders.geometry.distogram_ca import DistogramCA
from morpheo.encoders.geometry.distogram_bins_ca import DistogramBinsCA
from morpheo.encoders.geometry.contact_map_ca import ContactMapCA
from morpheo.encoders.geometry.contact_map_bins_ca import ContactMapBinsCA


register_encoder(
    "distogram_ca",
    lambda: DistogramCA(),
    type="matrix",
    level="ca",
    family="residue",
    desc="CA distogram (float NxN distances).",
)

register_encoder(
    "distogram_bins_ca",
    lambda: DistogramBinsCA(),
    type="matrix",
    level="ca",
    family="residue",
    desc="Binned CA distogram (int NxN classes with configurable bin edges).",
)

register_encoder(
    "contact_map_ca",
    lambda: ContactMapCA(),
    type="matrix",
    level="ca",
    family="residue",
    desc="Binary CA contact map (NxN) under a distance cutoff.",
)

register_encoder(
    "contact_map_bins_ca",
    lambda: ContactMapBinsCA(),
    type="matrix",
    level="ca",
    family="residue",
    desc="Binned CA contact map (NxN classes; contact/no-contact + distance bins).",
)
