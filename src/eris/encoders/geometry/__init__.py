from eris.core.registry import register_encoder

from eris.encoders.geometry.distogram_ca import DistogramCA
from eris.encoders.geometry.distogram_bins_ca import DistogramBinsCA
from eris.encoders.geometry.contact_map_ca import ContactMapCA
from eris.encoders.geometry.contact_map_bins_ca import ContactMapBinsCA


register_encoder(
    "distogram_residue",
    lambda: DistogramCA(),
    type="matrix",
    level="ca",
    description="CA distogram (float NxN pairwise distances).",
    params_schema={
        "include_self": {"type": "bool", "default": True},
    },
)

register_encoder(
    "distogram_bins_residue",
    lambda: DistogramBinsCA(),
    type="matrix",
    level="ca",
    description="Binned CA distogram (int NxN classes with configurable bin edges).",
    params_schema={
        "include_self": {"type": "bool", "default": True},
        "add_no_contact_class": {"type": "bool", "default": False},
        "dmin": {"type": "float", "default": 2.0},
        "dmax": {"type": "float", "default": 20.0},
        "num_bins": {"type": "int", "default": 32},
        "dtype": {"type": "str", "default": "uint8"},
    },
)

register_encoder(
    "contact_map_residue",
    lambda: ContactMapCA(),
    type="matrix",
    level="ca",
    description="Binary CA contact map (NxN) under a distance cutoff.",
    params_schema={
        "cutoff": {"type": "float", "default": 8.0},
        "include_self": {"type": "bool", "default": False},
        "dtype": {"type": "str", "default": "bool"},
    },
)

register_encoder(
    "contact_map_bins_residue",
    lambda: ContactMapBinsCA(),
    type="matrix",
    level="ca",
    description="Binned CA contact map (NxN classes; contact/no-contact + distance bins).",
    params_schema={
        "include_self": {"type": "bool", "default": False},
        "add_no_contact_class": {"type": "bool", "default": True},
        "dmin": {"type": "float", "default": 2.0},
        "dmax": {"type": "float", "default": 20.0},
        "num_bins": {"type": "int", "default": 32},
        "dtype": {"type": "str", "default": "uint8"},
    },
)
