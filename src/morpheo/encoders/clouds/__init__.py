from morpheo.core.registry import register_encoder

from morpheo.encoders.clouds.residue_point_cloud_ca import ResiduePointCloudCA
from morpheo.encoders.clouds.atom_point_cloud import AtomPointCloud  # noqa: F401

register_encoder(
    "residue_point_cloud_ca",
    lambda: ResiduePointCloudCA(),
    type="cloud",
    level="ca",
    family="cloud",
    description="Residue point cloud from CA coordinates (+ residue ids, aa_index, plddt/bfactor if available).",
    params_schema={
        "center": {"type": "bool", "default": True},
        "scale": {"type": ["null", "float"], "default": None},
    },
)

register_encoder(
    "atom_point_cloud",
    lambda: AtomPointCloud(),
    type="cloud",
    level="atom",
    family="cloud",
    description="Atom point cloud.",
    params_schema={
        "include_hetero": {"type": "bool", "default": False},
        "include_h": {"type": "bool", "default": False},
        "altloc": {"type": "str", "default": "first"},
        "amino_acids_only": {"type": "bool", "default": False},
        "mode": {"type": "str", "default": "none"},
    },
)
