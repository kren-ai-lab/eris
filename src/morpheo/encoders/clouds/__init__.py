from morpheo.core.registry import register_encoder

from morpheo.encoders.clouds.residue_point_cloud_ca import ResiduePointCloudCA
from morpheo.encoders.clouds.atom_point_cloud import AtomPointCloud  # noqa: F401

register_encoder(
    "residue_point_cloud_ca",
    lambda: ResiduePointCloudCA(),
    type="cloud",
    level="ca",
    family="residue",
    desc="Residue point cloud from CA coordinates (+ residue ids, aa_index, plddt/bfactor if available).",
)

### Checking
register_encoder(
    "atom_point_cloud",
    lambda: AtomPointCloud(),
    type="cloud",
    level="atom",
    family="atom",
    desc="Atom point cloud.",
)

