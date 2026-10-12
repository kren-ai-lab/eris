from eris.core.config import AtomPointCloudConfig, ResiduePointCloudConfig
from eris.core.registry import register_encoder
from eris.encoders.clouds.atom_point_cloud import AtomPointCloud
from eris.encoders.clouds.residue_point_cloud import ResiduePointCloudCA

register_encoder(
    ResiduePointCloudConfig,
    ResiduePointCloudCA,
    description=(
        "Residue point cloud from CA coordinates (+ residue ids, aa_index, plddt/bfactor if available)."
    ),
)

register_encoder(
    AtomPointCloudConfig,
    AtomPointCloud,
    description="Atom point cloud.",
)
