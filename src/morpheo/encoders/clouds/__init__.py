from morpheo.core.registry import register_encoder

from morpheo.encoders.clouds.residue_point_cloud_ca import ResiduePointCloudCA


register_encoder(
    "residue_point_cloud_ca",
    lambda: ResiduePointCloudCA(),
    type="cloud",
    level="ca",
    family="residue",
    desc="Residue point cloud from CA coordinates (+ residue ids, aa_index, plddt/bfactor if available).",
)
