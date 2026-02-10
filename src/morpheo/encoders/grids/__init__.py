from morpheo.core.registry import register_encoder

from morpheo.encoders.grids.voxel_grid_ca import VoxelGridCA


register_encoder(
    "voxel_grid_ca",
    lambda: VoxelGridCA(),
    type="grid",
    level="ca",
    family="residue",
    desc="Voxel grid representation around CA coordinates (occupancy/gaussian channels).",
)
