from eris.core.config import ResidueVoxelGridConfig
from eris.core.registry import register_encoder
from eris.encoders.grids.voxel_grid_ca import VoxelGridCA

register_encoder(
    ResidueVoxelGridConfig,
    VoxelGridCA,
    description=(
        "Voxel grid representation around CA coordinates "
        "(occupancy, density, pLDDT, B-factor, aa_onehot channels)."
    ),
)
