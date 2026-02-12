from morpheo.core.registry import register_encoder

from morpheo.encoders.grids.voxel_grid_ca import VoxelGridCA


register_encoder(
    "voxel_grid_ca",
    lambda: VoxelGridCA(),
    type="grid",
    level="ca",
    family="grid",
    description="Voxel grid representation around CA coordinates (occupancy, density, pLDDT, B-factor, aa_onehot channels).",
    params_schema={
        "grid_size": {"type": "int", "default": 32},
        "voxel_size": {"type": ["null", "float"], "default": None},
        "fit": {"type": "bool", "default": True},
        "margin": {"type": "float", "default": 2.0},
        "center": {"type": "bool", "default": True},
        "channels": {
            "type": "str",
            "default": "occupancy",
            "allowed": ["occupancy", "density", "plddt", "bfactor", "aa_onehot"],
        },
        "sigma": {"type": "float", "default": 1.0},
        "pool": {"type": "str", "default": "sum", "allowed": ["sum", "mean", "max"]},
        "clamp": {"type": "bool", "default": False},
        "dtype": {"type": "str", "default": "float32", "allowed": ["float32", "float16"]},
    },
)
