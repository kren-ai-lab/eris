"""Static scientific and adapter identity for the frozen GearNet-Edge backend."""

from __future__ import annotations

from typing import Any

from eris.core.learned.checkpoints import CheckpointDescriptor, StructuralModelDescriptor

GEARNET_EDGE_REPRESENTATION_NAME = "gearnet_edge"
GEARNET_EDGE_BACKEND_ID = "gearnet-edge"
GEARNET_EDGE_IMPLEMENTATION = "eris.core.learned.backends.gearnet.backend.GearNetEdgeBackend"
GEARNET_EDGE_ENCODER_VERSION = "3"

GEARNET_EDGE_ARCHITECTURE = "GearNet-Edge"
GEARNET_EDGE_CHECKPOINT_IDENTIFIER = "mc_gearnet_edge.pth"
GEARNET_EDGE_CHECKPOINT_SHA256 = "6dc11e2db6eff3c90f90672af227c647125adb7476de18a2bb860457bdb598ec"
GEARNET_EDGE_CHECKPOINT_REVISION = "Zenodo record v2"
GEARNET_EDGE_CHECKPOINT_SOURCE = "https://zenodo.org/records/7723075"
GEARNET_EDGE_CHECKPOINT_DOWNLOAD_URL = (
    "https://zenodo.org/records/7723075/files/mc_gearnet_edge.pth?download=1"
)
GEARNET_EDGE_NODE_INPUT_DIM = 21
GEARNET_EDGE_EDGE_INPUT_DIM = 59
GEARNET_EDGE_NUM_RELATION = 7
GEARNET_EDGE_NUM_ANGLE_BIN = 8
GEARNET_EDGE_HIDDEN_DIMS = (512, 512, 512, 512, 512, 512)

GEARNET_EDGE_SEQUENTIAL_MAX_DISTANCE = 2
GEARNET_EDGE_SPATIAL_RADIUS = 10.0
GEARNET_EDGE_SPATIAL_MIN_DISTANCE = 5
GEARNET_EDGE_SPATIAL_MAX_NEIGHBORS = 32
GEARNET_EDGE_KNN_K = 10
GEARNET_EDGE_KNN_MIN_DISTANCE = 5
GEARNET_EDGE_EDGE_FEATURE = "gearnet"

GEARNET_EDGE_CHECKPOINT_DESCRIPTOR = CheckpointDescriptor(
    identifier=GEARNET_EDGE_CHECKPOINT_IDENTIFIER,
    sha256=GEARNET_EDGE_CHECKPOINT_SHA256,
    revision=GEARNET_EDGE_CHECKPOINT_REVISION,
    source=GEARNET_EDGE_CHECKPOINT_SOURCE,
    download_url=GEARNET_EDGE_CHECKPOINT_DOWNLOAD_URL,
)
GEARNET_EDGE_MODEL_DESCRIPTOR = StructuralModelDescriptor(
    architecture=GEARNET_EDGE_ARCHITECTURE,
    checkpoint=GEARNET_EDGE_CHECKPOINT_DESCRIPTOR,
)

GEARNET_EDGE_CANONICAL_RESIDUES = (
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
)


def gearnet_edge_preprocessing_parameters() -> dict[str, Any]:
    """Return a fresh JSON-safe description of frozen v0.1 preprocessing."""

    return {
        "biological_selection": {
            "policy": "canonical_20_only",
            "residue_names": list(GEARNET_EDGE_CANONICAL_RESIDUES),
            "coordinate_source": "ca",
            "nonstandard_residues": "exclude",
            "mse": "exclude",
            "hetero": "exclude",
            "water": "exclude",
            "ligands": "exclude",
            "metals": "exclude",
            "ions": "exclude",
            "missing_retained_ca": "fail",
            "altloc": "first",
        },
        "graph_construction": {
            "node_layer": "alpha_carbon",
            "sequential_edge": {
                "max_distance": GEARNET_EDGE_SEQUENTIAL_MAX_DISTANCE,
            },
            "spatial_edge": {
                "radius": GEARNET_EDGE_SPATIAL_RADIUS,
                "min_distance": GEARNET_EDGE_SPATIAL_MIN_DISTANCE,
                "max_distance": None,
                "max_num_neighbors": GEARNET_EDGE_SPATIAL_MAX_NEIGHBORS,
            },
            "knn_edge": {
                "k": GEARNET_EDGE_KNN_K,
                "min_distance": GEARNET_EDGE_KNN_MIN_DISTANCE,
                "max_distance": None,
            },
            "edge_feature": GEARNET_EDGE_EDGE_FEATURE,
            "num_relation": GEARNET_EDGE_NUM_RELATION,
            "node_feature_width": GEARNET_EDGE_NODE_INPUT_DIM,
            "edge_feature_width": GEARNET_EDGE_EDGE_INPUT_DIM,
        },
    }


__all__ = [
    "GEARNET_EDGE_ARCHITECTURE",
    "GEARNET_EDGE_BACKEND_ID",
    "GEARNET_EDGE_CANONICAL_RESIDUES",
    "GEARNET_EDGE_CHECKPOINT_DESCRIPTOR",
    "GEARNET_EDGE_CHECKPOINT_DOWNLOAD_URL",
    "GEARNET_EDGE_CHECKPOINT_IDENTIFIER",
    "GEARNET_EDGE_CHECKPOINT_REVISION",
    "GEARNET_EDGE_CHECKPOINT_SHA256",
    "GEARNET_EDGE_CHECKPOINT_SOURCE",
    "GEARNET_EDGE_EDGE_FEATURE",
    "GEARNET_EDGE_EDGE_INPUT_DIM",
    "GEARNET_EDGE_ENCODER_VERSION",
    "GEARNET_EDGE_HIDDEN_DIMS",
    "GEARNET_EDGE_IMPLEMENTATION",
    "GEARNET_EDGE_KNN_K",
    "GEARNET_EDGE_KNN_MIN_DISTANCE",
    "GEARNET_EDGE_MODEL_DESCRIPTOR",
    "GEARNET_EDGE_NODE_INPUT_DIM",
    "GEARNET_EDGE_NUM_ANGLE_BIN",
    "GEARNET_EDGE_NUM_RELATION",
    "GEARNET_EDGE_REPRESENTATION_NAME",
    "GEARNET_EDGE_SEQUENTIAL_MAX_DISTANCE",
    "GEARNET_EDGE_SPATIAL_MAX_NEIGHBORS",
    "GEARNET_EDGE_SPATIAL_MIN_DISTANCE",
    "GEARNET_EDGE_SPATIAL_RADIUS",
    "gearnet_edge_preprocessing_parameters",
]
