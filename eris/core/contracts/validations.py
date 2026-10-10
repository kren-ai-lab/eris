from typing import Any, Dict

import numpy as np

from eris.core.contracts.granularity import GranularityContract


def validate_pos(data: Dict[str, Any]):
    if "pos" not in data:
        raise ValueError("Position validation: Missing 'pos'")

    if not isinstance(data["pos"], np.ndarray):
        raise TypeError("Position validation: 'pos' must be np.ndarray")

    if data["pos"].ndim != 2 or data["pos"].shape[1] != 3:
        raise ValueError("Position validation: 'pos' must have shape (N, 3)")

    if data["pos"].dtype != np.float32:
        raise TypeError("Position validation: 'pos' items must be np.float32")

    if not np.isfinite(data["pos"]).all():
        raise TypeError("Position validation: 'pos' items must be finite")


def validate_edge_index(data: Dict[str, Any]):
    if "edge_index" not in data:
        raise ValueError("Edge validation: missing 'edge_index'")

    edge_index = data["edge_index"]

    if edge_index.dtype != np.int64:
        raise TypeError("Edge validation: 'edge_index' must be int64")

    num_nodes = data["pos"].shape[0]

    if edge_index.size > 0:
        if edge_index.min() < 0 or edge_index.max() >= num_nodes:
            raise ValueError("GraphContract: edge_index contiene índices fuera de rango")


def validate_edge_features(data: Dict[str, Any], key: str = "edge_attr"):
    if key not in data:
        return

    edge_attr = data[key]

    if np.isinf(edge_attr).any():
        raise ValueError(f"Edge feature validation: {key} must not contain infinite values")


def validate_centroid(data: Dict[str, Any]):
    if "centroid" not in data:
        raise ValueError("Centroid validation: Missing 'centroid'")

    if not isinstance(data["centroid"], np.ndarray):
        raise TypeError("Centroid validation: 'centroid' must be np.ndarray")

    if data["centroid"].shape != (1, 3):
        raise ValueError("Centroid validation: 'centroid' must have shape (1, 3)")

    if data["centroid"].dtype != np.float32:
        raise TypeError("Centroid validation: 'centroid' items must be np.float32")

    if not np.isfinite(data["centroid"]).all():
        raise ValueError("Centroid validation: centroid contains non-finite values")


def validate_residue_data(data: Dict[str, Any]):
    required = ["residue_ids", "resname"]

    n = len(data["residue_ids"])
    if len(data["resname"]) != n:
        raise ValueError("Residue data validation: residue_ids and resname must have same length")

    for r in required:
        if r not in data:
            raise ValueError(f"Residue data validation: missing {r}")

        if data[r].ndim != 1:
            raise ValueError(f"Residue data validation: {r} must be 1D array")

        if len(data[r]) == 0:
            raise ValueError(f"Residue data validation: {r} cannot be empty")

        if not np.issubdtype(data[r].dtype, np.str_):
            raise TypeError(f"Residue data validation: {r} items must be np.str_")


def validate_grid(data: Dict[str, Any]):
    """
    Validate GRID representation contract.
    """

    # -------------------------
    # REQUIRED FIELDS
    # -------------------------
    required = ["grid"]

    for key in required:
        if key not in data:
            raise ValueError(f"Grid validation: missing required field '{key}'")

    grid = data["grid"]

    # -------------------------
    # GRID (core tensor)
    # -------------------------
    if not np.issubdtype(grid.dtype, np.floating):
        raise TypeError("Grid validation: 'grid' must be float dtype")

    if not np.isfinite(grid).all():
        raise ValueError("Grid validation: 'grid' contains non-finite values")

    # -------------------------
    # GRID SIZE
    # -------------------------
    if "grid_size" in data:
        gs = data["grid_size"]

        if not isinstance(gs, np.ndarray):
            raise TypeError("grid_size must be np.ndarray")

        if gs.shape != (1,) and gs.shape != (3,):
            raise ValueError("grid_size must have shape (1,) or (3,)")

        if not np.issubdtype(gs.dtype, np.integer):
            raise TypeError("grid_size must be integer dtype")

    # -------------------------
    # VOXEL SIZE
    # -------------------------
    if "voxel_size" in data:
        vs = data["voxel_size"]

        if not isinstance(vs, np.ndarray):
            raise TypeError("voxel_size must be np.ndarray")

        if vs.shape not in [(1,), (3,)]:
            raise ValueError("voxel_size must have shape (1,) or (3,)")

        if not np.issubdtype(vs.dtype, np.floating):
            raise TypeError("voxel_size must be float dtype")

        if not np.isfinite(vs).all():
            raise ValueError("voxel_size contains non-finite values")

    # -------------------------
    # ORIGIN
    # -------------------------
    if "origin" in data:
        origin = data["origin"]

        if not isinstance(origin, np.ndarray):
            raise TypeError("origin must be np.ndarray")

        if origin.shape != (3,):
            raise ValueError("origin must have shape (3,)")

        if not np.issubdtype(origin.dtype, np.floating):
            raise TypeError("origin must be float dtype")

        if not np.isfinite(origin).all():
            raise ValueError("origin contains non-finite values")

    return True


def validate_encoder_output(
    data: Dict[str, Any],
    representation_type: str,
    granularity_contract: GranularityContract,
) -> None:
    """Check explicit encoder output before Schema v2 strict validation.

    Only checks Schema v2 does not perform live here (dtypes, finiteness, index ranges,
    centroid and residue string arrays); shapes and cross-output alignment are left to it.
    """
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")
    if not data:
        raise ValueError("data cannot be empty")
    for k, v in data.items():
        if not isinstance(v, np.ndarray):
            raise TypeError(f"data[{k}] must be np.ndarray")

    granularity_contract.validate(data)

    if representation_type == "grid":
        validate_grid(data)
    else:
        validate_pos(data)
    if representation_type == "graph":
        validate_edge_index(data)
        validate_edge_features(data, "edge_attr")
    validate_centroid(data)
    validate_residue_data(data)
