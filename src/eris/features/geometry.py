from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np


FRAME_EPSILON = 1e-6


def signed_dihedral(
    point_0: Any,
    point_1: Any,
    point_2: Any,
    point_3: Any,
) -> float:
    """Return the signed dihedral in radians on ``[-pi, pi)``.

    Missing, non-finite, or degenerate geometry is represented by ``NaN``.
    """
    points = np.asarray([point_0, point_1, point_2, point_3], dtype=np.float64)
    if points.shape != (4, 3) or not np.isfinite(points).all():
        return float("nan")

    first = points[0] - points[1]
    axis = points[2] - points[1]
    last = points[3] - points[2]
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm == 0.0:
        return float("nan")

    axis_unit = axis / axis_norm
    first_normal = first - np.dot(first, axis_unit) * axis_unit
    last_normal = last - np.dot(last, axis_unit) * axis_unit
    first_norm = float(np.linalg.norm(first_normal))
    last_norm = float(np.linalg.norm(last_normal))
    if first_norm == 0.0 or last_norm == 0.0:
        return float("nan")

    first_unit = first_normal / first_norm
    last_unit = last_normal / last_norm
    x = float(np.dot(first_unit, last_unit))
    y = float(np.dot(np.cross(axis_unit, first_unit), last_unit))
    angle = float(np.arctan2(y, x))
    if angle >= np.pi:
        angle -= 2.0 * np.pi
    return angle


def local_frames(
    n_coord: np.ndarray,
    ca_coord: np.ndarray,
    c_coord: np.ndarray,
    n_valid: np.ndarray,
    ca_valid: np.ndarray,
    c_valid: np.ndarray,
    *,
    frame_epsilon: float = FRAME_EPSILON,
) -> tuple[np.ndarray, np.ndarray]:
    """Build aligned right-handed N-CA-C frames with axes stored as columns."""
    n_coord = np.asarray(n_coord, dtype=np.float64)
    ca_coord = np.asarray(ca_coord, dtype=np.float64)
    c_coord = np.asarray(c_coord, dtype=np.float64)
    n_rows = n_coord.shape[0]
    if any(coordinates.shape != (n_rows, 3) for coordinates in (n_coord, ca_coord, c_coord)):
        raise ValueError("backbone N, CA, and C coordinates must have shape (N, 3)")

    masks = tuple(np.asarray(mask, dtype=bool).reshape(-1) for mask in (n_valid, ca_valid, c_valid))
    if any(mask.shape != (n_rows,) for mask in masks):
        raise ValueError("backbone atom validity masks must have shape (N,)")
    if (
        isinstance(frame_epsilon, bool)
        or not isinstance(frame_epsilon, Real)
        or not np.isfinite(frame_epsilon)
        or frame_epsilon <= 0.0
    ):
        raise ValueError("frame_epsilon must be a finite positive number")

    frames = np.full((n_rows, 3, 3), np.nan, dtype=np.float64)
    valid = masks[0] & masks[1] & masks[2]
    valid &= np.isfinite(n_coord).all(axis=1)
    valid &= np.isfinite(ca_coord).all(axis=1)
    valid &= np.isfinite(c_coord).all(axis=1)

    for index in np.flatnonzero(valid):
        x_axis = c_coord[index] - ca_coord[index]
        x_norm = float(np.linalg.norm(x_axis))
        if x_norm < frame_epsilon:
            valid[index] = False
            continue
        x_axis /= x_norm

        n_vector = n_coord[index] - ca_coord[index]
        y_axis = n_vector - np.dot(n_vector, x_axis) * x_axis
        y_norm = float(np.linalg.norm(y_axis))
        if y_norm < frame_epsilon:
            valid[index] = False
            continue
        y_axis /= y_norm

        z_axis = np.cross(x_axis, y_axis)
        z_norm = float(np.linalg.norm(z_axis))
        if z_norm < frame_epsilon:
            valid[index] = False
            continue
        z_axis /= z_norm
        frames[index] = np.column_stack((x_axis, y_axis, z_axis))

    frames[~valid] = np.nan
    return frames.astype(np.float32), valid.astype(bool)


def relative_directions(
    edge_index: np.ndarray,
    ca_coord: np.ndarray,
    ca_valid: np.ndarray,
    frames: np.ndarray,
    frame_valid: np.ndarray,
) -> np.ndarray:
    """Express directed unit CA displacements in each source residue's local frame."""
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E), got {edge_index.shape}")

    ca_coord = np.asarray(ca_coord, dtype=np.float64)
    n_rows = ca_coord.shape[0]
    if ca_coord.shape != (n_rows, 3):
        raise ValueError("CA coordinates must have shape (N, 3)")
    ca_valid = np.asarray(ca_valid, dtype=bool).reshape(-1)
    frame_valid = np.asarray(frame_valid, dtype=bool).reshape(-1)
    frames = np.asarray(frames, dtype=np.float64)
    if ca_valid.shape != (n_rows,) or frame_valid.shape != (n_rows,):
        raise ValueError("CA and frame validity masks must have shape (N,)")
    if frames.shape != (n_rows, 3, 3):
        raise ValueError("local frames must have shape (N, 3, 3)")
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= n_rows):
        raise ValueError("edge_index contains a node index outside the coordinate rows")

    n_edges = edge_index.shape[1]
    directions = np.full((n_edges, 3), np.nan, dtype=np.float64)
    source, target = edge_index
    for edge in range(n_edges):
        source_index = source[edge]
        target_index = target[edge]
        if not (frame_valid[source_index] and ca_valid[source_index] and ca_valid[target_index]):
            continue
        displacement = ca_coord[target_index] - ca_coord[source_index]
        if not np.isfinite(displacement).all():
            continue
        distance = float(np.linalg.norm(displacement))
        if distance == 0.0:
            continue
        unit_displacement = displacement / distance
        directions[edge] = frames[source_index].T @ unit_displacement

    return directions.astype(np.float32)


def peptide_continuity(
    c_coord: np.ndarray,
    n_coord: np.ndarray,
    c_valid: np.ndarray,
    n_valid: np.ndarray,
    chain_id: np.ndarray,
    structural_residue_index: np.ndarray,
    *,
    max_peptide_bond_distance: float = 2.0,
) -> np.ndarray:
    """Return whether each aligned residue pair ``i -> i+1`` is peptide-connected."""
    c_coord = np.asarray(c_coord, dtype=np.float64)
    n_coord = np.asarray(n_coord, dtype=np.float64)
    chain_id = np.asarray(chain_id).reshape(-1)
    structural_residue_index = np.asarray(structural_residue_index, dtype=np.int64).reshape(-1)
    n_rows = chain_id.shape[0]

    if c_coord.shape != (n_rows, 3) or n_coord.shape != (n_rows, 3):
        raise ValueError("backbone C and N coordinates must have shape (N, 3)")
    c_valid = np.asarray(c_valid, dtype=bool).reshape(-1)
    n_valid = np.asarray(n_valid, dtype=bool).reshape(-1)
    if c_valid.shape != (n_rows,) or n_valid.shape != (n_rows,):
        raise ValueError("backbone C and N validity masks must have shape (N,)")
    if structural_residue_index.shape != (n_rows,):
        raise ValueError("structural residue indices must have shape (N,)")
    if not np.isfinite(max_peptide_bond_distance) or max_peptide_bond_distance <= 0.0:
        raise ValueError("max_peptide_bond_distance must be a finite positive number")

    connected = np.zeros((max(n_rows - 1, 0),), dtype=bool)
    if n_rows < 2:
        return connected

    candidates = (
        (chain_id[:-1] == chain_id[1:])
        & (structural_residue_index[1:] == structural_residue_index[:-1] + 1)
        & c_valid[:-1]
        & n_valid[1:]
        & np.isfinite(c_coord[:-1]).all(axis=1)
        & np.isfinite(n_coord[1:]).all(axis=1)
    )
    candidate_indices = np.flatnonzero(candidates)
    if candidate_indices.size:
        distances = np.linalg.norm(
            c_coord[candidate_indices] - n_coord[candidate_indices + 1],
            axis=1,
        )
        connected[candidate_indices] = distances <= float(max_peptide_bond_distance)
    return connected


def backbone_torsions(
    n_coord: np.ndarray,
    ca_coord: np.ndarray,
    c_coord: np.ndarray,
    n_valid: np.ndarray,
    ca_valid: np.ndarray,
    c_valid: np.ndarray,
    chain_id: np.ndarray,
    structural_residue_index: np.ndarray,
    *,
    max_peptide_bond_distance: float = 2.0,
) -> np.ndarray:
    """Compute aligned ``phi``, ``psi``, and ``omega`` signed-radian columns."""
    n_coord = np.asarray(n_coord, dtype=np.float64)
    ca_coord = np.asarray(ca_coord, dtype=np.float64)
    c_coord = np.asarray(c_coord, dtype=np.float64)
    n_rows = n_coord.shape[0]
    if n_coord.shape != (n_rows, 3) or ca_coord.shape != (n_rows, 3):
        raise ValueError("backbone N and CA coordinates must have shape (N, 3)")

    n_valid = np.asarray(n_valid, dtype=bool).reshape(-1)
    ca_valid = np.asarray(ca_valid, dtype=bool).reshape(-1)
    c_valid = np.asarray(c_valid, dtype=bool).reshape(-1)
    if any(mask.shape != (n_rows,) for mask in (n_valid, ca_valid, c_valid)):
        raise ValueError("backbone atom validity masks must have shape (N,)")

    connected = peptide_continuity(
        c_coord,
        n_coord,
        c_valid,
        n_valid,
        chain_id,
        structural_residue_index,
        max_peptide_bond_distance=max_peptide_bond_distance,
    )
    torsions = np.full((n_rows, 3), np.nan, dtype=np.float64)

    for index in range(n_rows):
        if (
            index > 0
            and connected[index - 1]
            and n_valid[index]
            and ca_valid[index]
            and c_valid[index]
        ):
            torsions[index, 0] = signed_dihedral(
                c_coord[index - 1],
                n_coord[index],
                ca_coord[index],
                c_coord[index],
            )

        if index + 1 < n_rows and connected[index]:
            if n_valid[index] and ca_valid[index] and c_valid[index] and n_valid[index + 1]:
                torsions[index, 1] = signed_dihedral(
                    n_coord[index],
                    ca_coord[index],
                    c_coord[index],
                    n_coord[index + 1],
                )
            if ca_valid[index] and c_valid[index] and n_valid[index + 1] and ca_valid[index + 1]:
                torsions[index, 2] = signed_dihedral(
                    ca_coord[index],
                    c_coord[index],
                    n_coord[index + 1],
                    ca_coord[index + 1],
                )

    return torsions.astype(np.float32)


__all__ = [
    "FRAME_EPSILON",
    "backbone_torsions",
    "local_frames",
    "peptide_continuity",
    "relative_directions",
    "signed_dihedral",
]
