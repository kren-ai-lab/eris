from __future__ import annotations

from typing import Any

import numpy as np


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


__all__ = ["backbone_torsions", "peptide_continuity", "signed_dihedral"]
