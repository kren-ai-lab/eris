from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np

from eris.features.utils import parse_mode

INTERFACE_DEFAULT_CUTOFF = 5.0
_ATOM_SELECTION = "selected_polymer_heavy_atoms"
_CHAIN_RELATION = "different_selected_chains_only"
_DISTANCE_DEFINITION = "minimum_inter_residue_atom_distance"


def normalize_interface_params(params: dict[str, Any]) -> dict[str, Any]:
    fixed = {
        "atom_selection": _ATOM_SELECTION,
        "chain_relation": _CHAIN_RELATION,
        "cutoff_inclusive": True,
        "distance_definition": _DISTANCE_DEFINITION,
    }
    allowed = {*fixed, "cutoff"}
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(f"Unknown interface parameters: {unknown}")

    cutoff = params.get("cutoff", INTERFACE_DEFAULT_CUTOFF)
    if isinstance(cutoff, bool) or not isinstance(cutoff, Real):
        raise TypeError("interface parameter 'cutoff' must be a real number")
    cutoff = float(cutoff)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("interface parameter 'cutoff' must be a finite positive number")

    for name, expected in fixed.items():
        if name in params and params[name] != expected:
            raise ValueError(f"interface parameter '{name}' must be {expected!r}")
    return {**fixed, "cutoff": cutoff}


def resolve_interface_cutoff(node_mode: Any, edge_mode: Any) -> float | None:
    node = dict(parse_mode(node_mode)).get("interface")
    edge = dict(parse_mode(edge_mode)).get("interface_contact")
    if (node is None) != (edge is None):
        raise ValueError(
            "residue graph interface characterization requires both node token "
            "'interface' and edge token 'interface_contact'"
        )
    if node is None:
        return None
    if node["cutoff"] != edge["cutoff"]:
        raise ValueError("interface and interface_contact must use the same cutoff")
    return float(node["cutoff"])


def compute_interface_contacts(
    heavy_atom_coordinates: np.ndarray,
    heavy_atom_residue_index: np.ndarray,
    chain_id: np.ndarray,
    *,
    cutoff: float = INTERFACE_DEFAULT_CUTOFF,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned residue flags and ordered contacting residue-index pairs."""
    coordinates = np.asarray(heavy_atom_coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("heavy atom coordinates must have shape (M, 3)")
    residue_index = np.asarray(heavy_atom_residue_index, dtype=np.int64).reshape(-1)
    if residue_index.shape != (coordinates.shape[0],):
        raise ValueError("heavy atom residue indices must have shape (M,)")
    chains = np.asarray(chain_id).reshape(-1)
    n_residues = chains.shape[0]
    if residue_index.size and (residue_index.min() < 0 or residue_index.max() >= n_residues):
        raise ValueError("heavy atom residue index is outside the aligned residue rows")
    normalized = normalize_interface_params({"cutoff": cutoff})
    cutoff_squared = normalized["cutoff"] ** 2

    residue_atoms = [
        coordinates[(residue_index == index) & np.isfinite(coordinates).all(axis=1)]
        for index in range(n_residues)
    ]
    flags = np.zeros((n_residues,), dtype=bool)
    pairs: list[tuple[int, int]] = []
    for source in range(n_residues):
        source_atoms = residue_atoms[source]
        if source_atoms.shape[0] == 0:
            continue
        for target in range(source + 1, n_residues):
            if chains[source] == chains[target]:
                continue
            target_atoms = residue_atoms[target]
            if target_atoms.shape[0] == 0:
                continue
            displacement = source_atoms[:, None, :] - target_atoms[None, :, :]
            squared_distance = np.einsum("ijk,ijk->ij", displacement, displacement)
            if np.any(squared_distance <= cutoff_squared):
                flags[source] = True
                flags[target] = True
                pairs.append((source, target))

    return flags, np.asarray(pairs, dtype=np.int64).reshape((-1, 2))


def interface_edge_flags(edge_index: np.ndarray, contact_pairs: np.ndarray) -> np.ndarray:
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E)")
    pairs = np.asarray(contact_pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("contact_pairs must have shape (K, 2)")
    contacts = {tuple(pair) for pair in pairs.tolist()}
    contacts.update((target, source) for source, target in tuple(contacts))
    return np.asarray(
        [(int(source), int(target)) in contacts for source, target in edges.T],
        dtype=bool,
    )


__all__ = [
    "INTERFACE_DEFAULT_CUTOFF",
    "compute_interface_contacts",
    "interface_edge_flags",
    "normalize_interface_params",
    "resolve_interface_cutoff",
]
