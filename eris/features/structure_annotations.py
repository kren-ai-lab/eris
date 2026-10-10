from __future__ import annotations

from importlib import metadata
from typing import Any

import numpy as np

from eris.features.geometry import peptide_continuity

BIOTITE_PACKAGE = "biotite"
BIOTITE_VERSION = metadata.version(BIOTITE_PACKAGE)
SECONDARY_STRUCTURE_KEYS = [
    "secondary_structure.helix",
    "secondary_structure.sheet",
    "secondary_structure.coil",
    "secondary_structure.unassigned",
]
SECONDARY_STRUCTURE_STATE_MAPPING = {
    "a": "helix",
    "b": "sheet",
    "c": "coil",
    "": "unassigned",
}

TIEN_MAXIMUM_ASA = {
    "ALA": 129.0,
    "ARG": 274.0,
    "ASN": 195.0,
    "ASP": 193.0,
    "CYS": 167.0,
    "GLN": 225.0,
    "GLU": 223.0,
    "GLY": 104.0,
    "HIS": 224.0,
    "ILE": 197.0,
    "LEU": 201.0,
    "LYS": 236.0,
    "MET": 224.0,
    "PHE": 240.0,
    "PRO": 159.0,
    "SER": 155.0,
    "THR": 172.0,
    "TRP": 285.0,
    "TYR": 263.0,
    "VAL": 174.0,
}

# Heavy atoms expected for a complete canonical residue. Terminal OXT is
# optional and additional atoms do not make an otherwise complete residue
# incomplete.
CANONICAL_REQUIRED_HEAVY_ATOMS = {
    "ALA": {"N", "CA", "C", "O", "CB"},
    "ARG": {"N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"},
    "ASN": {"N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"},
    "ASP": {"N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"},
    "CYS": {"N", "CA", "C", "O", "CB", "SG"},
    "GLN": {"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"},
    "GLU": {"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"},
    "GLY": {"N", "CA", "C", "O"},
    "HIS": {"N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"},
    "ILE": {"N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"},
    "LEU": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"},
    "LYS": {"N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"},
    "MET": {"N", "CA", "C", "O", "CB", "CG", "SD", "CE"},
    "PHE": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "PRO": {"N", "CA", "C", "O", "CB", "CG", "CD"},
    "SER": {"N", "CA", "C", "O", "CB", "OG"},
    "THR": {"N", "CA", "C", "O", "CB", "OG1", "CG2"},
    "TRP": {
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
    },
    "TYR": {
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "OH",
    },
    "VAL": {"N", "CA", "C", "O", "CB", "CG1", "CG2"},
}


def _load_biotite() -> Any:
    import biotite.structure as structure

    return structure


def secondary_structure_onehot(
    *,
    ca_coord: np.ndarray,
    ca_valid: np.ndarray,
    c_coord: np.ndarray,
    n_coord: np.ndarray,
    c_valid: np.ndarray,
    n_valid: np.ndarray,
    chain_id: np.ndarray,
    structural_residue_index: np.ndarray,
    resname: np.ndarray,
    max_peptide_bond_distance: float,
) -> np.ndarray:
    """Return aligned helix/sheet/coil/unassigned P-SEA one-hot rows."""
    structure = _load_biotite()
    ca_coord = np.asarray(ca_coord, dtype=np.float32)
    ca_valid = np.asarray(ca_valid, dtype=bool).reshape(-1)
    resname = np.asarray(resname, dtype=np.str_).reshape(-1)
    n_rows = resname.shape[0]
    if ca_coord.shape != (n_rows, 3) or ca_valid.shape != (n_rows,):
        raise ValueError("CA coordinates and validity must align with residue names")
    usable_ca = ca_valid & np.isfinite(ca_coord).all(axis=1)

    connected = peptide_continuity(
        c_coord,
        n_coord,
        c_valid,
        n_valid,
        chain_id,
        structural_residue_index,
        max_peptide_bond_distance=max_peptide_bond_distance,
    )
    segment_starts = np.r_[0, np.flatnonzero(~connected) + 1]
    segment_ends = np.r_[segment_starts[1:], n_rows]
    states = np.full((n_rows,), "", dtype="U1")

    for start, end in zip(segment_starts, segment_ends, strict=True):
        length = int(end - start)
        atom_array = structure.AtomArray(length)
        atom_array.coord = np.where(
            usable_ca[start:end, None],
            ca_coord[start:end],
            np.nan,
        ).astype(np.float32)
        atom_array.chain_id[:] = "A"
        atom_array.res_id[:] = np.arange(1, length + 1)
        atom_array.res_name[:] = resname[start:end]
        atom_array.atom_name[:] = np.where(usable_ca[start:end], "CA", "N")
        atom_array.element[:] = np.where(usable_ca[start:end], "C", "N")
        atom_array.hetero[:] = False
        assigned = np.asarray(structure.annotate_sse(atom_array), dtype="U1")
        if assigned.shape != (length,):
            raise RuntimeError(
                "Biotite annotate_sse returned an assignment that does not align with residues"
            )
        states[start:end] = assigned

    output = np.zeros((n_rows, 4), dtype=np.float32)
    column = {"a": 0, "b": 1, "c": 2, "": 3}
    for index, state in enumerate(states.tolist()):
        try:
            output[index, column[state]] = 1.0
        except KeyError as error:
            raise RuntimeError(f"Biotite annotate_sse returned unknown state {state!r}") from error
    return output


def _biotite_atom_array(
    structure: Any,
    *,
    coordinates: np.ndarray,
    residue_index: np.ndarray,
    atom_name: np.ndarray,
    element: np.ndarray,
    resname: np.ndarray,
    chain_id: np.ndarray,
) -> Any:
    n_atoms = coordinates.shape[0]
    atom_array = structure.AtomArray(n_atoms)
    atom_array.coord = coordinates.astype(np.float32, copy=False)
    atom_array.chain_id[:] = chain_id[residue_index]
    atom_array.res_id[:] = residue_index + 1
    atom_array.res_name[:] = resname[residue_index]
    atom_array.atom_name[:] = atom_name
    atom_array.element[:] = element
    atom_array.hetero[:] = False
    return atom_array


def solvent_accessibility(
    *,
    coordinates: np.ndarray,
    residue_index: np.ndarray,
    atom_name: np.ndarray,
    element: np.ndarray,
    resname: np.ndarray,
    chain_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute residue-aligned Shrake-Rupley SASA and Tien RSA."""
    structure = _load_biotite()
    from biotite.structure.info import vdw_radius_protor

    coordinates = np.asarray(coordinates, dtype=np.float32)
    residue_index = np.asarray(residue_index, dtype=np.int64).reshape(-1)
    atom_name = np.asarray(atom_name, dtype=np.str_).reshape(-1)
    element = np.asarray(element, dtype=np.str_).reshape(-1)
    resname = np.asarray(resname, dtype=np.str_).reshape(-1)
    chain_id = np.asarray(chain_id, dtype=np.str_).reshape(-1)
    n_rows = resname.shape[0]
    n_atoms = coordinates.shape[0]
    if coordinates.shape != (n_atoms, 3):
        raise ValueError("polymer heavy-atom coordinates must have shape (M, 3)")
    if any(values.shape != (n_atoms,) for values in (residue_index, atom_name, element)):
        raise ValueError("polymer heavy-atom metadata must align with coordinates")
    if chain_id.shape != (n_rows,):
        raise ValueError("chain IDs must align with residue names")
    if n_atoms and (residue_index.min() < 0 or residue_index.max() >= n_rows):
        raise ValueError("polymer heavy-atom residue index is outside the residue rows")
    if not np.isfinite(coordinates).all():
        raise ValueError("SASA requires finite selected polymer heavy-atom coordinates")

    for atom in range(n_atoms):
        residue_name = str(resname[residue_index[atom]])
        name = str(atom_name[atom])
        try:
            radius = vdw_radius_protor(residue_name, name)
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"ProtOr radius is unavailable for selected polymer atom {residue_name}:{name}"
            ) from error
        if radius is None or not np.isfinite(radius):
            raise ValueError(
                f"ProtOr radius is unavailable for selected polymer atom {residue_name}:{name}"
            )

    if n_atoms:
        atom_array = _biotite_atom_array(
            structure,
            coordinates=coordinates,
            residue_index=residue_index,
            atom_name=atom_name,
            element=element,
            resname=resname,
            chain_id=chain_id,
        )
        atom_sasa = np.asarray(
            structure.sasa(
                atom_array,
                probe_radius=1.4,
                atom_filter=None,
                ignore_ions=True,
                point_number=1000,
                point_distr="Fibonacci",
                vdw_radii="ProtOr",
            ),
            dtype=np.float64,
        )
        if atom_sasa.shape != (n_atoms,):
            raise RuntimeError("Biotite sasa returned values that do not align with selected atoms")
        if not np.isfinite(atom_sasa).all() or np.any(atom_sasa < 0.0):
            raise RuntimeError("Biotite sasa returned non-finite or negative atom areas")
    else:
        atom_sasa = np.empty((0,), dtype=np.float64)

    sasa = np.zeros((n_rows,), dtype=np.float64)
    atom_counts = np.zeros((n_rows,), dtype=np.int64)
    np.add.at(sasa, residue_index, atom_sasa)
    np.add.at(atom_counts, residue_index, 1)

    observed_names = [set() for _ in range(n_rows)]
    for index, name in zip(residue_index.tolist(), atom_name.tolist(), strict=True):
        observed_names[index].add(name)

    incomplete = atom_counts == 0
    for index, residue_name in enumerate(resname.tolist()):
        required = CANONICAL_REQUIRED_HEAVY_ATOMS.get(residue_name)
        if required is not None and not required.issubset(observed_names[index]):
            incomplete[index] = True
    sasa[incomplete] = np.nan

    rsa = np.full((n_rows,), np.nan, dtype=np.float64)
    for index, residue_name in enumerate(resname.tolist()):
        maximum = TIEN_MAXIMUM_ASA.get(residue_name)
        if maximum is not None and np.isfinite(sasa[index]):
            rsa[index] = sasa[index] / maximum

    return sasa.astype(np.float32), rsa.astype(np.float32)


__all__ = [
    "BIOTITE_PACKAGE",
    "BIOTITE_VERSION",
    "CANONICAL_REQUIRED_HEAVY_ATOMS",
    "SECONDARY_STRUCTURE_KEYS",
    "SECONDARY_STRUCTURE_STATE_MAPPING",
    "TIEN_MAXIMUM_ASA",
    "secondary_structure_onehot",
    "solvent_accessibility",
]
