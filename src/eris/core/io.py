from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import gemmi
import numpy as np

from eris.core.aminoacids import AA1_TO_INDEX, AA3_TO_AA1, UNKNOWN_INDEX
from eris.core.residue_identity import format_residue_id
from eris.core.types import StructureInput
from eris.exceptions import ErisError

logger = logging.getLogger(__name__)


class StructureLoadError(ErisError):
    pass


class LoadedStructureData(dict[str, np.ndarray]):
    """Extracted arrays with the parser-confirmed coordinate format as typed side data."""

    input_format: str
    chain_ids: tuple[str, ...]
    model_index: int

    def __init__(
        self,
        data: dict[str, np.ndarray],
        *,
        input_format: str,
        chain_ids: tuple[str, ...],
        model_index: int,
    ) -> None:
        super().__init__(data)
        self.input_format = input_format
        self.chain_ids = chain_ids
        self.model_index = model_index


_GEMMI_INPUT_FORMATS = {
    gemmi.CoorFormat.Pdb: "pdb",
    gemmi.CoorFormat.Mmcif: "mmcif",
    gemmi.CoorFormat.Mmjson: "mmjson",
    gemmi.CoorFormat.ChemComp: "chemcomp",
}


def _parsed_input_format(st: gemmi.Structure) -> str:
    input_format = getattr(st, "input_format", gemmi.CoorFormat.Unknown)
    try:
        return _GEMMI_INPUT_FORMATS[input_format]
    except KeyError as error:
        raise StructureLoadError(
            f"Gemmi did not report a supported parsed coordinate format: {input_format!r}"
        ) from error


# -----------------------
# Load / chain selection
# -----------------------
def load_structure(path: Path) -> gemmi.Structure:
    if not path.exists():
        raise StructureLoadError(f"File not found: {path}")
    try:
        st = gemmi.read_structure(str(path))
        if st is None or len(st) == 0:
            raise StructureLoadError(f"Empty/invalid structure: {path}")
        return st
    except Exception as e:
        raise StructureLoadError(f"Failed to parse structure: {path}") from e


def _is_relevant_protein_chain(chain: gemmi.Chain) -> bool:
    for residue in chain:
        residue_info = gemmi.find_tabulated_residue(residue.name)
        is_polymer = residue.entity_type == gemmi.EntityType.Polymer or residue.het_flag == "A"
        if is_polymer and residue_info.is_amino_acid():
            return True
    return False


def select_chains(
    st: gemmi.Structure,
    chain_id: tuple[str, ...] | None,
    model_index: int,
) -> tuple[gemmi.Chain, ...]:
    if len(st) == 0:
        raise StructureLoadError("Structure has no models.")
    if model_index < 0 or model_index >= len(st):
        raise StructureLoadError(f"Model index out of range: {model_index}")
    model = st[model_index]

    if chain_id is None:
        selected = tuple(chain for chain in model if _is_relevant_protein_chain(chain))
        if not selected:
            raise StructureLoadError("Model has no relevant protein/polymer chains.")
        return selected

    available = {chain.name for chain in model}
    missing = [name for name in chain_id if name not in available]
    if missing:
        raise StructureLoadError(f"Chain(s) not found: {missing}")
    requested = set(chain_id)
    return tuple(chain for chain in model if chain.name in requested)


def resolve_structure_input(inp: StructureInput) -> StructureInput:
    """Resolve a requested selection to an immutable tuple in model order."""
    structure = load_structure(Path(inp.source))
    chains = select_chains(structure, inp.chain_id, inp.model_index)
    return StructureInput(
        source=inp.source,
        chain_id=tuple(chain.name for chain in chains),
        model_index=inp.model_index,
    )


def _residue_id(chain: gemmi.Chain, res: gemmi.Residue) -> str:
    """
    Stable residue identifier for downstream mapping and reproducibility.

    Format: <chain>:<seqnum>:<icode>:<resname>
    Example: A:123::GLY   (empty insertion code becomes "")
    """
    return format_residue_id(
        chain_id=chain.name,
        residue_number=res.seqid.num,
        insertion_code=res.seqid.icode,
        residue_name=res.name,
    )


# -----------------------
# Altloc selection policy
# -----------------------
def _altloc_rank(alt: str) -> int:
    """
    Prefer blank altloc over 'A' over others (for deterministic tie-breaking).
    Lower rank is preferred.
    """
    a = _altloc_identifier(alt)
    if a == "":
        return 0
    if a == "A":
        return 1
    return 2


def _altloc_identifier(alt: str) -> str:
    return (alt or "").replace("\x00", "").strip().upper()


def _max_occupancy_sort_key(atom: gemmi.Atom) -> tuple[int, float, int, str]:
    occupancy = float(getattr(atom, "occ", float("nan")))
    identifier = _altloc_identifier(getattr(atom, "altloc", ""))
    finite_rank = 0 if np.isfinite(occupancy) else 1
    occupancy_rank = -occupancy if np.isfinite(occupancy) else 0.0
    return finite_rank, occupancy_rank, _altloc_rank(identifier), identifier


def _select_atom_altloc(atoms: Iterable[gemmi.Atom], altloc: str) -> Optional[gemmi.Atom]:
    """
    Select a single atom from an iterable of alternate locations.

    Parameters
    ----------
    atoms
        Iterable of gemmi.Atom objects, usually res["CA"] or atoms grouped by name.
    altloc
        Selection policy:
          - "first"          : pick the first atom as encountered (legacy-compatible)
          - "max_occupancy"  : pick atom with max occupancy; tie-break blank > A > others
          - "<LETTER>"       : pick that altloc letter if present, else blank if present, else first

    Returns
    -------
    gemmi.Atom or None
    """
    atoms_list = list(atoms)
    if not atoms_list:
        return None

    policy = (altloc or "first").strip()

    if policy.lower() == "first":
        return atoms_list[0]

    if policy.lower() == "max_occupancy":
        return min(atoms_list, key=_max_occupancy_sort_key)

    # explicit letter
    letter = policy.upper()
    preferred = [a for a in atoms_list if _altloc_identifier(getattr(a, "altloc", "")) == letter]
    if preferred:
        return preferred[0]

    blank = [a for a in atoms_list if _altloc_identifier(getattr(a, "altloc", "")) == ""]
    if blank:
        return blank[0]

    return atoms_list[0]


# -----------------------
# Residue-level bundle
# -----------------------

_ATOMIC_MASS = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "S": 32.06,
    "P": 30.974,
}


def _atomic_mass(atom: gemmi.Atom) -> float:
    sym = atom.element.name.upper() if atom.element is not None else ""
    return float(_ATOMIC_MASS.get(sym, 12.0))  # fallback


def _selected_atoms_from_residue(res: gemmi.Residue, altloc: str) -> List[gemmi.Atom]:
    """
    Select one atom per atom-name according to altloc policy.
    """
    selected: List[gemmi.Atom] = []

    atom_names = sorted({a.name for a in res})
    for atom_name in atom_names:
        if atom_name not in res:
            continue
        atom = _select_atom_altloc(res[atom_name], altloc)
        if atom is not None:
            selected.append(atom)

    return selected


def _selected_atom_coordinate(
    res: gemmi.Residue,
    atom_name: str,
    altloc: str,
) -> tuple[np.ndarray, bool]:
    atom = _select_atom_altloc(res[atom_name], altloc) if atom_name in res else None
    if atom is None:
        return np.full((3,), np.nan, dtype=np.float32), False
    coordinate = np.asarray(
        [float(atom.pos.x), float(atom.pos.y), float(atom.pos.z)],
        dtype=np.float32,
    )
    valid = bool(np.isfinite(coordinate).all())
    if not valid:
        coordinate[:] = np.nan
    return coordinate, valid


def _selected_heavy_atom_coordinates(res: gemmi.Residue, altloc: str) -> np.ndarray:
    coordinates: list[list[float]] = []
    for atom in _selected_atoms_from_residue(res, altloc):
        element = atom.element.name.upper() if atom.element is not None else ""
        if element in {"H", "D"}:
            continue
        coordinate = [float(atom.pos.x), float(atom.pos.y), float(atom.pos.z)]
        if np.isfinite(coordinate).all():
            coordinates.append(coordinate)
    return np.asarray(coordinates, dtype=np.float32).reshape((-1, 3))


def _compute_residue_center(res: gemmi.Residue, mode: str, altloc: str):
    """
    Returns:
        np.ndarray shape (3,) or None if center cannot be computed
        bfactor_ca: float
    """
    # CA-based signals stay anchored to CA for backward compatibility
    ca_atom = _select_atom_altloc(res["CA"], altloc) if "CA" in res else None

    if ca_atom is not None:
        bfactor_ca = float(ca_atom.b_iso)
    else:
        bfactor_ca = float("nan")

    if mode == "ca":
        if ca_atom is None:
            return None, bfactor_ca
        p = ca_atom.pos
        return (
            np.asarray([float(p.x), float(p.y), float(p.z)], dtype=np.float32),
            bfactor_ca,
        )

    atoms = _selected_atoms_from_residue(res, altloc)
    if not atoms:
        return None, bfactor_ca

    coords = np.asarray(
        [[float(a.pos.x), float(a.pos.y), float(a.pos.z)] for a in atoms], dtype=np.float32
    )

    if mode == "mean":
        center = coords.mean(axis=0).astype(np.float32)
        return center, bfactor_ca

    if mode == "com":
        masses = np.asarray([_atomic_mass(a) for a in atoms], dtype=np.float32)
        msum = float(masses.sum())
        if msum <= 0.0:
            return None, bfactor_ca
        center = (coords * masses[:, None]).sum(axis=0) / msum
        return center.astype(np.float32), bfactor_ca

    raise ValueError(f"Invalid residue_center: {mode!r}. Use 'ca', 'mean' or 'com'.")


def _next_biological_sequence_position(
    previous_residue_number: int | None,
    previous_position: int | None,
    residue_number: int,
) -> int:
    """Return a chain-relative position that preserves source residue semantics.

    Increasing residue numbers retain their numeric gap.  Residues with the same
    number (insertion-code variants) retain their source-order adjacency.
    """
    if previous_residue_number is None or previous_position is None:
        return 0
    if residue_number > previous_residue_number:
        return previous_position + residue_number - previous_residue_number
    return previous_position + 1


def extract_residue_bundle(
    chain: gemmi.Chain,
    amino_acids_only: bool = True,
    *,
    altloc: str = "max_occupancy",
    missing_residue: str = "skip",
    mode: str = "ca",
    plddt_source: str = "none",
) -> Dict[str, np.ndarray]:
    """
    Extract per-residue coordinates and basic residue-level signals.

    Parameters
    ----------
    chain
        Selected chain.
    amino_acids_only
        If True, keep only standard amino acids (3-letter in AA3_TO_AA1).
    altloc
        Altloc policy for atom selection.
    missing_residue
        acts as generic missing-center policy:
          - "skip": skip residue if center cannot be computed
          - "error": collect and raise
          - "nan": keep residue with pos=[nan,nan,nan]
    mode
        Residue center definition:
          - "ca": alpha carbon
          - "mean": mean of selected residue atom coordinates
          - "com": center of mass of selected residue atoms

    Returns
    -------
    dict with:
        pos : (N,3) float32
        residue_ids : (N,) str
        resname : (N,) str
        aa_index : (N,) int16
        plddt : (N,) float32
        bfactor_ca : (N,) float32
        backbone_n/backbone_ca/backbone_c : (N,3) float32
        backbone_n_mask/backbone_ca_mask/backbone_c_mask : (N,) bool
        polymer_heavy_atom_coordinates : (M,3) float32
        polymer_heavy_atom_residue_index : (M,) int64
        structural_residue_index : (N,) int64
    """
    missing_policy = (missing_residue or "skip").strip().lower()
    if missing_policy not in {"skip", "error", "nan"}:
        raise ValueError(
            f"Invalid missing_ca policy: {missing_residue!r}. Use 'skip', 'error', or 'nan'."
        )

    mode = (mode or "ca").strip().lower()
    if mode not in {"ca", "mean", "com"}:
        raise ValueError(f"Invalid residue_center: {mode!r}. Use 'ca', 'mean' or 'com'.")

    plddt_source = (plddt_source or "none").strip().lower()
    if plddt_source not in {"none", "bfactor"}:
        raise ValueError(f"Invalid plddt_source: {plddt_source!r}. Use 'none' or 'bfactor'.")

    coords: List[List[float]] = []
    ids: List[str] = []
    chain_ids: List[str] = []
    resnames: List[str] = []
    aa_idx: List[int] = []
    plddt_list: List[float] = []
    bfac_list: List[float] = []
    missing_ids: List[str] = []
    biological_sequence_positions: List[int] = []
    structural_residue_indices: List[int] = []
    backbone_n: List[List[float]] = []
    backbone_ca: List[List[float]] = []
    backbone_c: List[List[float]] = []
    backbone_n_mask: List[bool] = []
    backbone_ca_mask: List[bool] = []
    backbone_c_mask: List[bool] = []
    polymer_heavy_atom_coordinates: List[List[float]] = []
    polymer_heavy_atom_residue_index: List[int] = []
    previous_residue_number: int | None = None
    previous_sequence_position: int | None = None
    structural_residue_index = -1

    for res in chain:
        r3 = res.name.upper()

        if amino_acids_only and r3 not in AA3_TO_AA1:
            continue

        structural_residue_index += 1

        rid = _residue_id(chain, res)
        aa1 = AA3_TO_AA1.get(r3, "X")
        aa_index = AA1_TO_INDEX.get(aa1, UNKNOWN_INDEX)
        sequence_position = _next_biological_sequence_position(
            previous_residue_number,
            previous_sequence_position,
            int(res.seqid.num),
        )
        previous_residue_number = int(res.seqid.num)
        previous_sequence_position = sequence_position

        center, bfactor_ca = _compute_residue_center(res, mode=mode, altloc=altloc)
        n_coord, n_valid = _selected_atom_coordinate(res, "N", altloc)
        ca_coord, ca_valid = _selected_atom_coordinate(res, "CA", altloc)
        c_coord, c_valid = _selected_atom_coordinate(res, "C", altloc)
        heavy_atom_coordinates = _selected_heavy_atom_coordinates(res, altloc)
        if plddt_source == "none" or not np.isfinite(bfactor_ca):
            plddt_ca = float("nan")
        elif 0.0 <= bfactor_ca <= 100.0:
            plddt_ca = bfactor_ca
        else:
            raise StructureLoadError(
                f"CA B-factor interpreted as pLDDT for residue {rid} must be within "
                f"[0, 100], got {bfactor_ca}. Use plddt_source='none' for ordinary "
                "crystallographic B-factors."
            )

        if center is None:
            if missing_policy == "skip":
                missing_ids.append(rid)
                continue
            if missing_policy == "error":
                missing_ids.append(rid)
                continue

            # "nan"
            coords.append([float("nan"), float("nan"), float("nan")])
            ids.append(rid)
            chain_ids.append(chain.name)
            resnames.append(r3)
            aa_idx.append(aa_index)
            plddt_list.append(float("nan"))
            bfac_list.append(float("nan"))
            biological_sequence_positions.append(sequence_position)
            structural_residue_indices.append(structural_residue_index)
            backbone_n.append(n_coord.tolist())
            backbone_ca.append(ca_coord.tolist())
            backbone_c.append(c_coord.tolist())
            backbone_n_mask.append(n_valid)
            backbone_ca_mask.append(ca_valid)
            backbone_c_mask.append(c_valid)
            residue_index = len(ids) - 1
            polymer_heavy_atom_coordinates.extend(heavy_atom_coordinates.tolist())
            polymer_heavy_atom_residue_index.extend(
                [residue_index] * heavy_atom_coordinates.shape[0]
            )
            continue

        coords.append(center.tolist())
        ids.append(rid)
        chain_ids.append(chain.name)
        resnames.append(r3)
        aa_idx.append(aa_index)
        plddt_list.append(plddt_ca)
        bfac_list.append(bfactor_ca)
        biological_sequence_positions.append(sequence_position)
        structural_residue_indices.append(structural_residue_index)
        backbone_n.append(n_coord.tolist())
        backbone_ca.append(ca_coord.tolist())
        backbone_c.append(c_coord.tolist())
        backbone_n_mask.append(n_valid)
        backbone_ca_mask.append(ca_valid)
        backbone_c_mask.append(c_valid)
        residue_index = len(ids) - 1
        polymer_heavy_atom_coordinates.extend(heavy_atom_coordinates.tolist())
        polymer_heavy_atom_residue_index.extend([residue_index] * heavy_atom_coordinates.shape[0])

    if missing_policy == "error" and missing_ids:
        preview = ", ".join(missing_ids[:10])
        more = "" if len(missing_ids) <= 10 else f" (+{len(missing_ids) - 10} more)"
        raise StructureLoadError(f"Missing residue center in residues: {preview}{more}")

    return {
        "pos": np.asarray(coords, dtype=np.float32).reshape((-1, 3)),
        "residue_ids": np.asarray(ids, dtype=np.str_),
        "chain_id": np.asarray(chain_ids, dtype=np.str_),
        "resname": np.asarray(resnames, dtype=np.str_),
        "aa_index": np.asarray(aa_idx, dtype=np.int16),
        "plddt": np.asarray(plddt_list, dtype=np.float32),
        "bfactor_ca": np.asarray(bfac_list, dtype=np.float32),
        "omitted_residue_ids": np.asarray(missing_ids, dtype=np.str_),
        "biological_sequence_positions": np.asarray(biological_sequence_positions, dtype=np.int64),
        "structural_residue_index": np.asarray(structural_residue_indices, dtype=np.int64),
        "backbone_n": np.asarray(backbone_n, dtype=np.float32).reshape((-1, 3)),
        "backbone_ca": np.asarray(backbone_ca, dtype=np.float32).reshape((-1, 3)),
        "backbone_c": np.asarray(backbone_c, dtype=np.float32).reshape((-1, 3)),
        "backbone_n_mask": np.asarray(backbone_n_mask, dtype=bool),
        "backbone_ca_mask": np.asarray(backbone_ca_mask, dtype=bool),
        "backbone_c_mask": np.asarray(backbone_c_mask, dtype=bool),
        "polymer_heavy_atom_coordinates": np.asarray(
            polymer_heavy_atom_coordinates,
            dtype=np.float32,
        ).reshape((-1, 3)),
        "polymer_heavy_atom_residue_index": np.asarray(
            polymer_heavy_atom_residue_index,
            dtype=np.int64,
        ),
    }


def load_select_and_extract(
    inp: StructureInput,
    extractor: Callable[[object], dict[str, np.ndarray]],
) -> LoadedStructureData:
    try:
        st = load_structure(Path(inp.source))
        chains = select_chains(st, inp.chain_id, inp.model_index)
        bundles = [extractor(chain) for chain in chains]
        keys = bundles[0].keys()
        data = {key: np.concatenate([bundle[key] for bundle in bundles], axis=0) for key in keys}
        if "polymer_heavy_atom_residue_index" in keys:
            residue_offsets = np.cumsum([0, *(bundle["pos"].shape[0] for bundle in bundles[:-1])])
            data["polymer_heavy_atom_residue_index"] = np.concatenate(
                [
                    bundle["polymer_heavy_atom_residue_index"] + offset
                    for bundle, offset in zip(bundles, residue_offsets, strict=True)
                ],
                axis=0,
            ).astype(np.int64)
        if data["pos"].shape[0] == 0:
            raise StructureLoadError(
                "No positions found in selected chains (after filters/policies)."
            )
        return LoadedStructureData(
            data,
            input_format=_parsed_input_format(st),
            chain_ids=tuple(chain.name for chain in chains),
            model_index=inp.model_index,
        )
    except Exception:
        logger.exception(
            "Failed to load structure",
            extra={
                "source": str(inp.source),
                "chain_ids": inp.chain_id,
                "model_index": inp.model_index,
            },
        )
        raise


def load_and_extract_residue(
    inp: StructureInput,
    amino_acids_only: bool = True,
    *,
    mode: str = "ca",
    altloc: str = "max_occupancy",
    missing_ca: str = "skip",
    plddt_source: str = "none",
) -> LoadedStructureData:
    def _extract(ch):
        return extract_residue_bundle(
            ch,
            amino_acids_only=amino_acids_only,
            altloc=altloc,
            missing_residue=missing_ca,
            mode=mode,
            plddt_source=plddt_source,
        )

    return load_select_and_extract(inp, _extract)


# -----------------------
# Atom-level bundle
# -----------------------
# Minimal deterministic element vocabulary (extend later if needed)
_ELEM_VOCAB = [
    "UNK",
    "H",
    "C",
    "N",
    "O",
    "S",
    "P",
    "F",
    "CL",
    "BR",
    "I",
    "MG",
    "ZN",
    "CA",
    "FE",
    "NA",
    "K",
]
_ELEM_TO_INDEX = {e: i for i, e in enumerate(_ELEM_VOCAB)}


def _elem_symbol(atom: gemmi.Atom) -> str:
    try:
        s = (atom.element.name or "").strip().upper()
        if not s:
            return "UNK"
        return s.replace(" ", "")
    except Exception:
        return "UNK"


def _elem_index(atom: gemmi.Atom) -> int:
    return _ELEM_TO_INDEX.get(_elem_symbol(atom), 0)


def _atom_role(atom_name: str) -> int:
    """
    Very small role encoding (legacy-friendly).
      0 unknown
      1 backbone N
      2 CA
      3 backbone C
      4 backbone O
      5 sidechain/other
    """
    a = (atom_name or "").strip().upper()
    if a == "N":
        return 1
    if a == "CA":
        return 2
    if a == "C":
        return 3
    if a == "O":
        return 4
    if a:
        return 5
    return 0


def _is_water_residue(res: gemmi.Residue) -> bool:
    # Common water names: HOH, WAT, H2O
    r3 = (res.name or "").strip().upper()
    return r3 in {"HOH", "WAT", "H2O"}


def _is_ion_residue(res: gemmi.Residue) -> bool:
    # Very lightweight: many ions are 1-2 letter names, but not always.
    # We treat residue as ion if it is hetero and has a single atom and element != C/N/O/S etc.
    try:
        if not (getattr(res, "het_flag", "") or "").strip():
            return False
        if len(res) != 1:
            return False
        a = res[0]
        e = _elem_symbol(a)
        return e in {"NA", "K", "CA", "MG", "ZN", "FE", "CL", "BR", "I"}
    except Exception:
        return False


def extract_atom_bundle(
    chain: gemmi.Chain,
    *,
    include_hetero: bool = False,
    include_h: bool = False,
    drop_water: bool = True,
    keep_ions: bool = True,
    altloc: str = "first",  # "first" | "max_occupancy" | "A" etc.
    amino_acids_only: bool = False,  # if True, only residues in AA3_TO_AA1
) -> Dict[str, np.ndarray]:
    """
    Extract per-atom arrays from a chain.

    Altloc policy:
      - "first": keep atoms as encountered (legacy behavior; may include multiple altlocs)
      - "max_occupancy" or "<LETTER>": select one atom per atom-name within each residue

    Returns dict with:
      - pos: (N,3) float32
      - elem_index: (N,) int16
      - atom_role: (N,) int8
      - bfactor: (N,) float32
      - occ: (N,) float32
      - residue_ids: (N,) str
      - resname: (N,) str
      - atom_name: (N,) str
      - seq_pos_residue: (N,) int32 (residue index in chain order; per atom)
      - aa_index_residue: (N,) int16 (residue AA index; unknown for non-AA)
    """
    coords: List[List[float]] = []
    elem_idx: List[int] = []
    roles: List[int] = []
    bfac: List[float] = []
    occ: List[float] = []
    residue_ids: List[str] = []
    chain_ids: List[str] = []
    resname: List[str] = []
    atom_name: List[str] = []
    seq_pos_res: List[int] = []
    aa_idx_res: List[int] = []

    policy = (altloc or "first").strip()

    res_counter = -1
    for res in chain:
        r3 = (res.name or "").strip().upper()

        # optional filter
        if amino_acids_only and r3 not in AA3_TO_AA1:
            continue

        # hetero handling: gemmi uses res.het_flag; for polymer residues it is usually ""
        het_flag = (getattr(res, "het_flag", "") or "").strip().upper()
        is_hetero = bool(het_flag) and (het_flag in {"H", "W"})

        # water/ions handling (only matters if residue is hetero-ish)
        if drop_water and _is_water_residue(res):
            continue
        if (not keep_ions) and _is_ion_residue(res):
            continue

        if is_hetero and (not include_hetero):
            continue

        res_counter += 1
        rid = _residue_id(chain, res)

        aa1 = AA3_TO_AA1.get(r3, "X")
        aa_idx = AA1_TO_INDEX.get(aa1, UNKNOWN_INDEX)

        # Collect atoms based on altloc policy
        if policy.lower() == "first":
            atoms_iter = list(res)
        else:
            # group by atom name; select one per name
            by_name: Dict[str, List[gemmi.Atom]] = {}
            for a in res:
                name = (a.name or "").strip().upper()
                by_name.setdefault(name, []).append(a)

            chosen: List[gemmi.Atom] = []
            for atoms in by_name.values():
                chosen_atom = _select_atom_altloc(atoms, policy)
                if chosen_atom is not None:
                    chosen.append(chosen_atom)
            atoms_iter = chosen

        for atom in atoms_iter:
            if (not include_h) and _elem_symbol(atom) == "H":
                continue

            p = atom.pos
            coords.append([float(p.x), float(p.y), float(p.z)])

            elem_idx.append(_elem_index(atom))
            roles.append(_atom_role(atom.name))
            bfac.append(float(atom.b_iso))
            occ.append(float(getattr(atom, "occ", 0.0)))

            residue_ids.append(rid)
            chain_ids.append(chain.name)
            resname.append(r3)
            atom_name.append((atom.name or "").strip())

            seq_pos_res.append(res_counter)
            aa_idx_res.append(aa_idx)

    return {
        "pos": np.asarray(coords, dtype=np.float32).reshape((-1, 3)),
        "elem_index": np.asarray(elem_idx, dtype=np.int16),
        "atom_role": np.asarray(roles, dtype=np.int8),
        "bfactor": np.asarray(bfac, dtype=np.float32),
        "occ": np.asarray(occ, dtype=np.float32),
        "residue_ids": np.asarray(residue_ids, dtype=np.str_),
        "chain_id": np.asarray(chain_ids, dtype=np.str_),
        "resname": np.asarray(resname, dtype=np.str_),
        "atom_name": np.asarray(atom_name, dtype=np.str_),
        "seq_pos_residue": np.asarray(seq_pos_res, dtype=np.int32),
        "aa_index_residue": np.asarray(aa_idx_res, dtype=np.int16),
    }


def load_and_extract_atoms(
    inp: StructureInput,
    *,
    include_hetero: bool = False,
    include_h: bool = False,
    drop_water: bool = True,
    keep_ions: bool = True,
    altloc: str = "first",
    amino_acids_only: bool = False,
) -> LoadedStructureData:
    """
    Atom-level extraction wrapper using StructureInput.
    """

    def _extract(ch):
        return extract_atom_bundle(
            ch,
            include_hetero=include_hetero,
            include_h=include_h,
            drop_water=drop_water,
            keep_ions=keep_ions,
            altloc=altloc,
            amino_acids_only=amino_acids_only,
        )

    return load_select_and_extract(inp, _extract)
