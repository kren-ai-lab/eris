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

    def __init__(self, data: dict[str, np.ndarray], *, input_format: str) -> None:
        super().__init__(data)
        self.input_format = input_format


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


def select_chain(st: gemmi.Structure, chain_id: Optional[str], model_index: int) -> gemmi.Chain:
    if len(st) == 0:
        raise StructureLoadError("Structure has no models.")
    if model_index < 0 or model_index >= len(st):
        raise StructureLoadError(f"Model index out of range: {model_index}")
    model = st[model_index]

    if chain_id is None:
        if len(model) == 0:
            raise StructureLoadError("Model has no chains.")
        return model[0]

    for ch in model:
        if ch.name == chain_id:
            return ch
    raise StructureLoadError(f"Chain not found: {chain_id}")


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
    a = (alt or "").strip().upper()
    if a == "":
        return 0
    if a == "A":
        return 1
    return 2


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
        return max(
            atoms_list,
            key=lambda a: (float(getattr(a, "occ", 0.0)), -_altloc_rank(getattr(a, "altloc", ""))),
        )

    # explicit letter
    letter = policy.upper()
    preferred = [
        a for a in atoms_list if (getattr(a, "altloc", "") or "").strip().upper() == letter
    ]
    if preferred:
        return preferred[0]

    blank = [a for a in atoms_list if (getattr(a, "altloc", "") or "").strip() == ""]
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


def _compute_residue_center(res: gemmi.Residue, mode: str, altloc: str):
    """
    Returns:
        np.ndarray shape (3,) or None if center cannot be computed
        bfactor_ca: float
        plddt_ca: float
    """
    # CA-based signals stay anchored to CA for backward compatibility
    ca_atom = _select_atom_altloc(res["CA"], altloc) if "CA" in res else None

    if ca_atom is not None:
        b = float(ca_atom.b_iso)
        bfactor_ca = b
        plddt_ca = b if 0.0 <= b <= 100.0 else float("nan")
    else:
        bfactor_ca = float("nan")
        plddt_ca = float("nan")

    if mode == "ca":
        if ca_atom is None:
            return None, bfactor_ca, plddt_ca
        p = ca_atom.pos
        return (
            np.asarray([float(p.x), float(p.y), float(p.z)], dtype=np.float32),
            bfactor_ca,
            plddt_ca,
        )

    atoms = _selected_atoms_from_residue(res, altloc)
    if not atoms:
        return None, bfactor_ca, plddt_ca

    coords = np.asarray(
        [[float(a.pos.x), float(a.pos.y), float(a.pos.z)] for a in atoms], dtype=np.float32
    )

    if mode == "mean":
        center = coords.mean(axis=0).astype(np.float32)
        return center, bfactor_ca, plddt_ca

    if mode == "com":
        masses = np.asarray([_atomic_mass(a) for a in atoms], dtype=np.float32)
        msum = float(masses.sum())
        if msum <= 0.0:
            return None, bfactor_ca, plddt_ca
        center = (coords * masses[:, None]).sum(axis=0) / msum
        return center.astype(np.float32), bfactor_ca, plddt_ca

    raise ValueError(f"Invalid residue_center: {mode!r}. Use 'ca', 'mean' or 'com'.")


def extract_residue_bundle(
    chain: gemmi.Chain,
    amino_acids_only: bool = True,
    *,
    altloc: str = "first",
    missing_residue: str = "skip",
    mode: str = "ca",
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
    """
    missing_policy = (missing_residue or "skip").strip().lower()
    if missing_policy not in {"skip", "error", "nan"}:
        raise ValueError(
            f"Invalid missing_ca policy: {missing_residue!r}. Use 'skip', 'error', or 'nan'."
        )

    mode = (mode or "ca").strip().lower()
    if mode not in {"ca", "mean", "com"}:
        raise ValueError(f"Invalid residue_center: {mode!r}. Use 'ca', 'mean' or 'com'.")

    coords: List[List[float]] = []
    ids: List[str] = []
    resnames: List[str] = []
    aa_idx: List[int] = []
    plddt_list: List[float] = []
    bfac_list: List[float] = []
    missing_ids: List[str] = []

    for res in chain:
        r3 = res.name.upper()

        if amino_acids_only and r3 not in AA3_TO_AA1:
            continue

        rid = _residue_id(chain, res)
        aa1 = AA3_TO_AA1.get(r3, "X")
        aa_index = AA1_TO_INDEX.get(aa1, UNKNOWN_INDEX)

        center, bfactor_ca, plddt_ca = _compute_residue_center(res, mode=mode, altloc=altloc)

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
            resnames.append(r3)
            aa_idx.append(aa_index)
            plddt_list.append(float("nan"))
            bfac_list.append(float("nan"))
            continue

        coords.append(center.tolist())
        ids.append(rid)
        resnames.append(r3)
        aa_idx.append(aa_index)
        plddt_list.append(plddt_ca)
        bfac_list.append(bfactor_ca)

    if missing_policy == "error" and missing_ids:
        preview = ", ".join(missing_ids[:10])
        more = "" if len(missing_ids) <= 10 else f" (+{len(missing_ids) - 10} more)"
        raise StructureLoadError(f"Missing residue center in residues: {preview}{more}")

    if not coords:
        raise StructureLoadError(
            "No residue positions found in selected chain (after filters/policies)."
        )

    return {
        "pos": np.asarray(coords, dtype=np.float32),
        "residue_ids": np.asarray(ids, dtype=np.str_),
        "resname": np.asarray(resnames, dtype=np.str_),
        "aa_index": np.asarray(aa_idx, dtype=np.int16),
        "plddt": np.asarray(plddt_list, dtype=np.float32),
        "bfactor_ca": np.asarray(bfac_list, dtype=np.float32),
        "omitted_residue_ids": np.asarray(missing_ids, dtype=np.str_),
    }


def load_select_and_extract(
    inp: StructureInput,
    extractor: Callable[[object], dict[str, np.ndarray]],
) -> LoadedStructureData:
    try:
        st = load_structure(Path(inp.source))
        ch = select_chain(st, inp.chain_id, inp.model_index)
        data = extractor(ch)
        return LoadedStructureData(data, input_format=_parsed_input_format(st))
    except Exception:
        logger.exception(
            "Failed to load structure",
            extra={
                "source": str(inp.source),
                "chain_id": inp.chain_id,
                "model_index": inp.model_index,
            },
        )
        raise


def load_and_extract_residue(
    inp: StructureInput,
    amino_acids_only: bool = True,
    *,
    mode: str = "ca",
    altloc: str = "first",
    missing_ca: str = "skip",
) -> LoadedStructureData:
    def _extract(ch):
        return extract_residue_bundle(
            ch,
            amino_acids_only=amino_acids_only,
            altloc=altloc,
            missing_residue=missing_ca,
            mode=mode,
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
            resname.append(r3)
            atom_name.append((atom.name or "").strip())

            seq_pos_res.append(res_counter)
            aa_idx_res.append(aa_idx)

    if not coords:
        raise StructureLoadError("No atoms found in selected chain (after filters/policies).")

    return {
        "pos": np.asarray(coords, dtype=np.float32),
        "elem_index": np.asarray(elem_idx, dtype=np.int16),
        "atom_role": np.asarray(roles, dtype=np.int8),
        "bfactor": np.asarray(bfac, dtype=np.float32),
        "occ": np.asarray(occ, dtype=np.float32),
        "residue_ids": np.asarray(residue_ids, dtype=np.str_),
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
