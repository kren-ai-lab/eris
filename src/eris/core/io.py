from __future__ import annotations
import logging
logger = logging.getLogger(__name__)

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Callable, TypeVar

import gemmi
import numpy as np

from eris.exceptions import ErisError
from eris.core.types import StructureInput
from eris.core.aminoacids import AA3_TO_AA1, AA1_TO_INDEX, UNKNOWN_INDEX

T = TypeVar("T")


class StructureLoadError(ErisError):
    pass


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
    icode = (res.seqid.icode or "").strip()
    return f"{chain.name}:{res.seqid.num}:{icode}:{res.name}"


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
    preferred = [a for a in atoms_list if (getattr(a, "altloc", "") or "").strip().upper() == letter]
    if preferred:
        return preferred[0]

    blank = [a for a in atoms_list if (getattr(a, "altloc", "") or "").strip() == ""]
    if blank:
        return blank[0]

    return atoms_list[0]


# -----------------------
# Residue-level (CA) bundle
# -----------------------
def extract_ca_bundle(
    chain: gemmi.Chain,
    amino_acids_only: bool = True,
    *,
    altloc: str = "first",
    missing_ca: str = "skip",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract per-residue CA coordinates and basic residue-level signals.

    Parameters
    ----------
    chain
        Selected chain.
    amino_acids_only
        If True, keep only standard amino acids (3-letter in AA3_TO_AA1).
    altloc
        Altloc policy for CA selection:
          - "first" (default): first encountered (legacy-compatible)
          - "max_occupancy": choose CA altloc with highest occupancy
          - "<LETTER>": choose that altloc if present, else blank, else first
    missing_ca
        How to handle residues missing CA:
          - "skip" (default): skip such residues
          - "error": raise with a compact report
          - "nan": keep residue with pos = [nan, nan, nan] (and bfactor/plddt as nan)

    Returns dict with:
    -------
    pos : (N,3) float32
    residue_ids : (N,) str
    resname : (N,) str (3-letter)
    aa_index : (N,) int16 in [0..20], 20 = unknown
    plddt : (N,) float32 (NaN if not a [0,100] signal)
    bfactor_ca : (N,) float32 (raw CA B-factor; NaN if missing_ca="nan")
    """
    missing_ca_policy = (missing_ca or "skip").strip().lower()
    if missing_ca_policy not in {"skip", "error", "nan"}:
        raise ValueError(f"Invalid missing_ca policy: {missing_ca!r}. Use 'skip', 'error', or 'nan'.")

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

        if "CA" not in res:
            if missing_ca_policy == "skip":
                continue
            if missing_ca_policy == "error":
                missing_ids.append(rid)
                continue
            # "nan"
            coords.append([float("nan"), float("nan"), float("nan")])
            ids.append(rid)
            resnames.append(r3)
            aa1 = AA3_TO_AA1.get(r3, "X")
            aa_idx.append(AA1_TO_INDEX.get(aa1, UNKNOWN_INDEX))
            plddt_list.append(float("nan"))
            bfac_list.append(float("nan"))
            continue

        atom = _select_atom_altloc(res["CA"], altloc)
        if atom is None:
            # Extremely rare, but keep policy consistent
            if missing_ca_policy == "skip":
                continue
            if missing_ca_policy == "error":
                missing_ids.append(rid)
                continue
            coords.append([float("nan"), float("nan"), float("nan")])
            ids.append(rid)
            resnames.append(r3)
            aa1 = AA3_TO_AA1.get(r3, "X")
            aa_idx.append(AA1_TO_INDEX.get(aa1, UNKNOWN_INDEX))
            plddt_list.append(float("nan"))
            bfac_list.append(float("nan"))
            continue

        p = atom.pos
        coords.append([float(p.x), float(p.y), float(p.z)])
        ids.append(rid)
        resnames.append(r3)

        aa1 = AA3_TO_AA1.get(r3, "X")
        aa_idx.append(AA1_TO_INDEX.get(aa1, UNKNOWN_INDEX))

        b = float(atom.b_iso)
        bfac_list.append(b)

        # pLDDT proxy heuristic: AF2/ESMFold typically uses 0..100 in B-factor field
        plddt_list.append(b if 0.0 <= b <= 100.0 else float("nan"))

    if missing_ca_policy == "error" and missing_ids:
        preview = ", ".join(missing_ids[:10])
        more = "" if len(missing_ids) <= 10 else f" (+{len(missing_ids) - 10} more)"
        raise StructureLoadError(f"Missing CA in residues: {preview}{more}")

    if not coords:
        raise StructureLoadError("No CA residues found in selected chain (after filters/policies).")

    # pos = np.asarray(coords, dtype=np.float32)
    # residue_ids = np.asarray(ids, dtype=np.str_)
    # resname = np.asarray(resnames, dtype=np.str_)
    # aa_index = np.asarray(aa_idx, dtype=np.int16)
    # plddt = np.asarray(plddt_list, dtype=np.float32)
    # bfactor_ca = np.asarray(bfac_list, dtype=np.float32)

    # return pos, residue_ids, resname, aa_index, plddt, bfactor_ca

    return  {
        "pos": np.asarray(coords, dtype=np.float32),
        "residue_ids": np.asarray(ids, dtype=np.str_),
        "resname": np.asarray(resnames, dtype=np.str_),
        "aa_index": np.asarray(aa_idx, dtype=np.int16),
        "plddt": np.asarray(plddt_list, dtype=np.float32),
        "bfactor_ca": np.asarray(bfac_list, dtype=np.float32),
    }
    


def load_select_and_extract(inp, extractor: Callable[[object], T]) -> T:
    try:
        st = load_structure(Path(inp.source))
        ch = select_chain(st, inp.chain_id, inp.model_index)
        return extractor(ch)
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


def load_and_extract(
    inp: StructureInput,
    amino_acids_only: bool = True,
    *,
    altloc: str = "first",
    missing_ca: str = "skip",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def _extract(ch):
        return extract_ca_bundle(
            ch,
            amino_acids_only=amino_acids_only,
            altloc=altloc,
            missing_ca=missing_ca,
        )
    return load_select_and_extract(inp, _extract)



# -----------------------
# Atom-level bundle
# -----------------------
# Minimal deterministic element vocabulary (extend later if needed)
_ELEM_VOCAB = [
    "UNK", "H", "C", "N", "O", "S", "P", "F",
    "CL", "BR", "I", "MG", "ZN", "CA", "FE", "NA", "K",
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
) -> Dict[str, np.ndarray]:
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
