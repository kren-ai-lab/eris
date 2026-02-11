from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gemmi
import numpy as np

from morpheo.exceptions import MorpheoError
from morpheo.core.types import StructureInput
from morpheo.core.aminoacids import AA3_TO_AA1, AA1_TO_INDEX, UNKNOWN_INDEX


class StructureLoadError(MorpheoError):
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
    icode = (res.seqid.icode or "").strip()
    return f"{chain.name}:{res.seqid.num}:{icode}:{res.name}"


# -----------------------
# Residue-level (CA) bundle
# -----------------------
def extract_ca_bundle(
    chain: gemmi.Chain,
    amino_acids_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract per-residue CA coordinates and basic residue-level signals.

    Returns
    -------
    pos : (N,3) float32
    residue_ids : (N,) str
    resname : (N,) str (3-letter)
    aa_index : (N,) int16 in [0..20], 20 = unknown
    plddt : (N,) float32 (NaN if not a [0,100] signal)
    bfactor_ca : (N,) float32 (raw CA B-factor)
    """
    coords: List[List[float]] = []
    ids: List[str] = []
    resnames: List[str] = []
    aa_idx: List[int] = []
    plddt_list: List[float] = []
    bfac_list: List[float] = []

    for res in chain:
        r3 = res.name.upper()

        if amino_acids_only and r3 not in AA3_TO_AA1:
            continue

        if "CA" not in res:
            continue

        atom = res["CA"][0]
        p = atom.pos
        coords.append([p.x, p.y, p.z])

        rid = _residue_id(chain, res)
        ids.append(rid)

        resnames.append(r3)

        aa1 = AA3_TO_AA1.get(r3, "X")
        aa_idx.append(AA1_TO_INDEX.get(aa1, UNKNOWN_INDEX))

        b = float(atom.b_iso)
        bfac_list.append(b)

        # pLDDT proxy heuristic: AF2/ESMFold typically uses 0..100 in B-factor field
        plddt_list.append(b if 0.0 <= b <= 100.0 else float("nan"))

    if not coords:
        raise StructureLoadError("No CA residues found in selected chain (after filters).")

    pos = np.asarray(coords, dtype=np.float32)
    residue_ids = np.asarray(ids, dtype=np.str_)
    resname = np.asarray(resnames, dtype=np.str_)
    aa_index = np.asarray(aa_idx, dtype=np.int16)
    plddt = np.asarray(plddt_list, dtype=np.float32)
    bfactor_ca = np.asarray(bfac_list, dtype=np.float32)

    return pos, residue_ids, resname, aa_index, plddt, bfactor_ca


def load_and_extract(
    inp: StructureInput,
    amino_acids_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    st = load_structure(Path(inp.source))
    ch = select_chain(st, inp.chain_id, inp.model_index)
    return extract_ca_bundle(ch, amino_acids_only=amino_acids_only)


# -----------------------
# Atom-level bundle (NEW)
# -----------------------
# Minimal deterministic element vocabulary (extend later if needed)
_ELEM_VOCAB = [
    "UNK", "H", "C", "N", "O", "S", "P", "F",
    "CL", "BR", "I", "MG", "ZN", "CA", "FE", "NA", "K"
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
    Very small role encoding:
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


def extract_atom_bundle(
    chain: gemmi.Chain,
    *,
    include_hetero: bool = False,
    include_h: bool = False,
    altloc: str = "first",  # "first" or "A" etc.
    amino_acids_only: bool = False,  # if True, only residues in AA3_TO_AA1
) -> Dict[str, np.ndarray]:
    """
    Extract per-atom arrays from a chain.

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

    def _alt_ok(a: gemmi.Atom) -> bool:
        if altloc == "first":
            return True
        # accept blank/none or requested altloc letter
        al = (a.altloc or "").strip()
        return (al == "") or (al.upper() == altloc.upper())

    res_counter = -1
    for res in chain:
        r3 = res.name.upper()

        # optional filter
        if amino_acids_only and r3 not in AA3_TO_AA1:
            continue

        # hetero handling: gemmi uses res.het_flag; for polymer residues it is usually ""
        het_flag = (getattr(res, "het_flag", "") or "").strip().upper()

        # gemmi convention: 'A' (polymer/standard) is NOT hetero
        # treat only explicit hetero/water as hetero for filtering
        is_hetero = het_flag in {"H", "W"}

        if is_hetero and (not include_hetero):
            continue


        res_counter += 1
        rid = _residue_id(chain, res)

        aa1 = AA3_TO_AA1.get(r3, "X")
        aa_idx = AA1_TO_INDEX.get(aa1, UNKNOWN_INDEX)

        for atom in res:
            if not _alt_ok(atom):
                continue
            if (not include_h) and _elem_symbol(atom) == "H":
                continue

            p = atom.pos
            coords.append([float(p.x), float(p.y), float(p.z)])

            elem_idx.append(_elem_index(atom))
            roles.append(_atom_role(atom.name))
            bfac.append(float(atom.b_iso))
            occ.append(float(atom.occ))

            residue_ids.append(rid)
            resname.append(r3)
            atom_name.append((atom.name or "").strip())

            seq_pos_res.append(res_counter)
            aa_idx_res.append(aa_idx)

    if not coords:
        raise StructureLoadError("No atoms found in selected chain (after filters).")

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
    altloc: str = "first",
    amino_acids_only: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Atom-level extraction wrapper using StructureInput.
    """
    st = load_structure(Path(inp.source))
    ch = select_chain(st, inp.chain_id, inp.model_index)
    return extract_atom_bundle(
        ch,
        include_hetero=include_hetero,
        include_h=include_h,
        altloc=altloc,
        amino_acids_only=amino_acids_only,
    )
