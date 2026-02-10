from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import gemmi
import numpy as np

from morpheo.exceptions import MorpheoError
from morpheo.core.types import StructureInput
from morpheo.core.aminoacids import AA3_TO_AA1, AA1_TO_INDEX, UNKNOWN_INDEX


class StructureLoadError(MorpheoError):
    pass


def load_structure(path: Path) -> gemmi.Structure:
    if not path.exists():
        raise StructureLoadError(f"File not found: {path}")
    try:
        return gemmi.read_structure(str(path))
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

        r3 = res.name.upper()
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
