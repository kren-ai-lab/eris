from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import gemmi
import numpy as np

from morpheo.exceptions import MorpheoError
from morpheo.core.types import StructureInput


class StructureLoadError(MorpheoError):
    pass


def load_structure(path: Path) -> gemmi.Structure:
    if not path.exists():
        raise StructureLoadError(f"File not found: {path}")
    try:
        st = gemmi.read_structure(str(path))
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


def extract_ca_coords(chain: gemmi.Chain) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    pos : (N,3) float32
    residue_ids : (N,) unicode strings "CHAIN:RESNUM:ICODE:RESNAME"
    """
    coords: List[List[float]] = []
    ids: List[str] = []

    for res in chain:
        if "CA" not in res:
            continue
        atom = res["CA"][0]
        p = atom.pos
        coords.append([p.x, p.y, p.z])
        icode = (res.seqid.icode or "").strip()
        ids.append(f"{chain.name}:{res.seqid.num}:{icode}:{res.name}")

    if not coords:
        raise StructureLoadError("No CA atoms found in selected chain.")

    pos = np.asarray(coords, dtype=np.float32)
    residue_ids = np.asarray(ids, dtype=np.str_)
    return pos, residue_ids


def load_and_extract(inp: StructureInput) -> Tuple[np.ndarray, np.ndarray]:
    st = load_structure(Path(inp.source))
    ch = select_chain(st, inp.chain_id, inp.model_index)
    return extract_ca_coords(ch)
