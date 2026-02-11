from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

# 20 standard amino acids, UNK at the end
AA_ORDER: List[str] = [
    "A", "C", "D", "E", "F",
    "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R",
    "S", "T", "V", "W", "Y",
    "X",  # UNK
]

AA_TO_INDEX: Dict[str, int] = {aa: i for i, aa in enumerate(AA_ORDER)}
INDEX_TO_AA: Dict[int, str] = {i: aa for aa, i in AA_TO_INDEX.items()}

UNK_INDEX = AA_TO_INDEX["X"]
N_AA = len(AA_ORDER)

# Kyte-Doolittle hydrophobicity (approx, standard values)
_KD = {
    "A": 1.8,  "C": 2.5,  "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5,  "K": -3.9, "L": 3.8,
    "M": 1.9,  "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2,  "W": -0.9, "Y": -1.3,
}

# Approx side-chain volumes (A^3) (typical tabulated values)
_VOL = {
    "A": 88.6,  "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9,
    "G": 60.1,  "H": 153.2, "I": 166.7, "K": 168.6, "L": 166.7,
    "M": 162.9, "N": 114.1, "P": 112.7, "Q": 143.8, "R": 173.4,
    "S": 89.0,  "T": 116.1, "V": 140.0, "W": 227.8, "Y": 193.6,
}

# Simple categorical descriptors
_CHARGE = {  # at physiological pH (approx)
    "D": -1.0, "E": -1.0,
    "K": +1.0, "R": +1.0, "H": +0.1,
}
_AROMATIC = {"F", "W", "Y", "H"}
_POLAR = {"S", "T", "N", "Q", "C", "Y", "H", "D", "E", "K", "R"}

# Build a deterministic physchem table: shape (21, P)
# Columns:
# 0 hydrophobicity (KD) normalized
# 1 volume normalized
# 2 charge (approx)
# 3 aromatic (0/1)
# 4 polar (0/1)
# 5 gly/pro special (0/1)
# 6 proline (0/1)
P = 7

def _normalize(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mn = float(arr.min())
    mx = float(arr.max())
    return (arr - mn) / (mx - mn + eps)

def physchem_table() -> np.ndarray:
    kd = np.zeros((N_AA,), dtype=np.float32)
    vol = np.zeros((N_AA,), dtype=np.float32)
    charge = np.zeros((N_AA,), dtype=np.float32)
    aromatic = np.zeros((N_AA,), dtype=np.float32)
    polar = np.zeros((N_AA,), dtype=np.float32)
    glypro = np.zeros((N_AA,), dtype=np.float32)
    pro = np.zeros((N_AA,), dtype=np.float32)

    for aa, i in AA_TO_INDEX.items():
        if aa == "X":
            continue
        kd[i] = float(_KD.get(aa, 0.0))
        vol[i] = float(_VOL.get(aa, 0.0))
        charge[i] = float(_CHARGE.get(aa, 0.0))
        aromatic[i] = 1.0 if aa in _AROMATIC else 0.0
        polar[i] = 1.0 if aa in _POLAR else 0.0
        glypro[i] = 1.0 if aa in {"G", "P"} else 0.0
        pro[i] = 1.0 if aa == "P" else 0.0

    kd_n = _normalize(kd)
    vol_n = _normalize(vol)

    table = np.stack([kd_n, vol_n, charge, aromatic, polar, glypro, pro], axis=1).astype(np.float32)
    # UNK row stays zeros
    return table


_PHYS = physchem_table()

def get_physchem(aa_index: np.ndarray) -> np.ndarray:
    """aa_index: (N,) int -> physchem: (N,P) float32"""
    aa = np.asarray(aa_index, dtype=np.int64).reshape(-1)
    aa = np.clip(aa, 0, N_AA - 1)
    return _PHYS[aa]
