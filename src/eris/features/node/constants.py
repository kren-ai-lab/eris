from __future__ import annotations

from typing import Dict, List
import numpy as np

# Not importing from roxy "pKa values and residue classes"
from roxy.core.constants import KD, EISENBERG, HYDROPHILICITY,\
    SIDECHAIN_MASS, VOLUME, AA_MOLECULAR_WEIGHT, BOMAN, POLARITY,\
    FLEXIBILITY, CF_HELIX, CF_SHEET, CF_TURN, TOP_IDP, DONORS, ACCEPTORS


# Canonical AA vocab for Eris (matches typical AA_index conventions)
# 20 standard + X for unknown
AA_VOCAB: List[str] = list("ACDEFGHIKLMNPQRSTVWY") + ["X"]
AA_TO_INDEX: Dict[str, int] = {aa: i for i, aa in enumerate(AA_VOCAB)}
UNKNOWN_AA = "X"
UNKNOWN_INDEX = AA_TO_INDEX[UNKNOWN_AA]


# Auxiliary func. Looks in roxy dict by index.
def _build_lookup_from_roxy(roxy_dict: Dict[str, float]) -> np.ndarray:
    lookup = np.zeros(len(AA_VOCAB), dtype=np.float32)

    for aa in AA_VOCAB:
        if aa in roxy_dict:
            lookup[AA_TO_INDEX[aa]] = roxy_dict[aa]
        else:
            lookup[AA_TO_INDEX[aa]] = 0.0   # Default for X or other AA

    return lookup


# -------------------------
# Roxy Residue Features
# -------------------------

_KD_LOOKUP = _build_lookup_from_roxy(KD)
_EISENBERG_LOOKUP = _build_lookup_from_roxy(EISENBERG)
_HYDROPHILICITY_LOOKUP = _build_lookup_from_roxy(HYDROPHILICITY)
_SIDECHAIN_MASS_LOOKUP = _build_lookup_from_roxy(SIDECHAIN_MASS)
_VOLUME_LOOKUP = _build_lookup_from_roxy(VOLUME)
_AA_MOLECULAR_WEIGHT_LOOKUP = _build_lookup_from_roxy(AA_MOLECULAR_WEIGHT)
_BOMAN_LOOKUP = _build_lookup_from_roxy(BOMAN)
_POLARITY_LOOKUP = _build_lookup_from_roxy(POLARITY)
_FLEXIBILITY_LOOKUP = _build_lookup_from_roxy(FLEXIBILITY)
_CF_HELIX_LOOKUP = _build_lookup_from_roxy(CF_HELIX)
_CF_SHEET_LOOKUP = _build_lookup_from_roxy(CF_SHEET)
_CF_TURN_LOOKUP = _build_lookup_from_roxy(CF_TURN)
_TOP_IDP_LOOKUP = _build_lookup_from_roxy(TOP_IDP)
_DONORS_LOOKUP = _build_lookup_from_roxy(DONORS)
_ACCEPTORS_LOOKUP = _build_lookup_from_roxy(ACCEPTORS)

# DO NOT change ordering/keys once released.
PHYSCHEM_LOOKUPS = [
    ("hydrophobicity_kd", _KD_LOOKUP),
    ("hydrophobicity_eisenberg", _EISENBERG_LOOKUP),
    ("hydrophilicity", _HYDROPHILICITY_LOOKUP),
    ("sidechain_mass", _SIDECHAIN_MASS_LOOKUP),
    ("volume", _VOLUME_LOOKUP),
    ("molecular_weight", _AA_MOLECULAR_WEIGHT_LOOKUP),
    ("boman", _BOMAN_LOOKUP),
    ("polarity", _POLARITY_LOOKUP),
    ("flexibility", _FLEXIBILITY_LOOKUP),
    ("cf_helix", _CF_HELIX_LOOKUP),
    ("cf_sheet", _CF_SHEET_LOOKUP),
    ("cf_turn", _CF_TURN_LOOKUP),
    ("top_idp", _TOP_IDP_LOOKUP),
    ("hbond_donor", _DONORS_LOOKUP),
    ("hbond_acceptor", _ACCEPTORS_LOOKUP),
]

PHYSchem_KEYS = [f"physchem.{name}" for name, _ in PHYSCHEM_LOOKUPS]



AAINDEX_BASIC_SUBIDX = [
    0,   # hydrophobicity_kd
    4,   # volume
    7,   # polarity
    8,   # flexibility
    6,   # boman
    2,   # hydrophilicity
]

AAINDEX_BASIC_KEYS = [PHYSchem_KEYS[i] for i in AAINDEX_BASIC_SUBIDX]


# -------------------------
# Old Features
# -------------------------

# --- Deterministic physchem set (fixed + documented)
# Values are simple, stable proxies (not meant to be "best", meant to be consistent).
#
# physchem features (8):
#   hydrophobicity_kd, volume, polarity, charge, aromatic, aliphatic, hbond_donor, hbond_acceptor
#
# charge: -1, 0, +1 (rough)
# aromatic/aliphatic/hbond_* are binary proxies
#
# You can refine later, but DO NOT change ordering/keys once released.
PHYSchem_old_KEYS = [
    "hydrophobicity_kd",
    "volume",
    "polarity",
    "charge",
    "aromatic",
    "aliphatic",
    "hbond_donor",
    "hbond_acceptor",
]

PHYSchem_old: Dict[str, List[float]] = {
    # KD hydrophobicity, volume (A^3 proxy), polarity (proxy), charge, aromatic, aliphatic, donor, acceptor
    "A": [ 1.8,  88.6,  8.1,  0.0, 0.0, 1.0, 0.0, 0.0],
    "C": [ 2.5, 108.5,  5.5,  0.0, 0.0, 0.0, 0.0, 0.0],
    "D": [-3.5, 111.1, 13.0, -1.0, 0.0, 0.0, 0.0, 1.0],
    "E": [-3.5, 138.4, 12.3, -1.0, 0.0, 0.0, 0.0, 1.0],
    "F": [ 2.8, 189.9,  5.2,  0.0, 1.0, 0.0, 0.0, 0.0],
    "G": [-0.4,  60.1,  9.0,  0.0, 0.0, 0.0, 0.0, 0.0],
    "H": [-3.2, 153.2, 10.4,  0.5, 1.0, 0.0, 1.0, 1.0],
    "I": [ 4.5, 166.7,  5.2,  0.0, 0.0, 1.0, 0.0, 0.0],
    "K": [-3.9, 168.6, 11.3,  1.0, 0.0, 0.0, 1.0, 0.0],
    "L": [ 3.8, 166.7,  4.9,  0.0, 0.0, 1.0, 0.0, 0.0],
    "M": [ 1.9, 162.9,  5.7,  0.0, 0.0, 0.0, 0.0, 0.0],
    "N": [-3.5, 114.1, 11.6,  0.0, 0.0, 0.0, 1.0, 1.0],
    "P": [-1.6, 112.7,  8.0,  0.0, 0.0, 0.0, 0.0, 0.0],
    "Q": [-3.5, 143.8, 10.5,  0.0, 0.0, 0.0, 1.0, 1.0],
    "R": [-4.5, 173.4, 10.5,  1.0, 0.0, 0.0, 1.0, 0.0],
    "S": [-0.8,  89.0,  9.2,  0.0, 0.0, 0.0, 1.0, 1.0],
    "T": [-0.7, 116.1,  8.6,  0.0, 0.0, 0.0, 1.0, 1.0],
    "V": [ 4.2, 140.0,  5.9,  0.0, 0.0, 1.0, 0.0, 0.0],
    "W": [-0.9, 227.8,  5.4,  0.0, 1.0, 0.0, 1.0, 1.0],
    "Y": [-1.3, 193.6,  6.2,  0.0, 1.0, 0.0, 1.0, 1.0],
    "X": [ 0.0,   0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0],
}

# --- AAindex basic subset (fixed)
# Here we choose a small subset of the above proxies (6 dims) to keep it stable.
AAINDEX_BASIC_KEYS_old = [
    "aaindex.hydrophobicity_kd",
    "aaindex.volume",
    "aaindex.polarity",
    "aaindex.charge",
    "aaindex.aromatic",
    "aaindex.aliphatic",
]

AAINDEX_BASIC_SUBIDX_OLD = [0,1,2,3,4,5]


# -------------------------
# Atom Features
# -------------------------



