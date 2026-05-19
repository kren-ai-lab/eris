from __future__ import annotations

from typing import Dict, List

# Canonical AA vocab for Eris (matches typical AA_index conventions)
# 20 standard + X for unknown
AA_VOCAB: List[str] = list("ACDEFGHIKLMNPQRSTVWY") + ["X"]
AA_TO_INDEX: Dict[str, int] = {aa: i for i, aa in enumerate(AA_VOCAB)}
UNKNOWN_AA = "X"
UNKNOWN_INDEX = AA_TO_INDEX[UNKNOWN_AA]

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
PHYSchem_KEYS = [
    "hydrophobicity_kd",
    "volume",
    "polarity",
    "charge",
    "aromatic",
    "aliphatic",
    "hbond_donor",
    "hbond_acceptor",
]

PHYSchem: Dict[str, List[float]] = {
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
AAINDEX_BASIC_KEYS = [
    "aaindex.hydrophobicity_kd",
    "aaindex.volume",
    "aaindex.polarity",
    "aaindex.charge",
    "aaindex.aromatic",
    "aaindex.aliphatic",
]
