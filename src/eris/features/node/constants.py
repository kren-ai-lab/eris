from __future__ import annotations

from typing import Dict, List

import numpy as np

# Canonical AA vocab for Eris (matches typical AA_index conventions)
# 20 standard + X for unknown
AA_VOCAB: List[str] = list("ACDEFGHIKLMNPQRSTVWY") + ["X"]
AA_TO_INDEX: Dict[str, int] = {aa: i for i, aa in enumerate(AA_VOCAB)}
UNKNOWN_AA = "X"
UNKNOWN_INDEX = AA_TO_INDEX[UNKNOWN_AA]


# -------------------------
# Residue physicochemical features
# -------------------------
# Values were internalized from the historical Roxy tables consumed by Eris to
# remove the runtime dependency. Columns follow AA_VOCAB, including X = 0.0.
_KD_LOOKUP = np.array(
    [
        1.8,
        2.5,
        -3.5,
        -3.5,
        2.8,
        -0.4,
        -3.2,
        4.5,
        -3.9,
        3.8,
        1.9,
        -3.5,
        -1.6,
        -3.5,
        -4.5,
        -0.8,
        -0.7,
        4.2,
        -0.9,
        -1.3,
        0.0,
    ],
    dtype=np.float32,
)
_EISENBERG_LOOKUP = np.array(
    [
        0.62,
        0.29,
        -0.90,
        -0.74,
        1.19,
        0.48,
        -0.40,
        1.38,
        -1.50,
        1.06,
        0.64,
        -0.78,
        0.12,
        -0.85,
        -2.53,
        -0.18,
        -0.05,
        1.08,
        0.81,
        0.26,
        0.0,
    ],
    dtype=np.float32,
)
_HYDROPHILICITY_LOOKUP = np.array(
    [
        -0.50,
        -1.00,
        3.00,
        3.00,
        -2.50,
        0.00,
        -0.50,
        -1.80,
        3.00,
        -1.80,
        -1.30,
        0.20,
        0.00,
        0.20,
        3.00,
        0.30,
        -0.40,
        -1.50,
        -3.40,
        -2.30,
        0.0,
    ],
    dtype=np.float32,
)
_SIDECHAIN_MASS_LOOKUP = np.array(
    [
        15.0,
        47.0,
        59.0,
        73.0,
        91.0,
        1.0,
        82.0,
        57.0,
        72.0,
        57.0,
        75.0,
        58.0,
        41.0,
        72.0,
        100.0,
        31.0,
        45.0,
        43.0,
        130.0,
        107.0,
        0.0,
    ],
    dtype=np.float32,
)
_VOLUME_LOOKUP = np.array(
    [
        88.6,
        108.5,
        111.1,
        138.4,
        189.9,
        60.1,
        153.2,
        166.7,
        168.6,
        166.7,
        162.9,
        114.1,
        112.7,
        143.8,
        173.4,
        89.0,
        116.1,
        140.0,
        227.8,
        193.6,
        0.0,
    ],
    dtype=np.float32,
)
_AA_MOLECULAR_WEIGHT_LOOKUP = np.array(
    [
        89.09,
        121.15,
        133.10,
        147.13,
        165.19,
        75.07,
        155.16,
        131.17,
        146.19,
        131.17,
        149.21,
        132.12,
        115.13,
        146.15,
        174.20,
        105.09,
        119.12,
        117.15,
        204.23,
        181.19,
        0.0,
    ],
    dtype=np.float32,
)
_BOMAN_LOOKUP = np.array(
    [
        0.17,
        0.24,
        -1.23,
        -2.02,
        1.13,
        0.01,
        -0.96,
        0.31,
        -0.99,
        0.56,
        0.23,
        -0.42,
        -0.45,
        -0.58,
        -1.01,
        -0.13,
        -0.14,
        0.07,
        1.85,
        0.94,
        0.0,
    ],
    dtype=np.float32,
)
_POLARITY_LOOKUP = np.array(
    [
        8.1,
        5.5,
        13.0,
        12.3,
        5.2,
        9.0,
        10.4,
        5.2,
        11.3,
        4.9,
        5.7,
        11.6,
        8.0,
        10.5,
        10.5,
        9.2,
        8.6,
        5.9,
        5.4,
        6.2,
        0.0,
    ],
    dtype=np.float32,
)
_FLEXIBILITY_LOOKUP = np.array(
    [
        0.357,
        0.346,
        0.511,
        0.497,
        0.314,
        0.544,
        0.323,
        0.462,
        0.466,
        0.365,
        0.295,
        0.463,
        0.509,
        0.493,
        0.529,
        0.507,
        0.444,
        0.386,
        0.305,
        0.420,
        0.0,
    ],
    dtype=np.float32,
)
_CF_HELIX_LOOKUP = np.array(
    [
        1.45,
        0.77,
        1.01,
        1.53,
        1.12,
        0.53,
        1.24,
        1.00,
        1.07,
        1.34,
        1.20,
        0.73,
        0.59,
        1.17,
        0.79,
        0.79,
        0.82,
        1.14,
        1.14,
        0.61,
        0.0,
    ],
    dtype=np.float32,
)
_CF_SHEET_LOOKUP = np.array(
    [
        0.97,
        1.30,
        0.54,
        0.37,
        1.28,
        0.81,
        0.71,
        1.60,
        0.74,
        1.22,
        1.67,
        0.65,
        0.62,
        1.23,
        0.90,
        0.72,
        1.20,
        1.65,
        1.19,
        1.29,
        0.0,
    ],
    dtype=np.float32,
)
_CF_TURN_LOOKUP = np.array(
    [
        0.66,
        1.19,
        1.46,
        0.74,
        0.60,
        1.56,
        0.95,
        0.47,
        1.01,
        0.59,
        0.60,
        1.56,
        1.52,
        0.98,
        0.95,
        1.43,
        0.96,
        0.50,
        0.96,
        1.14,
        0.0,
    ],
    dtype=np.float32,
)
_TOP_IDP_LOOKUP = np.array(
    [
        0.06,
        -0.22,
        0.19,
        0.25,
        -0.15,
        0.16,
        -0.05,
        -0.20,
        0.27,
        -0.21,
        -0.09,
        0.00,
        0.12,
        0.06,
        0.21,
        0.10,
        0.05,
        -0.22,
        -0.23,
        -0.03,
        0.0,
    ],
    dtype=np.float32,
)
_DONORS_LOOKUP = np.array(
    [
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        1,
        0,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
        0,
    ],
    dtype=np.float32,
)
_ACCEPTORS_LOOKUP = np.array(
    [
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        1,
        1,
        0,
        0,
        1,
        0,
    ],
    dtype=np.float32,
)

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
    0,  # hydrophobicity_kd
    4,  # volume
    7,  # polarity
    8,  # flexibility
    6,  # boman
    2,  # hydrophilicity
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
    "A": [1.8, 88.6, 8.1, 0.0, 0.0, 1.0, 0.0, 0.0],
    "C": [2.5, 108.5, 5.5, 0.0, 0.0, 0.0, 0.0, 0.0],
    "D": [-3.5, 111.1, 13.0, -1.0, 0.0, 0.0, 0.0, 1.0],
    "E": [-3.5, 138.4, 12.3, -1.0, 0.0, 0.0, 0.0, 1.0],
    "F": [2.8, 189.9, 5.2, 0.0, 1.0, 0.0, 0.0, 0.0],
    "G": [-0.4, 60.1, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "H": [-3.2, 153.2, 10.4, 0.5, 1.0, 0.0, 1.0, 1.0],
    "I": [4.5, 166.7, 5.2, 0.0, 0.0, 1.0, 0.0, 0.0],
    "K": [-3.9, 168.6, 11.3, 1.0, 0.0, 0.0, 1.0, 0.0],
    "L": [3.8, 166.7, 4.9, 0.0, 0.0, 1.0, 0.0, 0.0],
    "M": [1.9, 162.9, 5.7, 0.0, 0.0, 0.0, 0.0, 0.0],
    "N": [-3.5, 114.1, 11.6, 0.0, 0.0, 0.0, 1.0, 1.0],
    "P": [-1.6, 112.7, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "Q": [-3.5, 143.8, 10.5, 0.0, 0.0, 0.0, 1.0, 1.0],
    "R": [-4.5, 173.4, 10.5, 1.0, 0.0, 0.0, 1.0, 0.0],
    "S": [-0.8, 89.0, 9.2, 0.0, 0.0, 0.0, 1.0, 1.0],
    "T": [-0.7, 116.1, 8.6, 0.0, 0.0, 0.0, 1.0, 1.0],
    "V": [4.2, 140.0, 5.9, 0.0, 0.0, 1.0, 0.0, 0.0],
    "W": [-0.9, 227.8, 5.4, 0.0, 1.0, 0.0, 1.0, 1.0],
    "Y": [-1.3, 193.6, 6.2, 0.0, 1.0, 0.0, 1.0, 1.0],
    "X": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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

AAINDEX_BASIC_SUBIDX_OLD = [0, 1, 2, 3, 4, 5]


# -------------------------
# Atom Features
# -------------------------
