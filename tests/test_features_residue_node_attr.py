from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from eris.features.node import constants as residue_constants
from eris.features.node.constants import (
    AA_VOCAB,
    AAINDEX_BASIC_KEYS,
    AAINDEX_BASIC_SUBIDX,
    PHYSCHEM_LOOKUPS,
    UNKNOWN_INDEX,
    PHYSchem_KEYS,
)
from eris.features.node.residue import build_node_attr

EXPECTED_PHYSCHEM_BASELINE = np.array([
    [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8, 1.9,
     -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3, 0.0],
    [0.62, 0.29, -0.90, -0.74, 1.19, 0.48, -0.40, 1.38, -1.50, 1.06,
     0.64, -0.78, 0.12, -0.85, -2.53, -0.18, -0.05, 1.08, 0.81, 0.26, 0.0],
    [-0.50, -1.00, 3.00, 3.00, -2.50, 0.00, -0.50, -1.80, 3.00, -1.80,
     -1.30, 0.20, 0.00, 0.20, 3.00, 0.30, -0.40, -1.50, -3.40, -2.30, 0.0],
    [15.0, 47.0, 59.0, 73.0, 91.0, 1.0, 82.0, 57.0, 72.0, 57.0, 75.0,
     58.0, 41.0, 72.0, 100.0, 31.0, 45.0, 43.0, 130.0, 107.0, 0.0],
    [88.6, 108.5, 111.1, 138.4, 189.9, 60.1, 153.2, 166.7, 168.6, 166.7,
     162.9, 114.1, 112.7, 143.8, 173.4, 89.0, 116.1, 140.0, 227.8, 193.6, 0.0],
    [89.09, 121.15, 133.10, 147.13, 165.19, 75.07, 155.16, 131.17, 146.19,
     131.17, 149.21, 132.12, 115.13, 146.15, 174.20, 105.09, 119.12, 117.15,
     204.23, 181.19, 0.0],
    [0.17, 0.24, -1.23, -2.02, 1.13, 0.01, -0.96, 0.31, -0.99, 0.56, 0.23,
     -0.42, -0.45, -0.58, -1.01, -0.13, -0.14, 0.07, 1.85, 0.94, 0.0],
    [8.1, 5.5, 13.0, 12.3, 5.2, 9.0, 10.4, 5.2, 11.3, 4.9, 5.7, 11.6, 8.0,
     10.5, 10.5, 9.2, 8.6, 5.9, 5.4, 6.2, 0.0],
    [0.357, 0.346, 0.511, 0.497, 0.314, 0.544, 0.323, 0.462, 0.466, 0.365,
     0.295, 0.463, 0.509, 0.493, 0.529, 0.507, 0.444, 0.386, 0.305, 0.420, 0.0],
    [1.45, 0.77, 1.01, 1.53, 1.12, 0.53, 1.24, 1.00, 1.07, 1.34, 1.20, 0.73,
     0.59, 1.17, 0.79, 0.79, 0.82, 1.14, 1.14, 0.61, 0.0],
    [0.97, 1.30, 0.54, 0.37, 1.28, 0.81, 0.71, 1.60, 0.74, 1.22, 1.67, 0.65,
     0.62, 1.23, 0.90, 0.72, 1.20, 1.65, 1.19, 1.29, 0.0],
    [0.66, 1.19, 1.46, 0.74, 0.60, 1.56, 0.95, 0.47, 1.01, 0.59, 0.60, 1.56,
     1.52, 0.98, 0.95, 1.43, 0.96, 0.50, 0.96, 1.14, 0.0],
    [0.06, -0.22, 0.19, 0.25, -0.15, 0.16, -0.05, -0.20, 0.27, -0.21, -0.09,
     0.00, 0.12, 0.06, 0.21, 0.10, 0.05, -0.22, -0.23, -0.03, 0.0],
    [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
], dtype=np.float32)


EXPECTED_PHYSCHEM_KEYS = [
    "physchem.hydrophobicity_kd",
    "physchem.hydrophobicity_eisenberg",
    "physchem.hydrophilicity",
    "physchem.sidechain_mass",
    "physchem.volume",
    "physchem.molecular_weight",
    "physchem.boman",
    "physchem.polarity",
    "physchem.flexibility",
    "physchem.cf_helix",
    "physchem.cf_sheet",
    "physchem.cf_turn",
    "physchem.top_idp",
    "physchem.hbond_donor",
    "physchem.hbond_acceptor",
]


def test_physchem_baseline_is_frozen_and_roxy_independent():
    assert "".join(AA_VOCAB) == "ACDEFGHIKLMNPQRSTVWYX"
    assert PHYSchem_KEYS == EXPECTED_PHYSCHEM_KEYS

    actual = np.stack([lookup for _, lookup in PHYSCHEM_LOOKUPS], axis=0)
    assert actual.shape == (15, 21)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, EXPECTED_PHYSCHEM_BASELINE)
    np.testing.assert_array_equal(actual[:, UNKNOWN_INDEX], np.zeros(15, dtype=np.float32))

    source = Path(residue_constants.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        name.split(".", 1)[0]
        for node in ast.walk(tree)
        for name in (
            [alias.name for alias in node.names]
            if isinstance(node, ast.Import)
            else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
        )
    }
    assert "roxy" not in imported_roots


def test_physchem_feature_preserves_values_and_column_order():
    aa_index = np.array([AA_VOCAB.index("A"), AA_VOCAB.index("W"), UNKNOWN_INDEX])
    x, keys = build_node_attr(aa_index, mode="physchem", return_keys=True)

    assert keys == EXPECTED_PHYSCHEM_KEYS
    assert x.shape == (3, 15)
    assert x.dtype == np.float32
    np.testing.assert_array_equal(x, EXPECTED_PHYSCHEM_BASELINE[:, aa_index].T)


def test_aaindex_basic_preserves_frozen_subset():
    expected_subidx = [0, 4, 7, 8, 6, 2]
    aa_index = np.array([AA_VOCAB.index("C"), UNKNOWN_INDEX])
    x, keys = build_node_attr(aa_index, mode="aaindex_basic", return_keys=True)

    assert AAINDEX_BASIC_SUBIDX == expected_subidx
    assert keys == [EXPECTED_PHYSCHEM_KEYS[i] for i in expected_subidx]
    assert keys == AAINDEX_BASIC_KEYS
    assert x.shape == (2, 6)
    assert x.dtype == np.float32
    np.testing.assert_array_equal(
        x,
        EXPECTED_PHYSCHEM_BASELINE[expected_subidx][:, aa_index].T,
    )


def test_build_node_attr_keys_and_shapes_full():
    aa_index = np.array([0, 1, 2, UNKNOWN_INDEX], dtype=np.int16)  # A, C, D, X (assuming AA_VOCAB order)
    plddt = np.array([50.0, np.nan, 90.0, 10.0], dtype=np.float32)
    bfac = np.array([20.0, 30.0, np.nan, 0.0], dtype=np.float32)
    has_ca = np.array([True, True, False, True])

    x, keys = build_node_attr(
        aa_index,
        plddt_ca=plddt,
        bfactor_ca=bfac,
        has_ca=has_ca,
        mode="flags+bfactor_ca+onehot+aa_index+plddt_ca+physchem+aaindex_basic",
        missing="zero",
        return_keys=True
    )

    # keys stable / canonical ordering
    assert keys[0].startswith("aa_onehot.")
    assert "aa_index" in keys
    assert "plddt_ca" in keys
    assert "bfactor_ca" in keys
    assert keys[-4:] == ["flag.is_gly", "flag.is_pro", "flag.is_unk", "flag.has_ca"]

    # dimensions
    expected_F = (
        len(AA_VOCAB)          # aa_onehot
        + 1                    # aa_index
        + len(PHYSchem_KEYS)   # physchem
        + len(AAINDEX_BASIC_KEYS)  # aaindex_basic
        + 1                    # plddt_ca
        + 1                    # bfactor_ca
        + 4                    # flags
    )
    assert x.shape == (4, expected_F)
    assert x.dtype == np.float32

    # missing -> zero policy
    # plddt second was nan -> 0
    plddt_col = keys.index("plddt_ca")
    assert float(x[1, plddt_col]) == 0.0


def test_build_node_attr_mode_empty_returns_empty_features():
    aa_index = np.array([0, 1, UNKNOWN_INDEX], dtype=np.int16)
    x, keys = build_node_attr(aa_index, mode="", return_keys=True)
    assert x.shape == (3, 0)
    assert keys == []


def test_build_node_attr_missing_nan_policy_keeps_nans():
    aa_index = np.array([0, 1], dtype=np.int16)
    plddt = np.array([np.nan, 10.0], dtype=np.float32)

    x, keys = build_node_attr(aa_index, plddt_ca=plddt, mode="plddt_ca", missing="nan", return_keys=True)
    assert keys == ["plddt_ca"]
    assert np.isnan(x[0, 0])
    assert float(x[1, 0]) == 10.0
