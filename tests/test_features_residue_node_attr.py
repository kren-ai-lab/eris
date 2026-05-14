from __future__ import annotations

import numpy as np

from morpheo.features.residue import build_node_attr
from morpheo.features.constants import AA_VOCAB, UNKNOWN_INDEX, PHYSchem_KEYS, AAINDEX_BASIC_KEYS


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
        mode="flags+bfactor_ca+aa_onehot+aa_index+plddt_ca+physchem+aaindex_basic",
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
