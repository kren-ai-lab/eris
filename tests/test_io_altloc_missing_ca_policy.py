from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import gemmi

from eris.core.io import extract_residue_bundle
from eris.features.node.residue import build_node_attr


def _chain_from_pdb_string(pdb_text: str, chain_id: str = "A", model_index: int = 0) -> gemmi.Chain:
    """
    Build a gemmi.Chain from an in-memory PDB string.
    This avoids shipping extra fixture files and keeps the test deterministic.
    """
    st = gemmi.read_structure_string(pdb_text)
    model = st[model_index]
    for ch in model:
        if ch.name == chain_id:
            return ch
    raise RuntimeError(f"Chain {chain_id} not found in test structure")


_PDB_ALTLOC_AND_MISSING_CA = """\
ATOM      1  CA AGLY A   1      12.000  10.000  10.000  0.40 20.00           C
ATOM      2  CA BGLY A   1      13.000  10.000  10.000  0.60 20.00           C
ATOM      3  N   ALA A   2      20.000  20.000  20.000  1.00 10.00           N
TER
END
"""


def test_altloc_max_occupancy_selects_highest_occ_ca():
    chain = _chain_from_pdb_string(_PDB_ALTLOC_AND_MISSING_CA)

    strc = extract_residue_bundle(
        chain,
        amino_acids_only=True,
        altloc="max_occupancy",
        missing_residue="skip",
    )

    # Residue 2 lacks CA, so only residue 1 remains
    assert strc["pos"].shape == (1, 3)
    # max occupancy altloc for CA should be B -> x=13.0
    assert np.isclose(strc["pos"][0, 0], 13.0, atol=1e-6)


def test_missing_ca_skip_drops_residues_without_ca():
    chain = _chain_from_pdb_string(_PDB_ALTLOC_AND_MISSING_CA)

    strc = extract_residue_bundle(
        chain,
        amino_acids_only=True,
        altloc="max_occupancy",
        missing_residue="skip",
    )
    assert strc["pos"].shape[0] == 1
    assert strc["residue_ids"].shape[0] == 1


def test_missing_ca_error_raises():
    chain = _chain_from_pdb_string(_PDB_ALTLOC_AND_MISSING_CA)

    with pytest.raises(Exception):
        extract_residue_bundle(
            chain,
            amino_acids_only=True,
            altloc="max_occupancy",
            missing_residue="error",
        )


def test_missing_ca_nan_keeps_residue_with_nans():
    chain = _chain_from_pdb_string(_PDB_ALTLOC_AND_MISSING_CA)

    strc = extract_residue_bundle(
        chain,
        amino_acids_only=True,
        altloc="max_occupancy",
        missing_residue="nan",
    )

    # Residue 1 has CA; residue 2 is kept as NaNs
    assert strc["pos"].shape == (2, 3)
    assert strc["residue_ids"].shape == (2,)

    # Residue 1 uses altloc B -> x=13.0
    assert np.isclose(strc["pos"][0, 0], 13.0, atol=1e-6)

    # Residue 2 missing CA -> NaNs
    assert np.isnan(strc["pos"][1]).all()
    assert np.isnan(strc["plddt"][1])
    assert np.isnan(strc["bfactor_ca"][1])


def test_ca_bfactor_propagates_to_residue_node_feature():
    chain = _chain_from_pdb_string(_PDB_ALTLOC_AND_MISSING_CA)

    strc = extract_residue_bundle(
        chain,
        amino_acids_only=True,
        altloc="max_occupancy",
        missing_residue="nan",
    )
    node_attr, keys = build_node_attr(
        strc["aa_index"],
        bfactor_ca=strc["bfactor_ca"],
        mode="bfactor_ca",
        return_keys=True,
    )

    assert np.isclose(strc["bfactor_ca"][0], 20.0)
    assert keys == ["bfactor_ca"]
    assert node_attr.shape == (2, 1)
    assert node_attr.dtype == np.float32
    assert np.isclose(node_attr[0, 0], 0.2)
    assert node_attr[1, 0] == 0.0
