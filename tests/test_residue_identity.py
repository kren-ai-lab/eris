from __future__ import annotations

import gemmi

from eris.core.io import extract_residue_bundle
from eris.core.residue_identity import format_residue_id


def test_format_residue_id_preserves_existing_semantics():
    assert (
        format_residue_id(
            chain_id="A",
            residue_number=10,
            insertion_code=" ",
            residue_name="ALA",
        )
        == "A:10::ALA"
    )
    assert (
        format_residue_id(
            chain_id="A",
            residue_number=10,
            insertion_code="A",
            residue_name="GLY",
        )
        == "A:10:A:GLY"
    )
    assert (
        format_residue_id(
            chain_id="B",
            residue_number=2,
            insertion_code=None,
            residue_name="THR",
        )
        == "B:2::THR"
    )


def test_residue_extraction_keeps_gap_and_insertion_code_ids():
    structure = gemmi.read_structure_string(
        """\
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00           C
ATOM      2  CA  GLY A   5       3.800   0.000   0.000  1.00 10.00           C
ATOM      3  CA  SER A  10A      7.600   0.000   0.000  1.00 10.00           C
TER
END
"""
    )

    bundle = extract_residue_bundle(structure[0][0])

    assert bundle["residue_ids"].tolist() == ["A:1::ALA", "A:5::GLY", "A:10:A:SER"]
