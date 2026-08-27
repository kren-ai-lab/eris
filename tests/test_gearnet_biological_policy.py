from __future__ import annotations

from pathlib import Path

import pytest

from eris.core.learned.backends.gearnet.preprocessing import (
    GearNetPreprocessingError,
    load_selected_biological_structure,
    select_gearnet_transport_records,
)
from eris.core.types import StructureInput

CANONICAL_AMINO_ACIDS = (
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
)


def _atom_line(
    serial: int,
    residue_name: str,
    residue_number: int,
    x: float,
    *,
    record: str = "ATOM",
    atom_name: str = "CA",
    insertion_code: str = " ",
    altloc: str = " ",
    occupancy: float = 1.0,
    element: str = "C",
) -> str:
    return (
        f"{record:<6}{serial:5d} {atom_name:>4s}{altloc}{residue_name:>3s} A"
        f"{residue_number:4d}{insertion_code}   {x:8.3f}{0.0:8.3f}{0.0:8.3f}"
        f"{occupancy:6.2f}{10.0:6.2f}          {element:>2s}  "
    )


def _write_pdb(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join([*lines, "TER", "END"]) + "\n", encoding="ascii")


def _select(path: Path, *, altloc: str = "first"):
    selected = load_selected_biological_structure(
        StructureInput(source=str(path), chain_id="A", model_index=0)
    )
    return selected, select_gearnet_transport_records(selected, altloc=altloc)


def test_canonical_residues_preserve_order_and_coordinates(tmp_path):
    path = tmp_path / "canonical.pdb"
    _write_pdb(
        path,
        [
            _atom_line(1, "ALA", 1, 1.0),
            _atom_line(2, "GLY", 3, 3.0),
            _atom_line(3, "SER", 10, 10.0, insertion_code="A"),
        ],
    )

    selected, records = _select(path)

    assert [record.node_id for record in selected.biological_table.records] == [
        "A:1::ALA",
        "A:3::GLY",
        "A:10:A:SER",
    ]
    assert [record.biological_index for record in records] == [0, 1, 2]
    assert [record.residue_name for record in records] == ["ALA", "GLY", "SER"]
    assert [record.coordinate for record in records] == [
        (1.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
    ]


def test_all_twenty_canonical_residues_are_accepted(tmp_path):
    path = tmp_path / "all-canonical.pdb"
    _write_pdb(
        path,
        [
            _atom_line(index, residue_name, index, float(index))
            for index, residue_name in enumerate(CANONICAL_AMINO_ACIDS, start=1)
        ],
    )

    _, records = _select(path)

    assert [record.residue_name for record in records] == list(CANONICAL_AMINO_ACIDS)
    assert [record.biological_index for record in records] == list(range(20))


def test_nonstandard_and_hetero_residues_are_excluded_with_sparse_carrier(tmp_path):
    path = tmp_path / "mixed.pdb"
    _write_pdb(
        path,
        [
            _atom_line(1, "ALA", 1, 1.0),
            _atom_line(2, "MSE", 2, 2.0, record="HETATM"),
            _atom_line(3, "GLY", 3, 3.0),
            _atom_line(4, "HOH", 4, 4.0, record="HETATM", atom_name="O", element="O"),
            _atom_line(5, "SER", 5, 5.0),
            _atom_line(6, "UNK", 6, 6.0),
            _atom_line(7, "SEC", 7, 7.0),
            _atom_line(8, "PYL", 8, 8.0),
            _atom_line(9, "LIG", 9, 9.0, record="HETATM"),
            _atom_line(
                10,
                "CA",
                10,
                10.0,
                record="HETATM",
                atom_name="CA",
                element="CA",
            ),
            _atom_line(11, "ALA", 11, 11.0, record="HETATM"),
        ],
    )

    selected, records = _select(path)

    assert [identity.residue_name for identity in selected.biological_table.records] == [
        "ALA",
        "MSE",
        "GLY",
        "HOH",
        "SER",
        "UNK",
        "SEC",
        "PYL",
        "LIG",
        "CA",
        "ALA",
    ]
    assert [record.residue_name for record in records] == ["ALA", "GLY", "SER"]
    assert [record.biological_index for record in records] == [0, 2, 4]
    assert "MET" not in [record.residue_name for record in records]


def test_missing_ca_in_retained_canonical_residue_fails_without_partial_result(tmp_path):
    path = tmp_path / "missing-ca.pdb"
    _write_pdb(
        path,
        [
            _atom_line(1, "ALA", 1, 1.0),
            _atom_line(2, "SER", 42, 2.0, atom_name="N", element="N"),
            _atom_line(3, "GLY", 50, 3.0),
        ],
    )
    selected = load_selected_biological_structure(
        StructureInput(source=str(path), chain_id="A", model_index=0)
    )

    with pytest.raises(GearNetPreprocessingError) as captured:
        select_gearnet_transport_records(selected)

    message = str(captured.value)
    assert "requires CA for every canonical residue" in message
    assert "missing CA" in message
    assert "A:42::SER" in message


def test_zero_canonical_residues_fails_before_transport(tmp_path):
    path = tmp_path / "empty-selection.pdb"
    _write_pdb(
        path,
        [
            _atom_line(1, "MSE", 1, 1.0, record="HETATM"),
            _atom_line(2, "HOH", 2, 2.0, record="HETATM", atom_name="O", element="O"),
            _atom_line(3, "LIG", 3, 3.0, record="HETATM"),
        ],
    )

    with pytest.raises(
        GearNetPreprocessingError,
        match="No valid canonical residues remain for GearNet v0.1",
    ):
        _select(path)


def test_existing_altloc_selector_remains_authoritative(tmp_path):
    path = tmp_path / "altloc.pdb"
    _write_pdb(
        path,
        [
            _atom_line(1, "ALA", 1, 1.0, altloc="A", occupancy=0.25),
            _atom_line(2, "ALA", 1, 2.0, altloc="B", occupancy=0.75),
        ],
    )

    _, first = _select(path)
    _, highest_occupancy = _select(path, altloc="max_occupancy")

    assert first[0].coordinate == (1.0, 0.0, 0.0)
    assert highest_occupancy[0].coordinate == (2.0, 0.0, 0.0)
