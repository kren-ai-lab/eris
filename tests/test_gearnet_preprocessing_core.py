from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from eris.core.learned.backends.gearnet.preprocessing import (
    BiologicalIdentityTable,
    BiologicalResidueIdentity,
    GearNetPreprocessingError,
    TransportNodeRecord,
    load_selected_biological_structure,
)
from eris.core.types import StructureInput
from eris.exceptions import ErisError


def _pdb_atom(
    serial: int,
    residue_name: str,
    chain_id: str,
    residue_number: int,
    x: float,
    *,
    insertion_code: str = " ",
) -> str:
    return (
        f"ATOM  {serial:5d}  CA  {residue_name:>3s} {chain_id}{residue_number:4d}"
        f"{insertion_code}   {x:8.3f}{0.0:8.3f}{0.0:8.3f}"
        f"{1.0:6.2f}{10.0:6.2f}           C  "
    )


def _write_selection_fixture(path: Path) -> None:
    lines = [
        "MODEL        1",
        _pdb_atom(1, "ALA", "A", 1, 0.0),
        _pdb_atom(2, "GLY", "A", 5, 3.8),
        _pdb_atom(3, "SER", "A", 10, 7.6, insertion_code="A"),
        "TER",
        _pdb_atom(4, "THR", "B", 2, 11.4),
        "TER",
        "ENDMDL",
        "MODEL        2",
        _pdb_atom(5, "VAL", "A", 7, 15.2),
        "TER",
        "ENDMDL",
        "END",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _identity(index: int, *, node_id: str | None = None) -> BiologicalResidueIdentity:
    residue_number = index + 1
    return BiologicalResidueIdentity(
        index=index,
        chain_id="A",
        residue_number=residue_number,
        insertion_code="",
        residue_name="ALA",
        node_id=node_id or f"A:{residue_number}::ALA",
    )


def _records() -> tuple[TransportNodeRecord, ...]:
    return (
        TransportNodeRecord(0, 0, "ALA", (0.0, 0.0, 0.0)),
        TransportNodeRecord(1, 2, "SER", (3.8, 0.0, 0.0)),
        TransportNodeRecord(2, 3, "THR", (7.6, 0.0, 0.0)),
    )


def test_model_and_explicit_chain_selection_preserve_identity_order(tmp_path):
    source = tmp_path / "selection.pdb"
    _write_selection_fixture(source)

    selected = load_selected_biological_structure(
        StructureInput(source=str(source), chain_id="A", model_index=0)
    )

    assert selected.input_format == "pdb"
    assert selected.model_index == 0
    assert selected.chain.name == "A"
    assert [record.index for record in selected.biological_table.records] == [0, 1, 2]
    assert [record.residue_number for record in selected.biological_table.records] == [1, 5, 10]
    assert [record.insertion_code for record in selected.biological_table.records] == ["", "", "A"]
    assert [record.node_id for record in selected.biological_table.records] == [
        "A:1::ALA",
        "A:5::GLY",
        "A:10:A:SER",
    ]


def test_first_chain_and_model_index_preserve_existing_eris_behavior(tmp_path):
    source = tmp_path / "selection.pdb"
    _write_selection_fixture(source)

    first_chain = load_selected_biological_structure(
        StructureInput(source=str(source), chain_id=None, model_index=0)
    )
    second_model = load_selected_biological_structure(
        StructureInput(source=str(source), chain_id="A", model_index=1)
    )

    assert first_chain.chain.name == "A"
    assert [record.node_id for record in first_chain.biological_table.records] == [
        "A:1::ALA",
        "A:5::GLY",
        "A:10:A:SER",
    ]
    assert [record.node_id for record in second_model.biological_table.records] == ["A:7::VAL"]


@pytest.mark.parametrize(
    "structure_input",
    [
        StructureInput(source="{source}", chain_id="Z", model_index=0),
        StructureInput(source="{source}", chain_id="A", model_index=2),
    ],
)
def test_missing_chain_and_model_raise_existing_eris_errors(tmp_path, structure_input):
    source = tmp_path / "selection.pdb"
    _write_selection_fixture(source)
    resolved = StructureInput(
        source=structure_input.source.format(source=source),
        chain_id=structure_input.chain_id,
        model_index=structure_input.model_index,
    )

    with pytest.raises(ErisError):
        load_selected_biological_structure(resolved)


def test_biological_table_rejects_duplicate_node_ids():
    with pytest.raises(GearNetPreprocessingError, match="node_ids must be unique"):
        BiologicalIdentityTable(
            records=(_identity(0, node_id="A:1::ALA"), _identity(1, node_id="A:1::ALA"))
        )


def test_biological_table_rejects_reordered_indexes_and_invalid_carriers():
    with pytest.raises(GearNetPreprocessingError, match="contiguous"):
        BiologicalIdentityTable(records=(_identity(1), _identity(0)))

    table = BiologicalIdentityTable(records=tuple(_identity(index) for index in range(2)))
    with pytest.raises(GearNetPreprocessingError, match="carrier indexes must be unique"):
        table.node_ids_for([0, 0])
    with pytest.raises(GearNetPreprocessingError, match="outside the identity table"):
        table.node_ids_for([2])


def test_mapping_uses_biological_carrier_not_transport_position():
    table = BiologicalIdentityTable(records=tuple(_identity(index) for index in range(4)))

    assert table.node_ids_for([record.biological_index for record in _records()]) == (
        "A:1::ALA",
        "A:3::ALA",
        "A:4::ALA",
    )


def test_policy_neutral_preprocessing_import_does_not_load_optional_frameworks():
    script = """
import sys
import eris.core.learned.backends.gearnet.preprocessing

blocked = {'torch', 'torch_scatter', 'torch_cluster', 'rdkit'}
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise SystemExit(f'optional GearNet frameworks loaded: {loaded}')
"""

    subprocess.run([sys.executable, "-c", script], check=True)


def test_model_coordinate_can_be_recorded_without_defining_selection_policy():
    identity = BiologicalResidueIdentity(
        index=0,
        chain_id="A",
        residue_number=1,
        insertion_code="",
        residue_name="MSE",
        node_id="A:1::MSE",
        model_coordinate=(1.0, 2.0, 3.0),
    )

    assert identity.model_coordinate == (1.0, 2.0, 3.0)
    assert np.isfinite(identity.model_coordinate).all()
