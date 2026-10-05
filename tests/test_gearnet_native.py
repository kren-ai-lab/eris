"""Current native GearNet-Edge scientific and runtime contracts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

import eris
from eris.core.learned.backends.gearnet import dependencies
from eris.core.learned.backends.gearnet.metadata import (
    GEARNET_EDGE_ENCODER_VERSION,
    GEARNET_EDGE_IMPLEMENTATION,
)
from eris.core.learned.backends.gearnet.native_graph import construct_native_gearnet_graph
from eris.core.learned.backends.gearnet.preprocessing import (
    GearNetPreprocessingError,
    load_selected_biological_structure,
    select_gearnet_transport_records,
)
from eris.core.types import StructureInput


DATA_DIR = Path(__file__).with_name("data")
CHECKPOINT_ENV = "ERIS_GEARNET_CHECKPOINT"
GRAPH_ARRAYS = (
    "node_feature",
    "node_position",
    "edge_source",
    "edge_destination",
    "edge_relation",
    "edge_feature",
    "line_edge_source",
    "line_edge_destination",
    "line_angle_relation",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _reference_graph(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name] for name in GRAPH_ARRAYS}


def _records(path: Path):
    selected = load_selected_biological_structure(
        StructureInput(source=path, chain_id="A", model_index=0)
    )
    return select_gearnet_transport_records(selected)


def test_gearnet_v01_rejects_multichain_selection(tmp_path: Path) -> None:
    source = tmp_path / "multi.pdb"
    source.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00           C  ",
                "TER",
                "ATOM      2  CA  GLY B   1       1.000   0.000   0.000  1.00 20.00           C  ",
                "TER",
                "END",
                "",
            ]
        ),
        encoding="ascii",
    )

    with pytest.raises(GearNetPreprocessingError, match="exactly one selected chain"):
        load_selected_biological_structure(StructureInput(source=source))


def _atom_line(
    serial: int,
    residue_name: str,
    residue_number: int,
    x: float,
    *,
    record: str = "ATOM",
    atom_name: str = "CA",
    altloc: str = " ",
    occupancy: float = 1.0,
) -> str:
    return (
        f"{record:<6}{serial:5d} {atom_name:>4s}{altloc}{residue_name:>3s} A"
        f"{residue_number:4d}    {x:8.3f}{0.0:8.3f}{0.0:8.3f}"
        f"{occupancy:6.2f}{10.0:6.2f}           C  "
    )


def test_frozen_reference_payloads_and_metadata_are_path_safe():
    primary = DATA_DIR / "gearnet_reference.pdb"
    metadata = json.loads((DATA_DIR / "gearnet_reference.json").read_text(encoding="utf-8"))

    assert _sha256(primary) == "7ef6662ceadaae64f187fa974ba866af02ab5337220bdc9fe005afaeb2b9814b"
    assert metadata["input"]["sha256"] == _sha256(primary)
    assert metadata["input"]["filename"] == primary.name
    assert metadata["reference_provenance"].startswith("Frozen upstream GearNet/TorchDrug")
    assert "\\" not in json.dumps(metadata, sort_keys=True)

    graph = _reference_graph(DATA_DIR / "gearnet_reference_graph.npz")
    assert graph["node_feature"].shape == (12, 21)
    assert graph["edge_feature"].shape[1] == 59
    assert graph["line_angle_relation"].ndim == 1


@pytest.mark.parametrize(
    ("input_name", "graph_name"),
    (
        ("gearnet_reference.pdb", "gearnet_reference_graph.npz"),
        ("gearnet_spatial_reference.pdb", "gearnet_spatial_reference_graph.npz"),
    ),
)
def test_native_graph_matches_each_frozen_reference_exactly(input_name: str, graph_name: str):
    native = construct_native_gearnet_graph(_records(DATA_DIR / input_name))
    expected = _reference_graph(DATA_DIR / graph_name)

    for name in GRAPH_ARRAYS:
        np.testing.assert_array_equal(getattr(native, name), expected[name], strict=True)


def test_spatial_reference_freezes_boundary_tie_and_neighbor_order():
    metadata = json.loads((DATA_DIR / "gearnet_spatial_reference.json").read_text(encoding="utf-8"))
    probe_metadata = json.loads(
        (DATA_DIR / "gearnet_spatial_max_neighbors_reference.json").read_text(encoding="utf-8")
    )
    graph = _reference_graph(DATA_DIR / "gearnet_spatial_reference_graph.npz")
    with np.load(
        DATA_DIR / "gearnet_spatial_max_neighbors_reference.npz", allow_pickle=False
    ) as probe:
        target_zero_sources = probe["spatial_edge_source"][probe["spatial_edge_destination"] == 0]

    assert metadata["purpose"] == "frozen spatial reference"
    assert metadata["observations"]["radius_comparison"] == "strict_less_than"
    assert metadata["observations"]["equal_distance_tie"]["sources_in_reference_order"] == [6, 7]
    assert np.flatnonzero(graph["edge_relation"] == 5).tolist() == list(range(54, 68))
    assert target_zero_sources.tolist() == probe_metadata["retained_sources_in_reference_order"]
    assert len(target_zero_sources) == 32
    assert set(np.unique(graph["line_angle_relation"])) <= set(range(8))


def test_biological_policy_preserves_identity_and_rejects_invalid_retained_residues(tmp_path):
    source = tmp_path / "selection.pdb"
    source.write_text(
        "\n".join(
            (
                _atom_line(1, "ALA", 1, 1.0),
                _atom_line(2, "MSE", 2, 2.0, record="HETATM"),
                _atom_line(3, "GLY", 3, 3.0),
                _atom_line(4, "HOH", 4, 4.0, record="HETATM", atom_name="O"),
                _atom_line(5, "SER", 5, 5.0, atom_name="N"),
                "TER",
                "END",
            )
        )
        + "\n",
        encoding="ascii",
    )
    selected = load_selected_biological_structure(
        StructureInput(source=source, chain_id="A", model_index=0)
    )

    with pytest.raises(GearNetPreprocessingError, match="missing CA.*A:5::SER"):
        select_gearnet_transport_records(selected)

    source.write_text(
        "\n".join(
            (
                _atom_line(1, "ALA", 1, 1.0, altloc="A", occupancy=0.2),
                _atom_line(2, "ALA", 1, 2.0, altloc="B", occupancy=0.8),
                _atom_line(3, "MSE", 2, 3.0, record="HETATM"),
                _atom_line(4, "GLY", 4, 4.0),
                "TER",
                "END",
            )
        )
        + "\n",
        encoding="ascii",
    )
    selected = load_selected_biological_structure(
        StructureInput(source=source, chain_id="A", model_index=0)
    )
    records = select_gearnet_transport_records(selected)

    assert [record.biological_index for record in records] == [0, 2]
    assert [record.coordinate for record in records] == [(1.0, 0.0, 0.0), (4.0, 0.0, 0.0)]


def test_native_dependency_boundary_and_current_cache_identity(monkeypatch):
    assert dependencies.GEARNET_PYTHON_REQUIREMENT == ">=3.11,<3.13"
    assert [item.requirement for item in dependencies.GEARNET_DEPENDENCY_REQUIREMENTS] == [
        "==2.11.0"
    ]
    assert GEARNET_EDGE_ENCODER_VERSION == "2"
    assert GEARNET_EDGE_IMPLEMENTATION.endswith("GearNetEdgeBackend")

    monkeypatch.setattr(dependencies, "_current_python_version", lambda: (3, 13))
    unavailable = dependencies.probe_gearnet_availability()
    assert unavailable.available is False
    assert ">=3.11,<3.13" in unavailable.reason


def test_public_native_production_matches_frozen_embedding_when_checkpoint_is_configured(tmp_path):
    checkpoint = os.environ.get(CHECKPOINT_ENV)
    if not checkpoint:
        pytest.skip(f"{CHECKPOINT_ENV} is not configured")

    artifact = eris.embed(
        DATA_DIR / "gearnet_reference.pdb",
        representation="gearnet_edge",
        checkpoint_path=Path(checkpoint),
        output=tmp_path / "artifact",
        use_cache=False,
    )
    with np.load(DATA_DIR / "gearnet_reference_embeddings.npz", allow_pickle=False) as golden:
        np.testing.assert_allclose(
            artifact.data["node_embedding"], golden["node_embedding"], rtol=1e-6, atol=2e-6
        )
        np.testing.assert_allclose(
            artifact.data["global_embedding"],
            golden["global_embedding"],
            rtol=1e-6,
            atol=2e-6,
        )
    assert artifact.meta["encoder"]["version"] == "2"
