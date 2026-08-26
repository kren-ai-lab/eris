from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchdrug")

from torchdrug import data, layers, transforms  # noqa: E402
from torchdrug.layers import geometry  # noqa: E402

from eris.core.learned.backends.gearnet.preprocessing import (  # noqa: E402
    BiologicalIdentityTable,
    BiologicalResidueIdentity,
    TransportNodeRecord,
    construct_gearnet_input,
)


def _identity(index: int, residue_name: str) -> BiologicalResidueIdentity:
    residue_number = index + 1
    return BiologicalResidueIdentity(
        index=index,
        chain_id="A",
        residue_number=residue_number,
        insertion_code="",
        residue_name=residue_name,
        node_id=f"A:{residue_number}::{residue_name}",
    )


def _table() -> BiologicalIdentityTable:
    return BiologicalIdentityTable(
        records=(
            _identity(0, "ALA"),
            _identity(1, "GLY"),
            _identity(2, "SER"),
            _identity(3, "THR"),
        )
    )


def _records() -> tuple[TransportNodeRecord, ...]:
    return (
        TransportNodeRecord(0, 0, "ALA", (0.0, 0.0, 0.0)),
        TransportNodeRecord(1, 2, "SER", (3.8, 0.0, 0.0)),
        TransportNodeRecord(2, 3, "THR", (7.6, 0.0, 0.0)),
    )


def _clean_pdb_atom(
    serial: int,
    residue_name: str,
    residue_number: int,
    x: float,
) -> str:
    return (
        f"ATOM  {serial:5d}  CA  {residue_name:>3s} A{residue_number:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{0.0:6.2f}           C  "
    )


def _upstream_graph(path: Path):
    protein = data.Protein.from_pdb(
        str(path),
        atom_feature=None,
        bond_feature=None,
        residue_feature="default",
    )
    protein = transforms.ProteinView(view="residue")({"graph": protein})["graph"]
    packed = data.Protein.pack([protein])
    construction = layers.GraphConstruction(
        node_layers=[geometry.AlphaCarbonNode()],
        edge_layers=[
            geometry.SequentialEdge(max_distance=2),
            geometry.SpatialEdge(
                radius=10.0,
                min_distance=5,
                max_distance=None,
                max_num_neighbors=32,
            ),
            geometry.KNNEdge(k=10, min_distance=5, max_distance=None),
        ],
        edge_feature="gearnet",
    )
    return construction(packed)


def test_real_stack_constructs_packed_graph_and_preserves_explicit_carrier():
    prepared = construct_gearnet_input(_records(), _table())

    assert isinstance(prepared.graph, data.PackedProtein)
    assert prepared.graph.view == "residue"
    assert int(prepared.graph.batch_size) == 1
    assert int(prepared.graph.num_node) == int(prepared.graph.num_residue) == 3
    assert prepared.carrier.tolist() == [0, 2, 3]
    assert prepared.node_ids.tolist() == ["A:1::ALA", "A:3::SER", "A:4::THR"]
    assert prepared.graph.eris_residue_index.tolist() == [0, 2, 3]
    assert prepared.graph.node_feature.shape == (3, 21)
    assert prepared.graph.edge_feature.shape[1] == 59
    assert int(prepared.graph.num_relation) == 7


def test_real_stack_graph_construction_is_deterministic():
    first = construct_gearnet_input(_records(), _table())
    second = construct_gearnet_input(_records(), _table())

    assert torch.equal(first.graph.node_feature, second.graph.node_feature)
    assert torch.equal(first.graph.node_position, second.graph.node_position)
    assert torch.equal(first.graph.edge_list, second.graph.edge_list)
    assert torch.equal(first.graph.edge_feature, second.graph.edge_feature)
    assert np.array_equal(first.carrier, second.carrier)
    assert np.array_equal(first.node_ids, second.node_ids)


def test_synthetic_transport_is_equivalent_to_upstream_clean_pdb(tmp_path):
    clean_path = tmp_path / "clean.pdb"
    clean_path.write_text(
        "\n".join(
            [
                _clean_pdb_atom(1, "ALA", 1, 0.0),
                _clean_pdb_atom(2, "SER", 2, 3.8),
                _clean_pdb_atom(3, "THR", 3, 7.6),
                "TER",
                "END",
            ]
        )
        + "\n",
        encoding="ascii",
    )
    upstream = _upstream_graph(clean_path)
    synthetic = construct_gearnet_input(_records(), _table()).graph

    assert int(upstream.num_node) == int(synthetic.num_node)
    assert torch.equal(upstream.node_feature, synthetic.node_feature)
    assert torch.equal(upstream.node_position, synthetic.node_position)
    assert torch.equal(upstream.edge_list, synthetic.edge_list)
    assert torch.equal(upstream.edge_feature, synthetic.edge_feature)
