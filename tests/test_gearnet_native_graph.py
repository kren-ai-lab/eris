"""Exact structural parity tests for the internal native GearNet graph builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eris.core.learned.backends.gearnet.native_graph import construct_native_gearnet_graph
from eris.core.learned.backends.gearnet.preprocessing import (
    TransportNodeRecord,
    load_selected_biological_structure,
    select_gearnet_transport_records,
)
from eris.core.types import StructureInput


DATA_DIR = Path(__file__).with_name("data")
_ARRAY_NAMES = (
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


def _records(path: Path) -> tuple[TransportNodeRecord, ...]:
    selected = load_selected_biological_structure(
        StructureInput(source=path, chain_id="A", model_index=0)
    )
    return select_gearnet_transport_records(selected)


def _graph_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as fixture:
        return {name: fixture[name] for name in _ARRAY_NAMES}


@pytest.mark.parametrize(
    ("input_name", "graph_name"),
    (
        ("gearnet_legacy_reference.pdb", "gearnet_legacy_reference_graph.npz"),
        ("gearnet_legacy_spatial_reference.pdb", "gearnet_legacy_spatial_reference_graph.npz"),
    ),
)
def test_native_graph_matches_each_frozen_legacy_reference_exactly(
    input_name: str, graph_name: str
) -> None:
    native = construct_native_gearnet_graph(_records(DATA_DIR / input_name))
    expected = _graph_arrays(DATA_DIR / graph_name)
    for name in _ARRAY_NAMES:
        np.testing.assert_array_equal(getattr(native, name), expected[name], strict=True)


def test_native_spatial_search_preserves_frozen_max_neighbor_order() -> None:
    records = tuple(
        TransportNodeRecord(
            transport_index=index,
            biological_index=index,
            residue_name="ALA",
            coordinate=(0.0, 0.0, 0.0)
            if index == 0
            else (float(100 + index), 0.0, 0.0)
            if index < 5
            else (5.0, 0.0, 0.0),
        )
        for index in range(40)
    )
    native = construct_native_gearnet_graph(records)
    with np.load(
        DATA_DIR / "gearnet_legacy_spatial_max_neighbors_probe.npz", allow_pickle=False
    ) as probe:
        mask = (native.edge_relation == 5) & (native.edge_destination == 0)
        np.testing.assert_array_equal(native.edge_source[mask], probe["spatial_edge_source"])
        np.testing.assert_array_equal(native.node_position, probe["node_position"], strict=True)
