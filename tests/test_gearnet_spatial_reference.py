"""Integrity checks for the frozen GearNet SpatialEdge reference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).with_name("data")
INPUT_PATH = DATA_DIR / "gearnet_spatial_reference.pdb"
GRAPH_PATH = DATA_DIR / "gearnet_spatial_reference_graph.npz"
METADATA_PATH = DATA_DIR / "gearnet_spatial_reference.json"
PROBE_PATH = DATA_DIR / "gearnet_spatial_max_neighbors_reference.npz"
PROBE_METADATA_PATH = DATA_DIR / "gearnet_spatial_max_neighbors_reference.json"
PRIMARY_FIXTURE_SHA256 = {
    "gearnet_reference.pdb": "7ef6662ceadaae64f187fa974ba866af02ab5337220bdc9fe005afaeb2b9814b",
    "gearnet_reference_graph.npz": (
        "09f4990e372ecece2f71feee3c0b6d16a15514049332f4e6999753f3639770d6"
    ),
    "gearnet_reference_embeddings.npz": (
        "5487b04e70c2c10186cec2856ebb5be5bee1d3bcf2a7249351641e58e64db1ba"
    ),
    "gearnet_reference.json": "d6b795db8dc0272c61a586898fd9fc9d6dca99610491b7d73ac387a67dda23eb",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name] for name in arrays.files}


def _spatial_pairs(graph: dict[str, np.ndarray]) -> list[tuple[int, int]]:
    mask = graph["edge_relation"] == 5
    return list(
        zip(graph["edge_source"][mask].tolist(), graph["edge_destination"][mask], strict=True)
    )


def test_primary_reference_bytes_remain_unchanged():
    assert {
        path.name: _sha256(path) for path in (DATA_DIR / name for name in PRIMARY_FIXTURE_SHA256)
    } == PRIMARY_FIXTURE_SHA256


def test_supplemental_spatial_graph_integrity_and_exact_relation_layout():
    graph = _npz(GRAPH_PATH)
    metadata = _json(METADATA_PATH)

    assert set(graph) == {
        "carrier",
        "edge_destination",
        "edge_feature",
        "edge_relation",
        "edge_source",
        "line_angle_relation",
        "line_edge_destination",
        "line_edge_source",
        "node_feature",
        "node_ids",
        "node_position",
    }
    assert metadata["relation_counts"] == {
        "0": 10,
        "1": 11,
        "2": 12,
        "3": 11,
        "4": 10,
        "5": 14,
        "6": 51,
    }
    edges = graph["edge_relation"].shape[0]
    line_edges = graph["line_angle_relation"].shape[0]
    assert graph["edge_source"].shape == graph["edge_destination"].shape == (edges,)
    assert graph["edge_feature"].shape == (edges, 59)
    assert graph["edge_feature"].dtype == np.dtype("float32")
    assert np.isfinite(graph["edge_feature"]).all()
    assert graph["line_edge_source"].shape == graph["line_edge_destination"].shape == (line_edges,)
    assert graph["line_angle_relation"].shape == (line_edges,)
    spatial_indexes = np.flatnonzero(graph["edge_relation"] == 5)
    assert spatial_indexes.tolist() == list(range(54, 68))
    assert metadata["observations"]["spatial_relation_slice"] == {"count": 14, "start": 54}


def test_supplemental_spatial_runtime_observations_are_frozen():
    graph = _npz(GRAPH_PATH)
    metadata = _json(METADATA_PATH)
    observations = metadata["observations"]
    spatial_pairs = _spatial_pairs(graph)
    spatial_pair_set = set(spatial_pairs)

    assert observations["source_destination_convention"] == "edge_source sends to edge_destination"
    assert observations["radius_comparison"] == "strict_less_than"
    assert (9, 0) not in spatial_pair_set
    assert (0, 9) not in spatial_pair_set
    assert (4, 0) not in spatial_pair_set
    assert (0, 4) not in spatial_pair_set
    assert (5, 0) in spatial_pair_set
    assert (0, 5) in spatial_pair_set
    assert observations["target_zero_spatial_sources_in_reference_order"] == [5, 6, 7, 8]
    assert [source for source, destination in spatial_pairs if destination == 0] == [5, 6, 7, 8]
    assert observations["equal_distance_tie"] == {
        "destination": 0,
        "distance": 6.0,
        "sources_in_reference_order": [6, 7],
    }
    knn_pairs = set(
        zip(
            graph["edge_source"][graph["edge_relation"] == 6].tolist(),
            graph["edge_destination"][graph["edge_relation"] == 6].tolist(),
            strict=True,
        )
    )
    assert observations["cross_relation_duplicate_pairs"] == [
        list(pair) for pair in sorted(spatial_pair_set.intersection(knn_pairs))
    ]


def test_supplemental_line_graph_preserves_spatial_edges_and_angle_bins():
    graph = _npz(GRAPH_PATH)
    metadata = _json(METADATA_PATH)
    spatial_indexes = np.flatnonzero(graph["edge_relation"] == 5)
    line_involving_spatial = np.isin(graph["line_edge_source"], spatial_indexes) | np.isin(
        graph["line_edge_destination"], spatial_indexes
    )
    angle_bins = sorted(
        int(value) for value in np.unique(graph["line_angle_relation"][line_involving_spatial])
    )

    assert int(line_involving_spatial.sum()) == 285
    assert angle_bins == [0, 1, 2, 3, 4, 7]
    assert metadata["observations"]["line_graph_spatial_edge_count"] == 285
    assert metadata["observations"]["line_graph_spatial_angle_bins"] == angle_bins


def test_max_neighbor_probe_freezes_reference_truncation_order():
    probe = _npz(PROBE_PATH)
    metadata = _json(PROBE_METADATA_PATH)
    target_zero_sources = probe["spatial_edge_source"][probe["spatial_edge_destination"] == 0]

    assert probe["node_position"].shape == (40, 3)
    assert target_zero_sources.shape == (32,)
    assert metadata["candidate_spatial_neighbors_before_truncation"] == 35
    assert metadata["max_num_neighbors"] == 32
    assert metadata["retained_spatial_neighbors"] == 32
    np.testing.assert_array_equal(
        target_zero_sources,
        np.asarray(metadata["retained_sources_in_reference_order"], dtype=np.int64),
    )


def test_supplemental_metadata_identifies_input_and_leaks_no_paths():
    metadata = _json(METADATA_PATH)

    assert metadata["fixture_schema_version"] == 1
    assert metadata["purpose"] == "frozen spatial reference"
    assert metadata["reference_provenance"] == (
        "Frozen upstream GearNet/TorchDrug graph reference retained as immutable evidence "
        "for native Eris regression."
    )
    assert metadata["input"] == {
        "chain_id": "A",
        "filename": INPUT_PATH.name,
        "model_index": 0,
        "sha256": _sha256(INPUT_PATH),
    }
    assert metadata["graph_parameters"] == {
        "edge_feature_width": 59,
        "num_angle_bin": 8,
        "num_relation": 7,
        "spatial_max_num_neighbors": 32,
        "spatial_min_sequence_distance": 5,
        "spatial_radius": 10.0,
    }
    serialized = json.dumps(metadata, sort_keys=True)
    assert "\\" not in serialized
    assert "/tmp/" not in serialized
    assert "eris-gearnet-" not in serialized
