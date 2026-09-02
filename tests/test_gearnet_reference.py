"""Integrity checks for the frozen GearNet scientific reference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from eris.core.learned.backends.gearnet.metadata import GEARNET_EDGE_CHECKPOINT_SHA256


DATA_DIR = Path(__file__).with_name("data")
INPUT_PATH = DATA_DIR / "gearnet_reference.pdb"
GRAPH_PATH = DATA_DIR / "gearnet_reference_graph.npz"
EMBEDDING_PATH = DATA_DIR / "gearnet_reference_embeddings.npz"
METADATA_PATH = DATA_DIR / "gearnet_reference.json"


def _metadata() -> dict[str, object]:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name] for name in arrays.files}


def test_reference_metadata_is_path_safe_and_identifies_input_and_checkpoint():
    metadata = _metadata()

    assert metadata["fixture_schema_version"] == 1
    assert metadata["input"] == {
        "chain_id": "A",
        "filename": INPUT_PATH.name,
        "model_index": 0,
        "sha256": _sha256(INPUT_PATH),
    }
    assert metadata["checkpoint"] == {
        "filename": "mc_gearnet_edge.pth",
        "sha256": GEARNET_EDGE_CHECKPOINT_SHA256,
    }
    assert metadata["backend"]["execution"] == {
        "device": "cpu",
        "eval": True,
        "precision": "fp32",
    }
    assert metadata["determinism"] == {
        "same_process_repeated_inference": "exact_array_equal",
    }
    assert metadata["reference_provenance"] == (
        "Frozen upstream GearNet/TorchDrug CPU/fp32 reference retained as immutable evidence "
        "for native Eris regression."
    )
    serialized = json.dumps(metadata, sort_keys=True)
    assert "eris-gearnet-" not in serialized
    assert "\\" not in serialized
    assert "/tmp/" not in serialized
    assert ":\\" not in serialized


def test_reference_graph_preserves_node_edge_and_line_graph_alignment():
    graph = _load_npz(GRAPH_PATH)
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
    nodes = graph["node_ids"].shape[0]
    edges = graph["edge_source"].shape[0]
    line_edges = graph["line_edge_source"].shape[0]
    assert nodes == 12
    assert graph["node_ids"].dtype.kind == "U"
    assert graph["carrier"].dtype == np.dtype("int64")
    assert graph["carrier"].shape == (nodes,)
    assert graph["node_feature"].dtype == np.dtype("int64")
    assert graph["node_feature"].shape == (nodes, 21)
    assert graph["node_position"].dtype == np.dtype("float32")
    assert graph["node_position"].shape == (nodes, 3)
    assert np.isfinite(graph["node_position"]).all()
    for name in ("edge_source", "edge_destination", "edge_relation"):
        assert graph[name].dtype == np.dtype("int64")
        assert graph[name].shape == (edges,)
    assert graph["edge_feature"].dtype == np.dtype("float32")
    assert graph["edge_feature"].shape == (edges, 59)
    assert np.isfinite(graph["edge_feature"]).all()
    for name in ("line_edge_source", "line_edge_destination", "line_angle_relation"):
        assert graph[name].dtype == np.dtype("int64")
        assert graph[name].shape == (line_edges,)
    assert np.all((0 <= graph["edge_source"]) & (graph["edge_source"] < nodes))
    assert np.all((0 <= graph["edge_destination"]) & (graph["edge_destination"] < nodes))
    assert np.all((0 <= graph["edge_relation"]) & (graph["edge_relation"] < 7))
    assert np.all((0 <= graph["line_edge_source"]) & (graph["line_edge_source"] < edges))
    assert np.all((0 <= graph["line_edge_destination"]) & (graph["line_edge_destination"] < edges))
    assert np.all((0 <= graph["line_angle_relation"]) & (graph["line_angle_relation"] < 8))


def test_reference_embedding_is_finite_fp32_and_aligned_to_graph_node_ids():
    graph = _load_npz(GRAPH_PATH)
    embeddings = _load_npz(EMBEDDING_PATH)

    assert set(embeddings) == {"global_embedding", "node_embedding", "node_ids"}
    assert embeddings["node_ids"].dtype.kind == "U"
    np.testing.assert_array_equal(embeddings["node_ids"], graph["node_ids"])
    assert embeddings["node_embedding"].dtype == np.dtype("float32")
    assert embeddings["node_embedding"].shape == (graph["node_ids"].shape[0], 3072)
    assert embeddings["global_embedding"].dtype == np.dtype("float32")
    assert embeddings["global_embedding"].shape == (3072,)
    assert embeddings["node_embedding"].flags.c_contiguous
    assert embeddings["global_embedding"].flags.c_contiguous
    assert np.isfinite(embeddings["node_embedding"]).all()
    assert np.isfinite(embeddings["global_embedding"]).all()


def test_primary_reference_includes_distance_ties_for_ordering_parity():
    metadata = _metadata()
    graph = _load_npz(GRAPH_PATH)
    positions = graph["node_position"]
    distances = np.asarray(
        [
            np.linalg.norm(positions[left] - positions[right])
            for left in range(positions.shape[0])
            for right in range(left + 1, positions.shape[0])
        ],
        dtype=np.float64,
    )
    rounded = np.round(distances, decimals=metadata["tie_behavior"]["tie_comparison_decimals"])

    assert metadata["tie_behavior"]["primary_fixture_contains_distance_ties"] is True
    assert np.unique(rounded).size < rounded.size
    assert metadata["tie_behavior"]["ordering_authority"] == (
        "frozen upstream graph and line-graph array order"
    )
