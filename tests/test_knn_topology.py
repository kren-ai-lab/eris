from __future__ import annotations

import numpy as np
import pytest

from eris.core.base_encoder import EncodingContext
from eris.core.types import StructureInput
from eris.encoders.graphs.atom_knn_graph import AtomKNNGraph
from eris.encoders.graphs.residue_knn_graph import ResidueKNNGraph
from eris.spatial.knn import _knn_edges_from_candidates, knn_edges


def _line_positions(*coordinates: float) -> np.ndarray:
    return np.asarray([[x, 0.0, 0.0] for x in coordinates], dtype=np.float32)


def _pairs(edge_index: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(edge) for edge in edge_index.T.tolist()]


def _assert_aligned(edge_index: np.ndarray, edge_dist: np.ndarray) -> None:
    edge_count = edge_index.shape[1]
    assert edge_index.shape == (2, edge_count)
    assert edge_index.dtype == np.int64
    assert edge_dist.shape == (edge_count,)
    assert edge_dist.dtype == np.float32


def _assert_unique_symmetric_without_loops(edge_index: np.ndarray) -> None:
    pairs = _pairs(edge_index)
    pair_set = set(pairs)

    assert len(pairs) == len(pair_set)
    assert all(source != target for source, target in pairs)
    assert all((target, source) in pair_set for source, target in pairs)


@pytest.mark.parametrize(
    ("positions", "k", "expected_pairs", "expected_distances"),
    [
        (
            _line_positions(0.0, 1.0),
            1,
            [(0, 1), (1, 0)],
            [1.0, 1.0],
        ),
        (
            _line_positions(0.0, 1.0, 3.0),
            1,
            [(0, 1), (1, 0), (2, 1), (1, 2)],
            [1.0, 1.0, 2.0, 2.0],
        ),
        (
            _line_positions(0.0, 1.0, 3.0),
            2,
            [(0, 1), (0, 2), (1, 0), (1, 2), (2, 1), (2, 0)],
            [1.0, 3.0, 1.0, 2.0, 2.0, 3.0],
        ),
    ],
)
def test_symmetric_knn_is_a_unique_preserve_first_closure(
    positions,
    k,
    expected_pairs,
    expected_distances,
):
    edge_index, edge_dist = knn_edges(
        positions,
        k,
        cutoff=None,
        symmetric=True,
    )

    assert _pairs(edge_index) == expected_pairs
    np.testing.assert_array_equal(
        edge_dist,
        np.asarray(expected_distances, dtype=np.float32),
    )
    _assert_aligned(edge_index, edge_dist)
    _assert_unique_symmetric_without_loops(edge_index)


def test_symmetric_deduplication_keeps_the_first_distance():
    candidates = [
        [(1, 1.0)],
        [(0, 1.25), (2, 2.0)],
        [],
    ]

    edge_index, edge_dist = _knn_edges_from_candidates(
        candidates,
        k=2,
        symmetric=True,
    )

    assert _pairs(edge_index) == [(0, 1), (1, 0), (1, 2), (2, 1)]
    np.testing.assert_array_equal(
        edge_dist,
        np.asarray([1.0, 1.25, 2.0, 2.0], dtype=np.float32),
    )
    _assert_aligned(edge_index, edge_dist)


def test_asymmetric_knn_preserves_the_directed_neighbor_selection():
    positions = _line_positions(0.0, 1.0, 3.0)

    edge_index, edge_dist = knn_edges(
        positions,
        1,
        cutoff=None,
        symmetric=False,
    )

    assert _pairs(edge_index) == [(0, 1), (1, 0), (2, 1)]
    np.testing.assert_array_equal(
        edge_dist,
        np.asarray([1.0, 1.0, 2.0], dtype=np.float32),
    )
    assert (1, 2) not in _pairs(edge_index)
    _assert_aligned(edge_index, edge_dist)


def test_equal_distances_use_node_index_as_tie_breaker_deterministically():
    positions = _line_positions(0.0, -1.0, 1.0)

    first_index, first_dist = knn_edges(
        positions,
        1,
        cutoff=2.0,
        symmetric=False,
    )
    second_index, second_dist = knn_edges(
        positions,
        1,
        cutoff=2.0,
        symmetric=False,
    )

    assert _pairs(first_index) == [(0, 1), (1, 0), (2, 0)]
    np.testing.assert_array_equal(first_index, second_index)
    np.testing.assert_array_equal(first_dist, second_dist)
    _assert_aligned(first_index, first_dist)


def test_n_smaller_than_k_uses_only_available_unique_candidates():
    edge_index, edge_dist = knn_edges(
        _line_positions(0.0, 1.0),
        16,
        cutoff=None,
        symmetric=True,
    )

    assert _pairs(edge_index) == [(0, 1), (1, 0)]
    _assert_aligned(edge_index, edge_dist)
    _assert_unique_symmetric_without_loops(edge_index)


@pytest.mark.parametrize("node_count", [0, 1])
def test_empty_edge_outputs_remain_aligned(node_count):
    positions = np.zeros((node_count, 3), dtype=np.float32)

    edge_index, edge_dist = knn_edges(
        positions,
        16,
        cutoff=None,
        symmetric=True,
    )

    _assert_aligned(edge_index, edge_dist)
    assert edge_index.shape == (2, 0)


def _residue_context(positions: np.ndarray) -> EncodingContext:
    node_count = positions.shape[0]
    raw_structure = {
        "aa_index": np.arange(node_count, dtype=np.int16),
        "plddt": np.full((node_count,), 90.0, dtype=np.float32),
        "bfactor_ca": np.arange(node_count, dtype=np.float32),
        "residue_ids": np.asarray(
            [f"A:{index + 1}" for index in range(node_count)],
            dtype=np.str_,
        ),
        "resname": np.asarray(["ALA"] * node_count, dtype=np.str_),
    }
    return EncodingContext(
        inp=StructureInput(source="synthetic.pdb"),
        raw_structure=raw_structure,
        centroid=np.zeros((1, 3), dtype=np.float32),
        pos=positions,
    )


def _atom_context(positions: np.ndarray) -> EncodingContext:
    node_count = positions.shape[0]
    raw_structure = {
        "elem_index": np.zeros((node_count,), dtype=np.int16),
        "atom_role": np.zeros((node_count,), dtype=np.int8),
        "bfactor": np.arange(node_count, dtype=np.float32),
        "occ": np.ones((node_count,), dtype=np.float32),
        "residue_ids": np.asarray(
            [f"A:{index + 1}" for index in range(node_count)],
            dtype=np.str_,
        ),
        "resname": np.asarray(["ALA"] * node_count, dtype=np.str_),
        "atom_name": np.asarray(["CA"] * node_count, dtype=np.str_),
        "seq_pos_residue": np.arange(node_count, dtype=np.int32),
        "aa_index_residue": np.zeros((node_count,), dtype=np.int16),
    }
    return EncodingContext(
        inp=StructureInput(source="synthetic.pdb"),
        raw_structure=raw_structure,
        centroid=np.zeros((1, 3), dtype=np.float32),
        pos=positions,
    )


@pytest.mark.parametrize("granularity", ["residue", "atom"])
def test_knn_encoders_receive_unique_symmetric_topology(granularity):
    positions = _line_positions(0.0, 1.0, 3.0)

    if granularity == "residue":
        encoder = ResidueKNNGraph(k=1, cutoff=None, symmetric=True)
        data = encoder._create_data(_residue_context(positions))
    else:
        encoder = AtomKNNGraph(k=1, cutoff=None, symmetric=True)
        data = encoder._create_data(_atom_context(positions))

    _assert_unique_symmetric_without_loops(data["edge_index"])
    _assert_aligned(data["edge_index"], data["edge_dist"])
    assert data["edge_attr"].shape[0] == data["edge_index"].shape[1]


def test_residue_encoder_adds_one_self_loop_per_node_after_knn_closure():
    positions = _line_positions(0.0, 1.0, 3.0)
    encoder = ResidueKNNGraph(
        k=1,
        cutoff=None,
        symmetric=True,
        include_self_loops=True,
    )

    data = encoder._create_data(_residue_context(positions))
    pairs = _pairs(data["edge_index"])
    loops = [edge for edge in pairs if edge[0] == edge[1]]
    non_loops = [edge for edge in pairs if edge[0] != edge[1]]

    assert loops == [(0, 0), (1, 1), (2, 2)]
    assert len(non_loops) == len(set(non_loops))
    assert all((target, source) in non_loops for source, target in non_loops)
    _assert_aligned(data["edge_index"], data["edge_dist"])
    assert data["edge_attr"].shape[0] == data["edge_index"].shape[1]
