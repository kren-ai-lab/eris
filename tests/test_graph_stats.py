from __future__ import annotations

import warnings

import numpy as np
import pytest

from eris.core.artifacts import StructureArtifact
from eris.core.base_encoder import EncodingContext
from eris.core.types import StructureInput
from eris.encoders.graphs.atom_knn_graph import AtomKNNGraph
from eris.encoders.graphs.residue_contact_graph import ResidueContactGraph
from eris.encoders.graphs.residue_knn_graph import ResidueKNNGraph
from eris.features.graph_stats import compute_graph_stats, pack_graph_stats

EXPECTED_GRAPH_ATTR_KEYS = [
    "degree_max",
    "degree_mean",
    "degree_min",
    "degree_std",
    "density",
    "edge_dist_max",
    "edge_dist_mean",
    "edge_dist_min",
    "edge_dist_std",
    "isolated_frac",
    "n_edges",
    "n_nodes",
    "n_self_loops",
]


def _compute_stats(
    pairs: list[tuple[int, int]],
    distances: list[float],
    n_nodes: int,
) -> dict[str, float]:
    if pairs:
        edge_index = np.asarray(pairs, dtype=np.int64).T
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    edge_dist = np.asarray(distances, dtype=np.float32)
    return compute_graph_stats(edge_index=edge_index, edge_dist=edge_dist, n_nodes=n_nodes)


def _line_positions(*coordinates: float) -> np.ndarray:
    return np.asarray([[x, 0.0, 0.0] for x in coordinates], dtype=np.float32)


def _residue_context(positions: np.ndarray) -> EncodingContext:
    n_nodes = positions.shape[0]
    return EncodingContext(
        inp=StructureInput(source="synthetic.pdb"),
        raw_structure={
            "aa_index": np.arange(n_nodes, dtype=np.int16),
            "plddt": np.full((n_nodes,), 90.0, dtype=np.float32),
            "bfactor_ca": np.arange(n_nodes, dtype=np.float32),
            "residue_ids": np.asarray(
                [f"A:{index + 1}" for index in range(n_nodes)],
                dtype=np.str_,
            ),
            "resname": np.asarray(["ALA"] * n_nodes, dtype=np.str_),
        },
        centroid=np.zeros((1, 3), dtype=np.float32),
        pos=positions,
    )


def _atom_context(positions: np.ndarray) -> EncodingContext:
    n_nodes = positions.shape[0]
    return EncodingContext(
        inp=StructureInput(source="synthetic.pdb"),
        raw_structure={
            "elem_index": np.zeros((n_nodes,), dtype=np.int16),
            "atom_role": np.zeros((n_nodes,), dtype=np.int8),
            "bfactor": np.arange(n_nodes, dtype=np.float32),
            "occ": np.ones((n_nodes,), dtype=np.float32),
            "residue_ids": np.asarray(
                [f"A:{index + 1}" for index in range(n_nodes)],
                dtype=np.str_,
            ),
            "resname": np.asarray(["ALA"] * n_nodes, dtype=np.str_),
            "atom_name": np.asarray(["CA"] * n_nodes, dtype=np.str_),
            "seq_pos_residue": np.arange(n_nodes, dtype=np.int32),
            "aa_index_residue": np.zeros((n_nodes,), dtype=np.int16),
        },
        centroid=np.zeros((1, 3), dtype=np.float32),
        pos=positions,
    )


def _unpack_graph_stats(data: dict[str, np.ndarray], ctx: EncodingContext) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in zip(
            ctx.runtime["graph_attr_keys"],
            data["graph_attr"],
            strict=True,
        )
    }


def _assert_encoder_statistics(
    data: dict[str, np.ndarray],
    ctx: EncodingContext,
    *,
    expected_self_loops: int,
) -> None:
    stats = _unpack_graph_stats(data, ctx)
    n_edges = data["edge_index"].shape[1]

    assert stats["n_nodes"] == 2.0
    assert stats["n_edges"] == float(n_edges)
    assert stats["n_self_loops"] == float(expected_self_loops)
    assert stats["degree_mean"] == 1.0
    assert stats["degree_std"] == 0.0
    assert stats["degree_min"] == 1.0
    assert stats["degree_max"] == 1.0
    assert stats["density"] == 1.0
    assert stats["isolated_frac"] == 0.0
    assert stats["edge_dist_mean"] == 1.0
    assert stats["edge_dist_std"] == 0.0
    assert stats["edge_dist_min"] == 1.0
    assert stats["edge_dist_max"] == 1.0

    artifact = StructureArtifact(
        data=data,
        meta={"feature_keys": {"graph_attr_keys": ctx.runtime["graph_attr_keys"]}},
    )
    artifact.finalize()
    assert artifact.meta["counts"]["n_edges"] == n_edges
    assert stats["n_edges"] == float(artifact.meta["counts"]["n_edges"])


def test_two_node_symmetric_pair_uses_unique_neighbor_degree():
    stats = _compute_stats([(0, 1), (1, 0)], [2.5, 2.5], n_nodes=2)

    assert stats["n_nodes"] == 2.0
    assert stats["n_edges"] == 2.0
    assert stats["n_self_loops"] == 0.0
    assert stats["degree_mean"] == 1.0
    assert stats["degree_std"] == 0.0
    assert stats["degree_min"] == 1.0
    assert stats["degree_max"] == 1.0
    assert stats["density"] == 1.0
    assert stats["isolated_frac"] == 0.0
    assert stats["edge_dist_min"] == 2.5


def test_three_node_symmetric_chain_uses_unique_neighbors():
    stats = _compute_stats(
        [(0, 1), (1, 0), (1, 2), (2, 1)],
        [1.0, 1.0, 2.0, 2.0],
        n_nodes=3,
    )

    assert stats["degree_mean"] == pytest.approx(4.0 / 3.0)
    assert stats["degree_std"] == pytest.approx(0.4714045)
    assert stats["degree_min"] == 1.0
    assert stats["degree_max"] == 2.0
    assert stats["density"] == pytest.approx(2.0 / 3.0)


def test_complete_symmetric_graph_density_is_one():
    stats = _compute_stats(
        [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)],
        [1.0, 1.0, 2.0, 2.0, 1.0, 1.0],
        n_nodes=3,
    )

    assert stats["n_edges"] == 6.0
    assert stats["n_self_loops"] == 0.0
    assert stats["degree_mean"] == 2.0
    assert stats["degree_std"] == 0.0
    assert stats["degree_min"] == 2.0
    assert stats["degree_max"] == 2.0
    assert stats["density"] == 1.0


def test_asymmetric_payload_uses_undirected_projection():
    stats = _compute_stats([(0, 1), (1, 2)], [5.0, 6.0], n_nodes=3)

    assert stats["degree_mean"] == pytest.approx(4.0 / 3.0)
    assert stats["degree_min"] == 1.0
    assert stats["degree_max"] == 2.0
    assert stats["density"] == pytest.approx(2.0 / 3.0)


def test_self_loops_only_change_physical_edge_and_loop_counts():
    non_loop_pairs = [(0, 1), (1, 0), (1, 2), (2, 1)]
    non_loop_distances = [1.0, 1.0, 2.0, 2.0]
    without_loops = _compute_stats(non_loop_pairs, non_loop_distances, n_nodes=3)
    with_loops = _compute_stats(
        non_loop_pairs + [(0, 0), (1, 1), (2, 2)],
        non_loop_distances + [0.0, 0.0, 0.0],
        n_nodes=3,
    )

    assert without_loops["n_edges"] == 4.0
    assert without_loops["n_self_loops"] == 0.0
    assert with_loops["n_edges"] == 7.0
    assert with_loops["n_self_loops"] == 3.0

    loop_invariant_keys = [
        "degree_mean",
        "degree_std",
        "degree_min",
        "degree_max",
        "density",
        "isolated_frac",
        "edge_dist_mean",
        "edge_dist_std",
        "edge_dist_min",
        "edge_dist_max",
    ]
    for key in loop_invariant_keys:
        assert with_loops[key] == without_loops[key]


def test_loop_only_nodes_remain_isolated():
    stats = _compute_stats([(0, 0)], [0.0], n_nodes=2)

    assert stats["n_edges"] == 1.0
    assert stats["n_self_loops"] == 1.0
    assert stats["degree_mean"] == 0.0
    assert stats["degree_min"] == 0.0
    assert stats["degree_max"] == 0.0
    assert stats["density"] == 0.0
    assert stats["isolated_frac"] == 1.0


def test_empty_non_loop_topology_is_finite_and_warning_free():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stats = _compute_stats([], [], n_nodes=3)

    assert stats["n_edges"] == 0.0
    assert stats["n_self_loops"] == 0.0
    assert stats["degree_mean"] == 0.0
    assert stats["degree_std"] == 0.0
    assert stats["degree_min"] == 0.0
    assert stats["degree_max"] == 0.0
    assert stats["density"] == 0.0
    assert stats["isolated_frac"] == 1.0
    for key in (
        "edge_dist_mean",
        "edge_dist_std",
        "edge_dist_min",
        "edge_dist_max",
    ):
        assert stats[key] == 0.0
        assert np.isfinite(stats[key])


def test_pack_graph_stats_preserves_lexicographic_order():
    stats = _compute_stats([(0, 1), (1, 0)], [1.0, 1.0], n_nodes=2)
    graph_attr, keys = pack_graph_stats(stats)

    assert keys == EXPECTED_GRAPH_ATTR_KEYS
    assert graph_attr.dtype == np.float32
    assert graph_attr.shape == (13,)
    np.testing.assert_array_equal(
        graph_attr,
        np.asarray([stats[key] for key in EXPECTED_GRAPH_ATTR_KEYS], dtype=np.float32),
    )


def test_residue_knn_stats_describe_final_topology_with_loops():
    positions = _line_positions(0.0, 1.0)
    ctx = _residue_context(positions)
    encoder = ResidueKNNGraph(
        use_cache=False,
        k=1,
        cutoff=None,
        symmetric=True,
        include_self_loops=True,
    )

    data = encoder._create_data(ctx)

    _assert_encoder_statistics(data, ctx, expected_self_loops=2)
    assert data["edge_attr"].shape[0] == data["edge_index"].shape[1]


def test_residue_contact_stats_describe_final_topology_with_loops():
    positions = _line_positions(0.0, 1.0)
    ctx = _residue_context(positions)
    encoder = ResidueContactGraph(
        use_cache=False,
        cutoff=2.0,
        include_self_loops=True,
    )

    data = encoder._create_data(ctx)

    _assert_encoder_statistics(data, ctx, expected_self_loops=2)
    assert data["edge_attr"].shape[0] == data["edge_index"].shape[1]


def test_atom_knn_stats_use_new_semantics_without_self_loops():
    positions = _line_positions(0.0, 1.0)
    ctx = _atom_context(positions)
    encoder = AtomKNNGraph(
        use_cache=False,
        k=1,
        cutoff=None,
        symmetric=True,
    )

    data = encoder._create_data(ctx)

    _assert_encoder_statistics(data, ctx, expected_self_loops=0)
    assert data["edge_attr"].shape[0] == data["edge_index"].shape[1]
