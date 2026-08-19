from __future__ import annotations

import numpy as np
import pytest

from eris.core.base_encoder import EncodingContext
from eris.core.types import StructureInput
from eris.encoders.graphs.atom_knn_graph import AtomKNNGraph
from eris.encoders.graphs.residue_contact_graph import ResidueContactGraph
from eris.encoders.graphs.residue_knn_graph import ResidueKNNGraph
from eris.features.rbf import compute_rbf

RBF_K = 4
RBF_DMAX = 4.0
LOCAL_CUTOFF = 1


def _line_positions(*coordinates: float) -> np.ndarray:
    return np.asarray([[x, 0.0, 0.0] for x in coordinates], dtype=np.float32)


def _pairs(edge_index: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(edge) for edge in edge_index.T.tolist()]


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


def _atom_context(
    positions: np.ndarray,
    sequence_positions: np.ndarray,
) -> EncodingContext:
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
        "seq_pos_residue": np.asarray(sequence_positions, dtype=np.int32),
        "aa_index_residue": np.zeros((node_count,), dtype=np.int16),
    }
    return EncodingContext(
        inp=StructureInput(source="synthetic.pdb"),
        raw_structure=raw_structure,
        centroid=np.zeros((1, 3), dtype=np.float32),
        pos=positions,
    )


def _assert_reverse_rows_match(data: dict[str, np.ndarray]) -> None:
    edge_index = data["edge_index"]
    pair_to_row = {pair: row for row, pair in enumerate(_pairs(edge_index))}

    for (source, target), row in pair_to_row.items():
        if source == target:
            continue
        reverse_row = pair_to_row[(target, source)]
        assert data["edge_dist"][row] == data["edge_dist"][reverse_row]
        np.testing.assert_array_equal(
            data["edge_attr"][row],
            data["edge_attr"][reverse_row],
        )


def _assert_edge_feature_alignment(
    data: dict[str, np.ndarray],
    edge_attr_keys: list[str],
    positions: np.ndarray,
    sequence_positions: np.ndarray,
) -> None:
    edge_index = data["edge_index"]
    edge_dist = data["edge_dist"]
    edge_attr = data["edge_attr"]
    edge_count = edge_index.shape[1]

    assert edge_index.shape == (2, edge_count)
    assert edge_dist.shape == (edge_count,)
    assert edge_attr.shape[0] == edge_count
    assert len(edge_attr_keys) == edge_attr.shape[1]

    source = edge_index[0]
    target = edge_index[1]
    expected_dist = np.linalg.norm(
        positions[source] - positions[target],
        axis=1,
    ).astype(np.float32)
    np.testing.assert_allclose(edge_dist, expected_dist, rtol=1e-6, atol=1e-7)

    dist_column = edge_attr_keys.index("dist")
    np.testing.assert_array_equal(edge_attr[:, dist_column], edge_dist)

    rbf_columns = [edge_attr_keys.index(f"rbf.{index}") for index in range(RBF_K)]
    expected_rbf = compute_rbf(
        edge_dist,
        num_kernels=RBF_K,
        dmin=0.0,
        dmax=RBF_DMAX,
    )
    np.testing.assert_allclose(
        edge_attr[:, rbf_columns],
        expected_rbf,
        rtol=1e-6,
        atol=1e-7,
    )

    expected_seq = np.abs(
        sequence_positions[source] - sequence_positions[target]
    ).astype(np.float32)
    seq_column = edge_attr_keys.index("seq_sep")
    np.testing.assert_array_equal(edge_attr[:, seq_column], expected_seq)

    local_column = edge_attr_keys.index(f"flag.is_local_le_{LOCAL_CUTOFF}")
    expected_local = (expected_seq <= LOCAL_CUTOFF).astype(np.float32)
    np.testing.assert_array_equal(edge_attr[:, local_column], expected_local)

    _assert_reverse_rows_match(data)


def _residue_knn_data(
    positions: np.ndarray,
    *,
    include_self_loops: bool = False,
) -> tuple[dict[str, np.ndarray], list[str]]:
    context = _residue_context(positions)
    encoder = ResidueKNNGraph(
        k=1,
        cutoff=None,
        symmetric=True,
        include_self_loops=include_self_loops,
        rbf_k=RBF_K,
        rbf_dmax=RBF_DMAX,
        local_cutoff=LOCAL_CUTOFF,
    )
    data = encoder._create_data(context)
    return data, context.runtime["edge_attr_keys"]


def test_residue_knn_deduplicated_symmetric_rows_remain_aligned():
    positions = _line_positions(0.0, 1.0)
    data, edge_attr_keys = _residue_knn_data(positions)

    assert _pairs(data["edge_index"]) == [(0, 1), (1, 0)]
    _assert_edge_feature_alignment(
        data,
        edge_attr_keys,
        positions,
        np.arange(positions.shape[0], dtype=np.int64),
    )


def test_residue_knn_partial_closure_preserves_feature_row_order():
    positions = _line_positions(0.0, 1.0, 3.0)
    data, edge_attr_keys = _residue_knn_data(positions)

    assert _pairs(data["edge_index"]) == [
        (0, 1),
        (1, 0),
        (2, 1),
        (1, 2),
    ]
    _assert_edge_feature_alignment(
        data,
        edge_attr_keys,
        positions,
        np.arange(positions.shape[0], dtype=np.int64),
    )


def test_atom_knn_alignment_uses_residue_sequence_positions():
    positions = _line_positions(0.0, 0.5, 2.0, 4.0)
    sequence_positions = np.asarray([0, 0, 1, 2], dtype=np.int64)
    context = _atom_context(positions, sequence_positions)
    encoder = AtomKNNGraph(
        k=1,
        cutoff=None,
        symmetric=True,
        rbf_k=RBF_K,
        rbf_dmax=RBF_DMAX,
        local_cutoff=LOCAL_CUTOFF,
    )

    data = encoder._create_data(context)
    pairs = _pairs(data["edge_index"])

    assert len(pairs) == len(set(pairs))
    _assert_edge_feature_alignment(
        data,
        context.runtime["edge_attr_keys"],
        positions,
        sequence_positions,
    )

    same_residue_row = pairs.index((0, 1))
    seq_column = context.runtime["edge_attr_keys"].index("seq_sep")
    assert data["edge_attr"][same_residue_row, seq_column] == 0.0


@pytest.mark.parametrize("graph_kind", ["knn", "contact"])
def test_residue_self_loop_rows_have_zero_distance_features(graph_kind):
    positions = _line_positions(0.0, 3.0, 1.0)
    context = _residue_context(positions)

    if graph_kind == "knn":
        encoder = ResidueKNNGraph(
            k=1,
            cutoff=None,
            symmetric=True,
            include_self_loops=True,
            rbf_k=RBF_K,
            rbf_dmax=RBF_DMAX,
            local_cutoff=LOCAL_CUTOFF,
        )
    else:
        encoder = ResidueContactGraph(
            cutoff=2.1,
            include_self_loops=True,
            rbf_k=RBF_K,
            rbf_dmax=RBF_DMAX,
            local_cutoff=LOCAL_CUTOFF,
        )

    data = encoder._create_data(context)
    edge_attr_keys = context.runtime["edge_attr_keys"]
    sequence_positions = np.arange(positions.shape[0], dtype=np.int64)
    _assert_edge_feature_alignment(
        data,
        edge_attr_keys,
        positions,
        sequence_positions,
    )

    pairs = _pairs(data["edge_index"])
    loop_rows = [row for row, (source, target) in enumerate(pairs) if source == target]
    assert len(loop_rows) == positions.shape[0]

    dist_column = edge_attr_keys.index("dist")
    seq_column = edge_attr_keys.index("seq_sep")
    local_column = edge_attr_keys.index(f"flag.is_local_le_{LOCAL_CUTOFF}")
    rbf_columns = [edge_attr_keys.index(f"rbf.{index}") for index in range(RBF_K)]
    expected_loop_rbf = compute_rbf(
        np.asarray([0.0], dtype=np.float32),
        num_kernels=RBF_K,
        dmin=0.0,
        dmax=RBF_DMAX,
    )[0]

    for row in loop_rows:
        assert data["edge_dist"][row] == 0.0
        assert data["edge_attr"][row, dist_column] == 0.0
        np.testing.assert_allclose(
            data["edge_attr"][row, rbf_columns],
            expected_loop_rbf,
            rtol=1e-6,
            atol=1e-7,
        )
        assert data["edge_attr"][row, seq_column] == 0.0
        assert data["edge_attr"][row, local_column] == 1.0


def test_residue_contact_graph_alignment_is_deterministic():
    positions = _line_positions(0.0, 3.0, 1.0)
    sequence_positions = np.arange(positions.shape[0], dtype=np.int64)
    encoder = ResidueContactGraph(
        cutoff=2.1,
        rbf_k=RBF_K,
        rbf_dmax=RBF_DMAX,
        local_cutoff=LOCAL_CUTOFF,
    )

    first_context = _residue_context(positions)
    first = encoder._create_data(first_context)
    second_context = _residue_context(positions)
    second = encoder._create_data(second_context)

    _assert_edge_feature_alignment(
        first,
        first_context.runtime["edge_attr_keys"],
        positions,
        sequence_positions,
    )
    np.testing.assert_array_equal(first["edge_index"], second["edge_index"])
    np.testing.assert_array_equal(first["edge_dist"], second["edge_dist"])
    np.testing.assert_array_equal(first["edge_attr"], second["edge_attr"])
