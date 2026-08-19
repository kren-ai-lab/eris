from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eris.core.io import StructureLoadError
from eris.core.types import StructureInput
from eris.encoders.clouds.atom_point_cloud import AtomPointCloud
from eris.encoders.clouds.residue_point_cloud import ResiduePointCloudCA
from eris.encoders.graphs.atom_knn_graph import AtomKNNGraph
from eris.encoders.graphs.residue_contact_graph import ResidueContactGraph
from eris.encoders.graphs.residue_knn_graph import ResidueKNNGraph


def _pdb_atom_line(
    serial: int,
    atom_name: str,
    residue_name: str,
    residue_number: int,
    x: float,
    *,
    record: str = "ATOM",
    element: str = "C",
) -> str:
    return (
        f"{record:<6}{serial:5d} {atom_name:^4s} {residue_name:>3s} A{residue_number:4d}"
        f"    {x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{10.0:6.2f}"
        f"          {element:>2s}\n"
    )


def _write_structure(tmp_path: Path, lines: list[str], name: str = "structure.pdb") -> Path:
    path = tmp_path / name
    path.write_text("".join([*lines, "TER\n", "END\n"]), encoding="ascii")
    return path


def _write_ca_structure(tmp_path: Path, coordinates: list[float]) -> Path:
    lines = [
        _pdb_atom_line(index, "CA", "ALA", index, coordinate)
        for index, coordinate in enumerate(coordinates, start=1)
    ]
    return _write_structure(tmp_path, lines)


def _encode_valid(encoder, path: Path):
    artifact = encoder.encode(StructureInput(source=str(path), chain_id="A"))
    artifact.finalize()
    artifact.validate(strict=True)
    return artifact


def _graph_stats(artifact) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in zip(
            artifact.meta["feature_keys"]["graph_attr_keys"],
            artifact.data["graph_attr"],
            strict=True,
        )
    }


def _edge_pairs(artifact) -> list[tuple[int, int]]:
    return [tuple(pair) for pair in artifact.data["edge_index"].T.tolist()]


def _assert_edge_arrays_aligned(artifact) -> None:
    edge_count = artifact.data["edge_index"].shape[1]
    edge_keys = artifact.meta["feature_keys"]["edge_attr_keys"]

    assert artifact.data["edge_dist"].shape == (edge_count,)
    assert artifact.data["edge_attr"].shape[0] == edge_count
    assert artifact.data["edge_attr"].shape[1] == len(edge_keys)
    dist_column = edge_keys.index("dist")
    np.testing.assert_array_equal(
        artifact.data["edge_attr"][:, dist_column],
        artifact.data["edge_dist"],
    )


def _assert_zero_edge_artifact(artifact, *, n_nodes: int) -> None:
    edge_index = artifact.data["edge_index"]
    edge_dist = artifact.data["edge_dist"]
    edge_attr = artifact.data["edge_attr"]

    assert edge_index.shape == (2, 0)
    assert edge_index.dtype == np.int64
    assert edge_dist.shape == (0,)
    assert edge_dist.dtype == np.float32
    assert edge_attr.shape[0] == 0
    assert edge_attr.dtype == np.float32
    _assert_edge_arrays_aligned(artifact)
    assert artifact.data["graph_attr"].ndim == 1
    assert np.isfinite(artifact.data["graph_attr"]).all()
    assert artifact.meta["counts"]["n_nodes"] == n_nodes
    assert artifact.meta["counts"]["n_edges"] == 0


@pytest.mark.parametrize(
    ("encoder_type", "expected_feature_width"),
    [
        (AtomPointCloud, 25),
        (ResiduePointCloudCA, 38),
    ],
)
def test_single_node_point_cloud_is_strict_valid(
    tmp_path,
    encoder_type,
    expected_feature_width,
):
    path = _write_ca_structure(tmp_path, [0.0])

    artifact = _encode_valid(encoder_type(use_cache=False), path)

    assert artifact.data["pos"].shape == (1, 3)
    assert artifact.data["node_attr"].shape == (1, expected_feature_width)
    assert artifact.meta["counts"]["n_nodes"] == 1


@pytest.mark.parametrize(
    ("encoder_type", "kwargs"),
    [
        (AtomKNNGraph, {"cutoff": None}),
        (ResidueKNNGraph, {"cutoff": None, "include_self_loops": False}),
        (ResidueContactGraph, {"include_self_loops": False}),
    ],
)
def test_single_node_graph_without_loops_is_strict_valid(tmp_path, encoder_type, kwargs):
    path = _write_ca_structure(tmp_path, [0.0])

    artifact = _encode_valid(encoder_type(use_cache=False, **kwargs), path)

    _assert_zero_edge_artifact(artifact, n_nodes=1)
    stats = _graph_stats(artifact)
    assert stats["n_nodes"] == 1.0
    assert stats["n_edges"] == 0.0
    assert stats["n_self_loops"] == 0.0


@pytest.mark.parametrize(
    "encoder_type",
    [ResidueKNNGraph, ResidueContactGraph],
)
def test_single_node_residue_graph_with_self_loop_is_strict_valid(tmp_path, encoder_type):
    path = _write_ca_structure(tmp_path, [0.0])
    kwargs = {"cutoff": None} if encoder_type is ResidueKNNGraph else {}
    encoder = encoder_type(use_cache=False, include_self_loops=True, **kwargs)

    artifact = _encode_valid(encoder, path)

    np.testing.assert_array_equal(
        artifact.data["edge_index"],
        np.asarray([[0], [0]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        artifact.data["edge_dist"],
        np.asarray([0.0], dtype=np.float32),
    )
    assert artifact.data["edge_attr"].shape[0] == 1
    assert artifact.meta["counts"]["n_nodes"] == 1
    assert artifact.meta["counts"]["n_edges"] == 1

    stats = _graph_stats(artifact)
    expected_zero_stats = [
        "degree_mean",
        "degree_std",
        "degree_min",
        "degree_max",
        "density",
        "edge_dist_mean",
        "edge_dist_std",
        "edge_dist_min",
        "edge_dist_max",
    ]
    assert stats["n_edges"] == 1.0
    assert stats["n_self_loops"] == 1.0
    assert stats["isolated_frac"] == 1.0
    assert all(stats[key] == 0.0 for key in expected_zero_stats)


@pytest.mark.parametrize(
    ("coordinates", "expected_pairs"),
    [
        ([0.0, 1.0], [(0, 1), (1, 0)]),
        (
            [0.0, 1.0, 3.0],
            [(0, 1), (0, 2), (1, 0), (1, 2), (2, 1), (2, 0)],
        ),
    ],
)
def test_residue_knn_uses_all_available_neighbors_when_n_is_smaller_than_k(
    tmp_path,
    coordinates,
    expected_pairs,
):
    path = _write_ca_structure(tmp_path, coordinates)
    encoder = ResidueKNNGraph(use_cache=False, k=16, cutoff=None, symmetric=True)

    artifact = _encode_valid(encoder, path)

    pairs = _edge_pairs(artifact)
    assert encoder.k == 16
    assert pairs == expected_pairs
    assert len(pairs) == len(set(pairs))
    assert all((target, source) in pairs for source, target in pairs)
    _assert_edge_arrays_aligned(artifact)


def test_atom_knn_uses_all_available_neighbors_when_n_is_smaller_than_k(tmp_path):
    path = _write_ca_structure(tmp_path, [0.0, 1.0])
    encoder = AtomKNNGraph(use_cache=False, k=16, cutoff=None, symmetric=True)

    artifact = _encode_valid(encoder, path)

    assert encoder.k == 16
    assert _edge_pairs(artifact) == [(0, 1), (1, 0)]
    _assert_edge_arrays_aligned(artifact)


def test_local_knn_with_no_candidates_is_strict_valid(tmp_path):
    path = _write_ca_structure(tmp_path, [0.0, 100.0])
    encoder = ResidueKNNGraph(use_cache=False, k=16, cutoff=8.0, symmetric=True)

    artifact = _encode_valid(encoder, path)

    _assert_zero_edge_artifact(artifact, n_nodes=2)


def test_contact_graph_with_no_contacts_is_strict_valid(tmp_path):
    path = _write_ca_structure(tmp_path, [0.0, 100.0, 200.0])
    encoder = ResidueContactGraph(use_cache=False, cutoff=8.0)

    artifact = _encode_valid(encoder, path)

    _assert_zero_edge_artifact(artifact, n_nodes=3)
    stats = _graph_stats(artifact)
    expected_zero_stats = [
        "n_edges",
        "n_self_loops",
        "degree_mean",
        "degree_std",
        "degree_min",
        "degree_max",
        "density",
        "edge_dist_mean",
        "edge_dist_std",
        "edge_dist_min",
        "edge_dist_max",
    ]
    assert stats["isolated_frac"] == 1.0
    assert all(stats[key] == 0.0 for key in expected_zero_stats)


@pytest.mark.parametrize("encoder_type", [AtomPointCloud, AtomKNNGraph])
def test_empty_atom_selection_raises_domain_error(tmp_path, encoder_type):
    path = _write_structure(
        tmp_path,
        [_pdb_atom_line(1, "C1", "LIG", 1, 0.0, record="HETATM")],
        name="hetero_only.pdb",
    )
    encoder = encoder_type(use_cache=False, include_hetero=False)

    with pytest.raises(StructureLoadError, match=r"No atoms found.*filters/policies"):
        encoder.encode(StructureInput(source=str(path), chain_id="A"))


def test_empty_residue_selection_raises_domain_error(tmp_path):
    path = _write_structure(
        tmp_path,
        [_pdb_atom_line(1, "N", "ALA", 1, 0.0, element="N")],
        name="no_ca.pdb",
    )
    encoder = ResiduePointCloudCA(use_cache=False)

    with pytest.raises(StructureLoadError, match=r"No residue positions found.*filters/policies"):
        encoder.encode(StructureInput(source=str(path), chain_id="A"))
