from __future__ import annotations

import copy
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest

import eris
from eris.core.artifacts import ArtifactValidationError
from eris.export.torch import save_artifact_to_torch, to_torch
from tests.helpers import PDB, learned_artifact

requires_torch = pytest.mark.skipif(
    find_spec("torch") is None,
    reason="requires optional dependency 'torch'",
)


def _torch():
    return pytest.importorskip("torch")


def _assert_tensor_equal(actual, expected: np.ndarray) -> None:
    torch = _torch()
    assert isinstance(actual, torch.Tensor)
    assert str(actual.dtype).removeprefix("torch.") == str(expected.dtype)
    np.testing.assert_array_equal(actual.cpu().numpy(), expected)


@requires_torch
def test_torch_export_preserves_metadata_and_does_not_mutate_artifact() -> None:
    artifact = eris.encode(PDB, representation="residue_knn_graph", use_cache=False)
    data_before = {name: array.copy() for name, array in artifact.data.items()}
    meta_before = copy.deepcopy(artifact.meta)

    out = to_torch(artifact)
    out["x"][0, 0] = out["x"][0, 0] + 100
    out["feature_keys"]["node_attr_keys"].append("mutated")

    assert out["representation_name"] == "residue_knn_graph"
    assert out["representation_family"] == "explicit"
    assert out["representation_type"] == "graph"
    assert out["granularity"] == "residue"
    assert out["parameters"] == meta_before["parameters"]
    assert out["outputs"] == meta_before["outputs"]
    for name, expected in data_before.items():
        np.testing.assert_array_equal(artifact.data[name], expected)
    assert artifact.meta == meta_before


@requires_torch
def test_graph_export_maps_node_attr_to_x_without_reordering_or_padding() -> None:
    artifact = eris.encode(PDB, representation="residue_contact_graph", use_cache=False)

    out = to_torch(artifact)

    _assert_tensor_equal(out["x"], artifact.data["node_attr"])
    _assert_tensor_equal(out["edge_index"], artifact.data["edge_index"])
    _assert_tensor_equal(out["edge_attr"], artifact.data["edge_attr"])
    _assert_tensor_equal(out["edge_dist"], artifact.data["edge_dist"])
    _assert_tensor_equal(out["pos"], artifact.data["pos"])
    _assert_tensor_equal(out["graph_attr"], artifact.data["graph_attr"])
    assert "node_attr" not in out
    assert out["edge_index"].shape[1] == out["edge_attr"].shape[0]
    assert out["edge_index"].shape[1] == out["edge_dist"].shape[0]


@requires_torch
def test_cloud_export_maps_positions_and_features_and_preserves_auxiliary_ids() -> None:
    artifact = eris.encode(PDB, representation="residue_point_cloud", use_cache=False)

    out = to_torch(artifact)

    _assert_tensor_equal(out["pos"], artifact.data["pos"])
    _assert_tensor_equal(out["x"], artifact.data["node_attr"])
    _assert_tensor_equal(out["centroid"], artifact.data["centroid"])
    assert out["extra"]["residue_ids"] == artifact.data["residue_ids"].tolist()
    assert out["extra"]["chain_id"] == artifact.data["chain_id"].tolist()
    _assert_tensor_equal(out["extra"]["aa_index"], artifact.data["aa_index"])


@requires_torch
@pytest.mark.parametrize(
    ("representation", "matrix_key"),
    [("residue_distogram", "dist"), ("residue_contact_map_bins", "contact_bins")],
)
def test_matrix_export_preserves_main_matrix_and_channel_option(
    representation: str,
    matrix_key: str,
) -> None:
    artifact = eris.encode(PDB, representation=representation, use_cache=False)

    default = to_torch(artifact)
    native = to_torch(artifact, matrix_as_channel=False)

    assert default["x"].shape == (1, *artifact.data[matrix_key].shape)
    _assert_tensor_equal(default["x"][0], artifact.data[matrix_key])
    _assert_tensor_equal(native["x"], artifact.data[matrix_key])
    assert default["matrix_key"] == native["matrix_key"] == matrix_key
    assert default["n"] == artifact.data[matrix_key].shape[0]
    if "bin_edges" in artifact.data:
        _assert_tensor_equal(default["bin_edges"], artifact.data["bin_edges"])


@requires_torch
def test_grid_export_preserves_grid_shape_and_spatial_metadata() -> None:
    artifact = eris.encode(PDB, representation="residue_voxel_grid", use_cache=False)

    out = to_torch(artifact)

    _assert_tensor_equal(out["x"], artifact.data["grid"])
    _assert_tensor_equal(out["grid_size"], artifact.data["grid_size"])
    _assert_tensor_equal(out["voxel_size"], artifact.data["voxel_size"])
    _assert_tensor_equal(out["origin"], artifact.data["origin"])
    _assert_tensor_equal(out["centroid"], artifact.data["centroid"])


@requires_torch
def test_pooled_vector_export_preserves_pooled_features() -> None:
    source = eris.encode(PDB, representation="residue_point_cloud", use_cache=False)
    artifact = eris.pool(source, mode="mean_std")

    out = to_torch(artifact)

    _assert_tensor_equal(out["pooled_attr"], artifact.data["pooled_attr"])
    assert out["representation_type"] == "vector"
    assert out["feature_keys"] == artifact.meta["feature_keys"]


def test_save_torch_strictly_loads_artifact_dir_and_writes_downstream_pt(tmp_path: Path) -> None:
    torch = _torch()
    artifact_dir = tmp_path / "artifact"
    artifact = eris.encode(
        PDB,
        representation="residue_contact_map",
        output=artifact_dir,
        use_cache=False,
    )

    out_path = tmp_path / "export" / "data.pt"
    save_artifact_to_torch(artifact_dir, out_path=out_path)
    saved = torch.load(out_path, weights_only=False)
    assert saved["representation_name"] == "residue_contact_map"
    assert saved["x"].shape == (1, *artifact.data["contact"].shape)
    _assert_tensor_equal(saved["x"], artifact.data["contact"][None, ...])


def test_learned_node_export_preserves_node_ids_mask_and_dtype() -> None:
    torch = _torch()
    artifact = learned_artifact("node", dtype=np.float64, with_mask=True)

    out = to_torch(artifact)

    assert out["representation_family"] == "learned"
    assert out["representation_type"] == "embedding"
    assert out["node_embedding"].shape == (2, 3)
    assert out["node_embedding"].dtype == torch.float64
    assert out["node_ids"] == ["A:1::ALA", "A:2::GLY"]
    assert out["chain_id"] == ["A", "A"]
    assert torch.equal(out["valid_mask"], torch.tensor([True, False]))
    assert "edge_index" not in out
    assert "batch" not in out
    assert "ptr" not in out


def test_learned_global_export_preserves_single_artifact_shape() -> None:
    torch = _torch()
    artifact = learned_artifact("global", dtype=np.float64)

    out = to_torch(artifact)

    assert out["global_embedding"].shape == (3,)
    assert out["global_embedding"].dtype == torch.float64
    assert "node_embedding" not in out
    assert "node_ids" not in out


@requires_torch
def test_learned_both_export_preserves_all_outputs_and_collate_compatibility() -> None:
    from eris.datasets import collate_learned

    first = to_torch(learned_artifact("both", with_mask=True))
    second = to_torch(learned_artifact("both", with_mask=True))

    assert {
        "node_embedding",
        "global_embedding",
        "node_ids",
        "chain_id",
        "valid_mask",
    } <= first.keys()
    out = collate_learned([first, second])

    assert out["node_embedding"].shape == (4, 3)
    assert out["global_embedding"].shape == (2, 3)
    assert out["node_ids"] == ["A:1::ALA", "A:2::GLY"] * 2
    assert out["chain_id"] == ["A", "A", "A", "A"]


def test_learned_export_rejects_malformed_programmatic_artifacts() -> None:
    artifact = learned_artifact("node", dtype=np.float64, with_mask=True)
    artifact.data = dict(artifact.data)
    artifact.data["node_ids"] = np.asarray(["one"], dtype=np.str_)

    with pytest.raises(ArtifactValidationError, match="node_ids"):
        to_torch(artifact)
