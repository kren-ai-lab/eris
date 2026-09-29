from __future__ import annotations

import copy
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest


requires_torch = pytest.mark.skipif(
    find_spec("torch") is None,
    reason="requires optional dependency 'torch'",
)

import eris
from eris.core.artifact_schema_v2 import build_input_metadata
from eris.core.artifacts import ArtifactValidationError, StructureArtifact
from eris.core.learned import (
    CheckpointDescriptor,
    StructuralEmbedderConfig,
    StructuralEmbeddingResult,
    StructuralModelDescriptor,
    build_learned_artifact,
)
from eris.export.torch import TorchExportError, save_artifact_to_torch, to_torch

PDB = Path("tests/data/1AQ2.pdb")


def _torch():
    return pytest.importorskip("torch")


def _assert_tensor_equal(actual, expected: np.ndarray) -> None:
    torch = _torch()
    assert isinstance(actual, torch.Tensor)
    assert str(actual.dtype).removeprefix("torch.") == str(expected.dtype)
    np.testing.assert_array_equal(actual.cpu().numpy(), expected)


def _input_metadata():
    return build_input_metadata(
        source="memory://torch-export",
        source_sha256="a" * 64,
        input_format="pdb",
        model_index=0,
        chain_id=None,
    )


def _model() -> StructuralModelDescriptor:
    return StructuralModelDescriptor(
        architecture="synthetic-structural-model",
        checkpoint=CheckpointDescriptor(
            identifier="synthetic-checkpoint",
            revision="revision-1",
            source="https://models.example/synthetic.bin",
            sha256="b" * 64,
        ),
    )


def _learned_artifact(
    output_mode: str,
    *,
    with_mask: bool = True,
    dtype=np.float64,
) -> StructureArtifact:
    node_embedding = np.arange(6, dtype=dtype).reshape(2, 3)
    global_embedding = np.asarray([7.0, 8.0, 9.0], dtype=dtype)
    node_ids = np.asarray(["A:1::ALA", "opaque-node-β"], dtype=np.str_)
    valid_mask = np.asarray([True, False], dtype=np.bool_)
    result = {
        "node": StructuralEmbeddingResult(
            node_embedding=node_embedding,
            node_ids=node_ids,
            valid_mask=valid_mask if with_mask else None,
        ),
        "global": StructuralEmbeddingResult(global_embedding=global_embedding),
        "both": StructuralEmbeddingResult(
            node_embedding=node_embedding,
            global_embedding=global_embedding,
            node_ids=node_ids,
            valid_mask=valid_mask if with_mask else None,
        ),
    }[output_mode]
    return build_learned_artifact(
        config=StructuralEmbedderConfig(
            representation_name="synthetic_structure_embedding",
            output_mode=output_mode,
        ),
        result=result,
        model=_model(),
        input_metadata=_input_metadata(),
        encoder_metadata={
            "implementation": "tests.fake_backend.FakeStructuralEncoder",
            "version": "test-adapter-v1",
        },
    )


def test_import_eris_export_does_not_eagerly_import_torch_or_backend_frameworks() -> None:
    script = """
import sys
import eris.export

blocked = {"torch", "torch_geometric", "torchdrug", "torch_scatter", "torch_cluster"}
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise SystemExit(f"optional frameworks loaded: {loaded}")
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_torch_export_rejects_schema_v1_boundary() -> None:
    artifact = StructureArtifact(
        data={"pos": np.zeros((1, 3), dtype=np.float32)},
        meta={
            "schema_version": 1,
            "encoder_name": "legacy",
            "encoder_version": "0",
            "encoder_fingerprint": "legacy",
            "type": "cloud",
            "level": "residue",
            "params": {},
            "input": {},
            "feature_keys": {},
        },
    )

    with pytest.raises(ArtifactValidationError, match="Schema v1"):
        to_torch(artifact)


@requires_torch
def test_torch_export_preserves_metadata_and_does_not_mutate_artifact() -> None:
    artifact = eris.encode(PDB, representation="residue_knn_graph", use_cache=False)
    data_before = {name: array.copy() for name, array in artifact.data.items()}
    meta_before = copy.deepcopy(artifact.meta)

    out = to_torch(artifact, prefer="graph")
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

    out = to_torch(artifact, prefer="graph")

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

    out = to_torch(artifact, prefer="cloud")

    _assert_tensor_equal(out["pos"], artifact.data["pos"])
    _assert_tensor_equal(out["x"], artifact.data["node_attr"])
    _assert_tensor_equal(out["centroid"], artifact.data["centroid"])
    assert out["extra"]["residue_ids"] == artifact.data["residue_ids"].tolist()
    _assert_tensor_equal(out["extra"]["aa_index"], artifact.data["aa_index"])


@requires_torch
@pytest.mark.parametrize(
    ("representation", "matrix_key"),
    [
        ("residue_distogram", "dist"),
        ("residue_distogram_bins", "dist_bins"),
        ("residue_contact_map", "contact"),
        ("residue_contact_map_bins", "contact_bins"),
    ],
)
def test_matrix_export_preserves_main_matrix_and_historical_channel_option(
    representation: str,
    matrix_key: str,
) -> None:
    artifact = eris.encode(PDB, representation=representation, use_cache=False)

    default = to_torch(artifact, prefer="matrix")
    native = to_torch(artifact, prefer="matrix", matrix_as_channel=False)

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

    out = to_torch(artifact, prefer="grid")

    _assert_tensor_equal(out["x"], artifact.data["grid"])
    _assert_tensor_equal(out["grid_size"], artifact.data["grid_size"])
    _assert_tensor_equal(out["voxel_size"], artifact.data["voxel_size"])
    _assert_tensor_equal(out["origin"], artifact.data["origin"])
    _assert_tensor_equal(out["centroid"], artifact.data["centroid"])


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
    save_artifact_to_torch(artifact_dir, out_path=out_path, prefer="matrix")
    saved = torch.load(out_path, weights_only=False)
    assert saved["representation_name"] == "residue_contact_map"
    assert saved["x"].shape == (1, *artifact.data["contact"].shape)
    _assert_tensor_equal(saved["x"], artifact.data["contact"][None, ...])


def test_learned_node_export_preserves_node_ids_mask_and_dtype() -> None:
    torch = _torch()
    artifact = _learned_artifact("node")

    out = to_torch(artifact)

    assert out["representation_family"] == "learned"
    assert out["representation_type"] == "embedding"
    assert out["node_embedding"].shape == (2, 3)
    assert out["node_embedding"].dtype == torch.float64
    assert out["node_ids"] == ["A:1::ALA", "opaque-node-β"]
    assert torch.equal(out["valid_mask"], torch.tensor([True, False]))
    assert "edge_index" not in out
    assert "batch" not in out
    assert "ptr" not in out


def test_learned_global_export_preserves_single_artifact_shape() -> None:
    torch = _torch()
    artifact = _learned_artifact("global")

    out = to_torch(artifact)

    assert out["global_embedding"].shape == (3,)
    assert out["global_embedding"].dtype == torch.float64
    assert "node_embedding" not in out
    assert "node_ids" not in out


@requires_torch
def test_learned_both_export_preserves_all_outputs_and_collate_compatibility() -> None:
    from eris.datasets import collate_learned

    first = to_torch(_learned_artifact("both"))
    second = to_torch(_learned_artifact("both"))

    assert {"node_embedding", "global_embedding", "node_ids", "valid_mask"} <= first.keys()
    out = collate_learned([first, second])

    assert out["node_embedding"].shape == (4, 3)
    assert out["global_embedding"].shape == (2, 3)
    assert out["node_ids"] == [
        "A:1::ALA",
        "opaque-node-β",
        "A:1::ALA",
        "opaque-node-β",
    ]


def test_learned_export_rejects_malformed_programmatic_artifacts() -> None:
    artifact = _learned_artifact("node")
    artifact.data = dict(artifact.data)
    artifact.data["node_ids"] = np.asarray(["one"], dtype=np.str_)

    with pytest.raises(ArtifactValidationError, match="node_ids"):
        to_torch(artifact)


def test_torch_export_rejects_unsupported_preference() -> None:
    artifact = eris.encode(PDB, representation="residue_point_cloud", use_cache=False)

    with pytest.raises(TorchExportError, match="Cannot export cloud"):
        to_torch(artifact, prefer="graph")
