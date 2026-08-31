from __future__ import annotations

import builtins
import copy
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

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
from eris.export.pyg import PyGExportError, load_pyg, save_artifact_to_pyg, to_pyg
from eris.export.torch import to_torch
from tests.graph_artifact_v2_fixture import save_graph_artifact_v2

PDB = Path("tests/data/1AQ2.pdb")
GRAPH_REPRESENTATIONS = (
    "atom_knn_graph",
    "residue_knn_graph",
    "residue_contact_graph",
)
NON_GRAPH_REPRESENTATIONS = (
    "atom_point_cloud",
    "residue_point_cloud",
    "residue_distogram",
    "residue_distogram_bins",
    "residue_contact_map",
    "residue_contact_map_bins",
    "residue_voxel_grid",
)


class FakePyGData:
    def to(self, device):
        torch = pytest.importorskip("torch")
        for key, value in list(self.__dict__.items()):
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(device))
        return self


def _torch_and_data_type(monkeypatch):
    torch = pytest.importorskip("torch")
    try:
        from torch_geometric.data import Data
    except ImportError:
        pyg_module = types.ModuleType("torch_geometric")
        data_module = types.ModuleType("torch_geometric.data")
        data_module.Data = FakePyGData
        pyg_module.data = data_module
        monkeypatch.setitem(sys.modules, "torch_geometric", pyg_module)
        monkeypatch.setitem(sys.modules, "torch_geometric.data", data_module)
        return torch, FakePyGData
    return torch, Data


def _assert_tensor_matches_array(tensor: Any, array: np.ndarray, monkeypatch) -> None:
    torch, _ = _torch_and_data_type(monkeypatch)
    assert isinstance(tensor, torch.Tensor)
    assert str(tensor.dtype) == f"torch.{array.dtype}"
    np.testing.assert_array_equal(tensor.cpu().numpy(), array)


def _input_metadata():
    return build_input_metadata(
        source="memory://pyg-export",
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


def _learned_artifact(output_mode: str) -> StructureArtifact:
    node_embedding = np.arange(6, dtype=np.float32).reshape(2, 3)
    global_embedding = np.asarray([7.0, 8.0, 9.0], dtype=np.float32)
    node_ids = np.asarray(["A:1::ALA", "A:2::GLY"], dtype=np.str_)
    valid_mask = np.asarray([True, False], dtype=np.bool_)
    result = {
        "node": StructuralEmbeddingResult(
            node_embedding=node_embedding,
            node_ids=node_ids,
            valid_mask=valid_mask,
        ),
        "global": StructuralEmbeddingResult(global_embedding=global_embedding),
        "both": StructuralEmbeddingResult(
            node_embedding=node_embedding,
            global_embedding=global_embedding,
            node_ids=node_ids,
            valid_mask=valid_mask,
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


def test_import_eris_export_does_not_eagerly_import_torch_or_pyg_or_gearnet() -> None:
    script = """
import sys
import eris
import eris.export

blocked = {"torch", "torch_geometric", "torchdrug", "torch_scatter", "torch_cluster"}
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise SystemExit(f"optional frameworks loaded: {loaded}")
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_pyg_export_reports_missing_optional_dependencies_actionably(monkeypatch) -> None:
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch_geometric" or name.startswith("torch_geometric."):
            raise ImportError("blocked torch_geometric")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(
        ImportError,
        match="requires optional dependencies 'torch' and 'torch_geometric'",
    ):
        to_pyg(eris.encode(PDB, representation="residue_contact_graph", use_cache=False))


def test_pyg_export_rejects_schema_v1_before_optional_imports() -> None:
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
        to_pyg(artifact)


@pytest.mark.parametrize("representation", GRAPH_REPRESENTATIONS)
def test_all_explicit_graph_representations_export_to_pyg_without_reordering(
    representation: str,
    monkeypatch,
) -> None:
    _, Data = _torch_and_data_type(monkeypatch)
    artifact = eris.encode(PDB, representation=representation, use_cache=False)

    graph = to_pyg(artifact)

    assert isinstance(graph, Data)
    assert graph.representation_name == representation
    assert graph.representation_family == "explicit"
    assert graph.representation_type == "graph"
    assert graph.granularity == artifact.meta["granularity"]
    assert graph.coordinate_source == artifact.meta["coordinate_source"]
    assert graph.schema_version == 2
    assert graph.feature_keys == artifact.meta["feature_keys"]
    assert graph.parameters == artifact.meta["parameters"]
    assert graph.outputs == artifact.meta["outputs"]
    _assert_tensor_matches_array(graph.edge_index, artifact.data["edge_index"], monkeypatch)
    _assert_tensor_matches_array(graph.x, artifact.data["node_attr"], monkeypatch)
    assert graph.edge_index.shape[1] == artifact.data["edge_index"].shape[1]

    for key in ("edge_attr", "edge_dist", "pos", "graph_attr", "centroid"):
        if key in artifact.data:
            _assert_tensor_matches_array(getattr(graph, key), artifact.data[key], monkeypatch)
    if "edge_attr" in artifact.data:
        assert graph.edge_attr.shape[0] == graph.edge_index.shape[1]
    if "edge_dist" in artifact.data:
        assert graph.edge_dist.shape[0] == graph.edge_index.shape[1]


def test_pyg_export_preserves_residue_node_ids_losslessly(monkeypatch) -> None:
    _torch_and_data_type(monkeypatch)
    artifact = eris.encode(PDB, representation="residue_knn_graph", use_cache=False)

    graph = to_pyg(artifact)

    assert graph.node_ids == artifact.data["residue_ids"].tolist()


def test_pyg_export_preserves_edge_feature_alignment_row_for_row(monkeypatch) -> None:
    _torch_and_data_type(monkeypatch)
    artifact = eris.encode(PDB, representation="residue_contact_graph", use_cache=False)

    graph = to_pyg(artifact)

    for edge_position in range(artifact.data["edge_index"].shape[1]):
        np.testing.assert_array_equal(
            graph.edge_index[:, edge_position].cpu().numpy(),
            artifact.data["edge_index"][:, edge_position],
        )
        np.testing.assert_array_equal(
            graph.edge_attr[edge_position].cpu().numpy(),
            artifact.data["edge_attr"][edge_position],
        )
        np.testing.assert_array_equal(
            graph.edge_dist[edge_position].cpu().numpy(),
            artifact.data["edge_dist"][edge_position],
        )


def test_pyg_export_preserves_graph_attr_from_strict_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _torch_and_data_type(monkeypatch)
    artifact_dir = tmp_path / "artifact"
    artifact = save_graph_artifact_v2(artifact_dir, graph_attr=(2.0, 2.0))

    graph = load_pyg(artifact_dir)

    _assert_tensor_matches_array(graph.x, artifact.data["node_attr"], monkeypatch)
    _assert_tensor_matches_array(graph.edge_index, artifact.data["edge_index"], monkeypatch)
    _assert_tensor_matches_array(graph.graph_attr, artifact.data["graph_attr"], monkeypatch)
    assert graph.graph_attr.ndim == 1


def test_pyg_export_does_not_mutate_artifact_data_or_metadata(monkeypatch) -> None:
    _torch_and_data_type(monkeypatch)
    artifact = eris.encode(PDB, representation="residue_contact_graph", use_cache=False)
    data_before = {name: array.copy() for name, array in artifact.data.items()}
    meta_before = copy.deepcopy(artifact.meta)

    graph = to_pyg(artifact)
    graph.x[0, 0] = graph.x[0, 0] + 100
    graph.edge_attr[0, 0] = graph.edge_attr[0, 0] + 100
    graph.feature_keys["node_attr_keys"].append("mutated")

    for name, expected in data_before.items():
        np.testing.assert_array_equal(artifact.data[name], expected)
    assert artifact.meta == meta_before


def test_load_and_save_pyg_use_strict_artifact_boundary(tmp_path: Path, monkeypatch) -> None:
    torch, Data = _torch_and_data_type(monkeypatch)
    artifact_dir = tmp_path / "artifact"
    artifact = eris.encode(
        PDB,
        representation="residue_contact_graph",
        output=artifact_dir,
        use_cache=False,
    )

    loaded = load_pyg(artifact_dir)
    _assert_tensor_matches_array(loaded.edge_index, artifact.data["edge_index"], monkeypatch)

    out_path = tmp_path / "exports" / "data_pyg.pt"
    save_artifact_to_pyg(artifact_dir, out_path=out_path)
    saved = torch.load(out_path, weights_only=False)
    assert isinstance(saved, Data)
    _assert_tensor_matches_array(saved.edge_index, artifact.data["edge_index"], monkeypatch)


@pytest.mark.parametrize("representation", NON_GRAPH_REPRESENTATIONS)
def test_non_graph_explicit_artifacts_are_not_converted_to_synthetic_pyg_graphs(
    representation: str,
) -> None:
    artifact = eris.encode(PDB, representation=representation, use_cache=False)

    with pytest.raises(PyGExportError, match="Unsupported representation_type"):
        to_pyg(artifact)


@pytest.mark.parametrize("output_mode", ("node", "global", "both"))
def test_learned_embedding_artifacts_do_not_leak_internal_graph_topology(
    output_mode: str,
) -> None:
    artifact = _learned_artifact(output_mode)

    with pytest.raises(PyGExportError, match="do not expose public graph topology"):
        to_pyg(artifact)


def test_torch_mapping_compatibility_helper_is_graph_only(monkeypatch) -> None:
    from eris.export.pyg import torch_dict_to_pyg

    graph_mapping = to_torch(
        eris.encode(PDB, representation="residue_contact_graph", use_cache=False),
        prefer="graph",
    )

    _torch_and_data_type(monkeypatch)
    graph = torch_dict_to_pyg(graph_mapping)
    _assert_tensor_matches_array(
        graph.edge_index,
        graph_mapping["edge_index"].cpu().numpy(),
        monkeypatch,
    )
    _assert_tensor_matches_array(graph.x, graph_mapping["x"].cpu().numpy(), monkeypatch)

    with pytest.raises(PyGExportError, match="Only graph Torch mappings"):
        torch_dict_to_pyg({"representation_family": "explicit", "representation_type": "matrix"})


def test_pyg_export_rejects_non_graph_preferences() -> None:
    artifact = eris.encode(PDB, representation="residue_contact_graph", use_cache=False)

    with pytest.raises(PyGExportError, match="Cannot export graph artifact with prefer='matrix'"):
        to_pyg(artifact, prefer="matrix")
