from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from typer.testing import CliRunner

import eris
from eris.cli.main import app
from eris.core.artifact_schema import build_input_metadata
from eris.core.artifacts import StructureArtifact
from eris.core.hashing import sha256_file
from eris.core.learned import (
    CheckpointDescriptor,
    StructuralEmbedderConfig,
    StructuralEmbeddingResult,
    StructuralModelDescriptor,
    build_learned_artifact,
)
from eris.datasets import ErisFolderDataset

PDB = Path("tests/data/1AQ2.pdb")
runner = CliRunner()


class _FakePyGData:
    def to(self, device: str):
        import torch

        for key, value in list(self.__dict__.items()):
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(device))
        return self


def _install_fake_pyg(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    try:
        import torch_geometric.data  # noqa: F401
    except ImportError:
        pyg_module = cast("Any", types.ModuleType("torch_geometric"))
        data_module = cast("Any", types.ModuleType("torch_geometric.data"))
        data_module.Data = _FakePyGData
        pyg_module.data = data_module
        monkeypatch.setitem(sys.modules, "torch_geometric", pyg_module)
        monkeypatch.setitem(sys.modules, "torch_geometric.data", data_module)


def _write_structure_manifest(path: Path, rows: list[tuple[str, Path]]) -> None:
    path.write_text(
        "id,path,chain,model\n" + "".join(f"{sample_id},{source},,0\n" for sample_id, source in rows),
        encoding="utf-8",
    )


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and path.name in {"meta.json", "data.npz"}
    }


def _synthetic_learned_artifact(source: Path) -> StructureArtifact:
    return build_learned_artifact(
        config=StructuralEmbedderConfig(
            representation_name="gearnet_edge",
            output_mode="both",
        ),
        result=StructuralEmbeddingResult(
            node_embedding=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            global_embedding=np.asarray([5.0, 6.0], dtype=np.float32),
            node_ids=np.asarray(["A:1::ALA", "A:2::GLY"], dtype=np.str_),
        ),
        model=StructuralModelDescriptor(
            architecture="synthetic-gearnet-edge",
            checkpoint=CheckpointDescriptor(
                identifier="synthetic.pth",
                revision="test",
                source="memory://synthetic",
                sha256="a" * 64,
            ),
        ),
        input_metadata=build_input_metadata(
            source=source,
            source_sha256=sha256_file(source),
            input_format="pdb",
            model_index=0,
            chain_ids=["A"],
        ),
        encoder_metadata={"implementation": "tests.synthetic", "version": "1"},
    )


def test_explicit_public_workflow_preserves_identity_failures_and_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "structures.csv"
    missing = tmp_path / "missing.pdb"
    _write_structure_manifest(
        manifest,
        [("first", PDB), ("second", PDB), ("failed", missing)],
    )
    artifact_root = tmp_path / "artifacts"
    results_path = tmp_path / "results.csv"

    results = eris.batch_encode(
        manifest,
        representation="residue_contact_graph",
        artifact_root=artifact_root,
        results_path=results_path,
        use_cache=False,
    )

    assert results.select("id", "status").rows() == [
        ("first", "success"),
        ("second", "success"),
        ("failed", "failed"),
    ]
    assert_frame_equal(pl.read_csv(results_path, infer_schema=False), results)
    assert not (artifact_root / "failed").exists()
    for sample_id in ("first", "second"):
        artifact = StructureArtifact.load(artifact_root / sample_id / "residue_contact_graph")
        assert artifact.meta["representation_name"] == "residue_contact_graph"
        assert artifact.meta["representation_family"] == "explicit"

    before_pack = _artifact_bytes(artifact_root)
    index_path = tmp_path / "index.csv"
    packed_rows = eris.pack(artifact_root, index_path)
    assert packed_rows["id"].to_list() == ["first", "second"]
    assert _artifact_bytes(artifact_root) == before_pack

    resume_manifest = tmp_path / "resume.csv"
    _write_structure_manifest(resume_manifest, [("first", PDB), ("second", PDB)])

    def must_not_execute(*args: object, **kwargs: object) -> StructureArtifact:  # noqa: ARG001
        raise AssertionError("equivalent artifacts must be reused during resume")

    monkeypatch.setattr(eris, "encode", must_not_execute)
    resumed = eris.batch_encode(
        resume_manifest,
        representation="residue_contact_graph",
        artifact_root=artifact_root,
        resume=True,
        use_cache=False,
    )
    assert resumed.select("id", "status").rows() == [
        ("first", "success"),
        ("second", "success"),
    ]


def test_explicit_tensor_and_pyg_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    from eris.datasets.collate import collate_graph

    manifest = tmp_path / "structures.csv"
    _write_structure_manifest(manifest, [("sample", PDB)])
    artifact_root = tmp_path / "artifacts"
    eris.batch_encode(
        manifest,
        representation="residue_contact_graph",
        artifact_root=artifact_root,
        use_cache=False,
    )
    index_path = tmp_path / "index.csv"
    eris.pack(artifact_root, index_path)

    dataset = ErisFolderDataset(index_path)
    samples = [dataset[index] for index in range(len(dataset))]
    assert [sample["id"] for sample in samples] == ["sample"]
    assert collate_graph(samples)["id"] == ["sample"]

    artifact = StructureArtifact.load(artifact_root / "sample" / "residue_contact_graph")
    assert eris.to_torch(artifact)["representation_name"] == "residue_contact_graph"
    _install_fake_pyg(monkeypatch)
    pyg_data = eris.to_pyg(artifact)
    assert pyg_data.edge_index.shape[1] == artifact.data["edge_index"].shape[1]


def test_learned_public_workflow_uses_canonical_artifacts_and_torch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "structures.csv"
    _write_structure_manifest(manifest, [("learned", PDB)])
    artifact_root = tmp_path / "artifacts"

    monkeypatch.setattr(
        eris,
        "embed",
        lambda source, **kwargs: _synthetic_learned_artifact(Path(source)),  # noqa: ARG005
    )
    results = eris.batch_embed(
        manifest,
        representation="gearnet_edge",
        artifact_root=artifact_root,
        output_mode="both",
        device="cpu",
        precision="fp32",
        use_cache=False,
    )

    assert results.select("id", "status", "representation_family").rows() == [
        ("learned", "success", "learned")
    ]
    artifact_dir = artifact_root / "learned" / "gearnet_edge"
    artifact = StructureArtifact.load(artifact_dir)
    assert artifact.meta["representation_family"] == "learned"

    index_path = tmp_path / "learned-index.csv"
    eris.pack(artifact_root, index_path)
    sample = ErisFolderDataset(index_path)[0]
    np.testing.assert_array_equal(sample["node_ids"], artifact.data["node_ids"])


def test_cli_workflow_uses_the_same_public_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "structures.csv"
    _write_structure_manifest(manifest, [("cli", PDB)])
    artifact_root = tmp_path / "artifacts"

    result = runner.invoke(
        app,
        [
            "batch",
            "encode",
            str(manifest),
            "--representation",
            "residue_contact_graph",
            "--out",
            str(artifact_root),
            "--no-cache",
        ],
    )
    assert result.exit_code == 0, result.output
    index_path = tmp_path / "index.csv"
    result = runner.invoke(app, ["pack", str(artifact_root), "--out", str(index_path)])
    assert result.exit_code == 0, result.output
    assert index_path.is_file()

    monkeypatch.setattr(
        eris,
        "embed",
        lambda source, **kwargs: _synthetic_learned_artifact(Path(source)),  # noqa: ARG005
    )
    learned_root = tmp_path / "learned-artifacts"
    result = runner.invoke(
        app,
        [
            "batch",
            "embed",
            str(manifest),
            "--representation",
            "gearnet_edge",
            "--out",
            str(learned_root),
            "--device",
            "cpu",
            "--precision",
            "fp32",
            "--no-cache",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (learned_root / "cli" / "gearnet_edge" / "meta.json").is_file()

    assert runner.invoke(app, ["encode", "batch"]).exit_code == 2
    assert runner.invoke(app, ["embed", "batch"]).exit_code == 2


def test_cli_export_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    manifest = tmp_path / "structures.csv"
    _write_structure_manifest(manifest, [("cli", PDB)])
    artifact_root = tmp_path / "artifacts"
    result = runner.invoke(
        app,
        [
            "batch",
            "encode",
            str(manifest),
            "--representation",
            "residue_contact_graph",
            "--out",
            str(artifact_root),
            "--no-cache",
        ],
    )
    assert result.exit_code == 0, result.output
    artifact_dir = artifact_root / "cli" / "residue_contact_graph"

    torch_path = tmp_path / "graph.pt"
    result = runner.invoke(
        app,
        ["export", "to-torch", "--artifact-dir", str(artifact_dir), "--out", str(torch_path)],
    )
    assert result.exit_code == 0, result.output
    assert torch_path.is_file()

    _install_fake_pyg(monkeypatch)
    pyg_path = tmp_path / "graph-pyg.pt"
    result = runner.invoke(
        app,
        ["export", "to-pyg", "--artifact-dir", str(artifact_dir), "--out", str(pyg_path)],
    )
    assert result.exit_code == 0, result.output
    assert pyg_path.is_file()
