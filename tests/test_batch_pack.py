from __future__ import annotations

import importlib
import shutil
from pathlib import Path

import numpy as np
import pytest

import eris
from eris.core.artifact_schema_v2 import build_input_metadata
from eris.core.artifacts import StructureArtifact
from eris.core.hashing import sha256_file
from eris.core.learned.artifacts import build_learned_artifact
from eris.core.learned.backends.gearnet.config import GearNetEdgeConfig
from eris.core.learned.backends.gearnet.metadata import (
    GEARNET_EDGE_ENCODER_VERSION,
    GEARNET_EDGE_IMPLEMENTATION,
    GEARNET_EDGE_MODEL_DESCRIPTOR,
    GEARNET_EDGE_REPRESENTATION_NAME,
)
from eris.core.learned.execution import (
    ResolvedExecution,
    StructuralExecutionConfig,
    build_execution_provenance,
)
from eris.core.learned.types import StructuralEmbeddingResult
from eris.pipelines.batch_encode import batch_embed, batch_encode
from eris.pipelines.manifest import BatchResultRow, StructureManifestRow, read_manifest

PDB = Path("tests/data/1AQ2.pdb")
OTHER_PDB = Path("tests/data/3EC1.pdb")
batch_module = importlib.import_module("eris.pipelines.batch_encode")


def _explicit_artifact(
    *,
    representation: str = "residue_contact_map",
    source: Path = PDB,
    chain_id: str | None = None,
    model_index: int = 0,
    use_cache: bool = False,
    **params,
) -> StructureArtifact:
    return eris.encode(
        source,
        representation=representation,
        chain_id=chain_id,
        model_index=model_index,
        use_cache=use_cache,
        **params,
    )


def _learned_artifact(
    *,
    source: Path = PDB,
    output_mode: str = "both",
    chain_id: str | None = None,
    model_index: int = 0,
    precision: str = "fp32",
    checkpoint_identifier: str | None = None,
) -> StructureArtifact:
    config = GearNetEdgeConfig(
        representation_name=GEARNET_EDGE_REPRESENTATION_NAME,
        output_mode=output_mode,
    )
    execution_request = StructuralExecutionConfig(device="cpu", precision=precision)
    execution_resolved = ResolvedExecution(device="cpu", precision=precision)
    result = StructuralEmbeddingResult(
        node_embedding=(
            np.ones((2, 4), dtype=np.float32) if output_mode in {"node", "both"} else None
        ),
        global_embedding=(
            np.ones((4,), dtype=np.float32) if output_mode in {"global", "both"} else None
        ),
        node_ids=(np.array(["A:1::ALA", "A:2::GLY"]) if output_mode in {"node", "both"} else None),
    )
    model = GEARNET_EDGE_MODEL_DESCRIPTOR
    if checkpoint_identifier is not None:
        from eris.core.learned.checkpoints import CheckpointDescriptor, StructuralModelDescriptor

        checkpoint = CheckpointDescriptor(
            identifier=checkpoint_identifier,
            sha256=model.checkpoint.sha256,
            revision=model.checkpoint.revision,
            source=model.checkpoint.source,
        )
        model = StructuralModelDescriptor(
            architecture=model.architecture,
            checkpoint=checkpoint,
        )

    return build_learned_artifact(
        config=config,
        result=result,
        model=model,
        input_metadata=build_input_metadata(
            source=source,
            source_sha256=sha256_file(source),
            input_format="pdb",
            model_index=model_index,
            chain_id=chain_id,
        ),
        encoder_metadata={
            "implementation": GEARNET_EDGE_IMPLEMENTATION,
            "version": GEARNET_EDGE_ENCODER_VERSION,
        },
        provenance_environment={
            "execution": build_execution_provenance(
                execution_request,
                execution_resolved,
            )
        },
    )


def test_batch_encode_success_preserves_order_layout_and_strict_v2(tmp_path: Path) -> None:
    rows = [
        StructureManifestRow(id="first", path=str(PDB)),
        StructureManifestRow(id="second", path=str(PDB)),
    ]
    root = tmp_path / "artifacts"

    results = batch_encode(
        rows,
        representation="residue_contact_map",
        artifact_root=root,
        use_cache=False,
    )

    assert [row.id for row in results] == ["first", "second"]
    assert all(row.status == "success" for row in results)
    for row in rows:
        target = root / row.id / "residue_contact_map"
        assert target.is_dir()
        artifact = StructureArtifact.load(target, strict=True)
        assert artifact.meta["representation_name"] == "residue_contact_map"
        assert artifact.meta["representation_family"] == "explicit"


def test_batch_encode_delegates_to_public_api_and_propagates_config(monkeypatch, tmp_path):
    artifact = _explicit_artifact()
    captured = []

    def fake_encode(input, **kwargs):
        captured.append((input, kwargs))
        return artifact

    monkeypatch.setattr(batch_module.eris, "encode", fake_encode)
    rows = [StructureManifestRow(id="protein", path="input.pdb", chain="A", model=2)]

    results = batch_encode(
        rows,
        representation="residue_knn_graph",
        artifact_root=tmp_path,
        params={"k": 8},
        use_cache=False,
    )

    assert results[0].status == "success"
    assert captured == [
        (
            "input.pdb",
            {
                "representation": "residue_knn_graph",
                "chain_id": "A",
                "model_index": 2,
                "use_cache": False,
                "k": 8,
            },
        )
    ]


def test_batch_encode_partial_failure_continues_and_writes_results(tmp_path, monkeypatch):
    artifact = _explicit_artifact()
    calls = []

    def fake_encode(input, **kwargs):
        calls.append(input)
        if input == "bad.pdb":
            raise ValueError("cannot parse")
        return artifact

    monkeypatch.setattr(batch_module.eris, "encode", fake_encode)
    rows = [
        StructureManifestRow(id="ok1", path="ok1.pdb"),
        StructureManifestRow(id="bad", path="bad.pdb"),
        StructureManifestRow(id="ok2", path="ok2.pdb"),
    ]
    results_path = tmp_path / "results.csv"

    results = batch_encode(
        rows,
        representation="residue_contact_map",
        artifact_root=tmp_path / "artifacts",
        results_path=results_path,
        use_cache=False,
    )

    assert calls == ["ok1.pdb", "bad.pdb", "ok2.pdb"]
    assert [row.status for row in results] == ["success", "failed", "success"]
    assert results[1].error_type == "ValueError"
    assert results[1].error_message == "cannot parse"
    assert not (tmp_path / "artifacts" / "bad").exists()
    assert read_manifest(results_path, BatchResultRow) == results


def test_existing_output_without_resume_or_overwrite_is_a_failure(tmp_path, monkeypatch):
    target = tmp_path / "artifacts" / "protein" / "residue_contact_map"
    _explicit_artifact().save(target)

    def fail_encode(*args, **kwargs):
        raise AssertionError("existing output must not execute without resume or overwrite")

    monkeypatch.setattr(batch_module.eris, "encode", fail_encode)

    results = batch_encode(
        [StructureManifestRow(id="protein", path=str(PDB))],
        representation="residue_contact_map",
        artifact_root=tmp_path / "artifacts",
    )

    assert results[0].status == "failed"
    assert results[0].error_type == "FileExistsError"
    assert (
        StructureArtifact.load(target, strict=True).meta["representation_name"]
        == "residue_contact_map"
    )


def test_resume_reuses_equivalent_artifact_without_execution(tmp_path, monkeypatch):
    target = tmp_path / "artifacts" / "protein" / "residue_contact_map"
    _explicit_artifact().save(target)

    def fail_encode(*args, **kwargs):
        raise AssertionError("equivalent resume must not execute")

    monkeypatch.setattr(batch_module.eris, "encode", fail_encode)

    results = batch_encode(
        [StructureManifestRow(id="protein", path=str(PDB))],
        representation="residue_contact_map",
        artifact_root=tmp_path / "artifacts",
        resume=True,
    )

    assert results == [
        BatchResultRow(
            id="protein",
            representation_name="residue_contact_map",
            representation_family="explicit",
            artifact_dir=str(target),
            status="success",
        )
    ]


def test_resume_regenerates_invalid_artifact_without_deleting_first(tmp_path, monkeypatch):
    target = tmp_path / "artifacts" / "protein" / "residue_contact_map"
    target.mkdir(parents=True)
    (target / "meta.json").write_text("{bad json", encoding="utf-8")
    (target / "data.npz").write_bytes(b"bad")
    artifact = _explicit_artifact()
    calls = []

    def fake_encode(*args, **kwargs):
        calls.append((args, kwargs))
        return artifact

    monkeypatch.setattr(batch_module.eris, "encode", fake_encode)

    results = batch_encode(
        [StructureManifestRow(id="protein", path=str(PDB))],
        representation="residue_contact_map",
        artifact_root=tmp_path / "artifacts",
        resume=True,
        use_cache=False,
    )

    assert len(calls) == 1
    assert results[0].status == "success"
    assert StructureArtifact.load(target, strict=True).meta["schema_version"] == 2


@pytest.mark.parametrize(
    "existing_artifact_factory,kwargs",
    [
        (
            lambda: _explicit_artifact(representation="residue_contact_map"),
            {"representation": "residue_knn_graph"},
        ),
        (
            lambda: _explicit_artifact(representation="residue_knn_graph"),
            {
                "representation": "residue_knn_graph",
                "params": {"k": 8},
            },
        ),
        (
            lambda: _explicit_artifact(representation="residue_knn_graph", chain_id="A"),
            {"representation": "residue_knn_graph"},
        ),
        (
            lambda: _explicit_artifact(representation="residue_knn_graph", source=OTHER_PDB),
            {"representation": "residue_knn_graph"},
        ),
    ],
)
def test_resume_regenerates_non_equivalent_explicit_artifacts(
    tmp_path, monkeypatch, existing_artifact_factory, kwargs
):
    representation = kwargs["representation"]
    target = tmp_path / "artifacts" / "protein" / representation
    existing_artifact_factory().save(target)
    replacement = _explicit_artifact(representation=representation, **kwargs.get("params", {}))
    calls = []

    def fake_encode(*args, **execute_kwargs):
        calls.append((args, execute_kwargs))
        return replacement

    monkeypatch.setattr(batch_module.eris, "encode", fake_encode)

    results = batch_encode(
        [StructureManifestRow(id="protein", path=str(PDB))],
        artifact_root=tmp_path / "artifacts",
        resume=True,
        use_cache=False,
        **kwargs,
    )

    assert len(calls) == 1
    assert results[0].status == "success"
    loaded = StructureArtifact.load(target, strict=True)
    assert loaded.meta["representation_name"] == representation
    if "params" in kwargs:
        assert loaded.meta["parameters"]["k"] == 8


def test_fail_fast_persists_current_failure_before_raising(tmp_path, monkeypatch):
    artifact = _explicit_artifact()

    def fake_encode(input, **kwargs):
        if input == "bad.pdb":
            raise ValueError("boom")
        return artifact

    monkeypatch.setattr(batch_module.eris, "encode", fake_encode)
    results_path = tmp_path / "results.jsonl"

    with pytest.raises(ValueError, match="boom"):
        batch_encode(
            [
                StructureManifestRow(id="ok", path="ok.pdb"),
                StructureManifestRow(id="bad", path="bad.pdb"),
                StructureManifestRow(id="never", path="never.pdb"),
            ],
            representation="residue_contact_map",
            artifact_root=tmp_path / "artifacts",
            results_path=results_path,
            fail_fast=True,
        )

    results = read_manifest(results_path, BatchResultRow)
    assert [row.id for row in results] == ["ok", "bad"]
    assert [row.status for row in results] == ["success", "failed"]


def test_batch_embed_delegates_to_public_api_and_propagates_learned_options(tmp_path, monkeypatch):
    artifact = _learned_artifact(output_mode="node")
    captured = []

    def fake_embed(input, **kwargs):
        captured.append((input, kwargs))
        return artifact

    monkeypatch.setattr(batch_module.eris, "embed", fake_embed)

    results = batch_embed(
        [StructureManifestRow(id="protein", path="input.pdb", chain="A", model=0)],
        representation="gearnet_edge",
        artifact_root=tmp_path / "artifacts",
        output_mode="node",
        checkpoint_path=Path("checkpoint.pth"),
        device="cpu",
        precision="fp32",
        use_cache=False,
    )

    assert results[0].representation_family == "learned"
    assert results[0].status == "success"
    assert captured == [
        (
            "input.pdb",
            {
                "representation": "gearnet_edge",
                "chain_id": "A",
                "model_index": 0,
                "output_mode": "node",
                "checkpoint_path": Path("checkpoint.pth"),
                "device": "cpu",
                "precision": "fp32",
                "use_cache": False,
            },
        )
    ]


def test_learned_resume_reuses_only_matching_model_identity(tmp_path, monkeypatch):
    target = tmp_path / "artifacts" / "protein" / "gearnet_edge"
    _learned_artifact().save(target)

    def fail_embed(*args, **kwargs):
        raise AssertionError("matching learned artifact must be reused")

    monkeypatch.setattr(batch_module.eris, "embed", fail_embed)

    results = batch_embed(
        [StructureManifestRow(id="protein", path=str(PDB))],
        representation="gearnet_edge",
        artifact_root=tmp_path / "artifacts",
        resume=True,
        device="cpu",
        precision="fp32",
    )

    assert results[0].status == "success"

    shutil.rmtree(target)
    _learned_artifact(checkpoint_identifier="other.pth").save(target)
    replacement = _learned_artifact()
    calls = []

    def fake_embed(*args, **kwargs):
        calls.append((args, kwargs))
        return replacement

    monkeypatch.setattr(batch_module.eris, "embed", fake_embed)

    results = batch_embed(
        [StructureManifestRow(id="protein", path=str(PDB))],
        representation="gearnet_edge",
        artifact_root=tmp_path / "artifacts",
        resume=True,
        device="cpu",
        precision="fp32",
    )

    assert len(calls) == 1
    assert results[0].status == "success"
    assert (
        StructureArtifact.load(target, strict=True).meta["model"]
        == GEARNET_EDGE_MODEL_DESCRIPTOR.to_dict()
    )


def test_result_manifest_uses_canonical_batch_fields(tmp_path, monkeypatch):
    artifact = _explicit_artifact()
    monkeypatch.setattr(batch_module.eris, "encode", lambda *args, **kwargs: artifact)
    results_path = tmp_path / "batch_results.csv"

    batch_encode(
        [StructureManifestRow(id="protein", path=str(PDB))],
        representation="residue_contact_map",
        artifact_root=tmp_path / "artifacts",
        results_path=results_path,
        use_cache=False,
    )

    expected_header = (
        "id,representation_name,representation_family,artifact_dir,status,error_type,error_message"
    )
    assert results_path.read_text(encoding="utf-8").splitlines()[0] == expected_header
