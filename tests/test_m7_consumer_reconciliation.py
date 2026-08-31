from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from eris.core.artifact_schema_v2 import (
    build_artifact_metadata,
    build_feature_keys,
    build_input_metadata,
    build_provenance,
)
from eris.core.artifacts import StructureArtifact
from eris.pipelines.manifest import BatchResultRow, ManifestValidationError, write_manifest

batch_export_module = importlib.import_module("eris.pipelines.batch_export")
manifest_module = importlib.import_module("eris.pipelines.manifest")


def _save_cloud_artifact(
    target: Path,
    *,
    representation_name: str = "residue_point_cloud",
    representation_family: str = "explicit",
) -> StructureArtifact:
    data = {
        "pos": np.asarray([[0.0, 1.0, 2.0]], dtype=np.float32),
        "node_attr": np.asarray([[1.0, 0.0]], dtype=np.float32),
    }
    parameters = {"fixture": "m7_consumer_reconciliation"}
    metadata = build_artifact_metadata(
        data=data,
        representation_name=representation_name,
        representation_family=representation_family,
        representation_type="cloud",
        granularity="residue",
        coordinate_source="ca",
        encoder={
            "implementation": "tests.fixtures.M7ConsumerProducer",
            "version": "1",
        },
        input_metadata=build_input_metadata(
            source="memory://m7-consumer",
            source_sha256="b" * 64,
            input_format="pdb",
            model_index=0,
            chain_id=None,
        ),
        parameters=parameters,
        provenance=build_provenance(
            representation_name=representation_name,
            parameters=parameters,
        ),
        output_declarations=[
            {
                "name": "pos",
                "role": "coordinates",
                "granularity": "residue",
                "layout": "NC",
            },
            {
                "name": "node_attr",
                "role": "node_features",
                "granularity": "residue",
                "layout": "NF",
            },
        ],
        feature_keys=build_feature_keys(node_attr_keys=["fixture.one", "fixture.two"]),
        counts={"n_nodes": 1},
    )
    artifact = StructureArtifact(data=data, meta=metadata)
    artifact.save(target)
    return artifact


def _write_results(path: Path, rows: list[BatchResultRow]) -> None:
    write_manifest(path, rows)


def _fake_torch_export(calls: list[tuple[Path, Path]]):
    def export(
        artifact_dir: Path,
        *,
        out_path: Path,
        prefer: str,
        device: str | None,
        matrix_as_channel: bool,
    ) -> None:
        calls.append((Path(artifact_dir), out_path))
        out_path.write_text(
            f"{prefer}|{device}|{matrix_as_channel}",
            encoding="utf-8",
        )

    return export


def test_artifact_manifest_row_legacy_symbol_is_removed() -> None:
    assert not hasattr(manifest_module, "ArtifactManifestRow")


def test_batch_export_consumes_canonical_results_and_preserves_manifest_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "artifacts" / "first" / "residue_point_cloud"
    second = tmp_path / "artifacts" / "second" / "residue_point_cloud"
    _save_cloud_artifact(first)
    _save_cloud_artifact(second)
    manifest = tmp_path / "results.csv"
    _write_results(
        manifest,
        [
            BatchResultRow(
                id="first",
                representation_name="residue_point_cloud",
                representation_family="explicit",
                artifact_dir=str(first),
                status="success",
            ),
            BatchResultRow(
                id="second",
                representation_name="residue_point_cloud",
                representation_family="explicit",
                artifact_dir=str(second),
                status="success",
            ),
        ],
    )
    calls: list[tuple[Path, Path]] = []
    monkeypatch.setattr(batch_export_module, "save_artifact_to_torch", _fake_torch_export(calls))

    stats = batch_export_module.batch_export(manifest, out=tmp_path / "exports")

    assert stats == {"tasks": 2, "ok": 2, "failed": 0, "skipped": 0}
    assert calls == [
        (first, tmp_path / "exports" / "first" / "residue_point_cloud" / "data.pt"),
        (second, tmp_path / "exports" / "second" / "residue_point_cloud" / "data.pt"),
    ]
    assert not (tmp_path / "exports" / "export_manifest.csv").exists()


def test_batch_export_counts_failed_rows_without_treating_them_as_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_dir = tmp_path / "artifacts" / "ok" / "residue_point_cloud"
    _save_cloud_artifact(artifact_dir)
    manifest = tmp_path / "results.csv"
    _write_results(
        manifest,
        [
            BatchResultRow(
                id="ok",
                representation_name="residue_point_cloud",
                representation_family="explicit",
                artifact_dir=str(artifact_dir),
                status="success",
            ),
            BatchResultRow(
                id="bad",
                representation_name="residue_point_cloud",
                representation_family="explicit",
                artifact_dir=None,
                status="failed",
                error_type="ValueError",
                error_message="cannot parse",
            ),
        ],
    )
    calls: list[tuple[Path, Path]] = []
    monkeypatch.setattr(batch_export_module, "save_artifact_to_torch", _fake_torch_export(calls))

    stats = batch_export_module.batch_export(manifest, out=tmp_path / "exports")

    assert stats == {"tasks": 2, "ok": 1, "failed": 1, "skipped": 0}
    assert calls == [
        (artifact_dir, tmp_path / "exports" / "ok" / "residue_point_cloud" / "data.pt")
    ]


def test_batch_export_rejects_legacy_two_column_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "artifact_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "artifact_dir"])
        writer.writeheader()
        writer.writerow({"id": "protein", "artifact_dir": "out/protein"})

    with pytest.raises(ManifestValidationError, match="representation_name"):
        batch_export_module.batch_export(manifest, out=tmp_path / "exports")


def test_batch_export_strict_loads_and_rejects_schema_v1_artifacts(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts" / "legacy" / "residue_point_cloud"
    artifact_dir.mkdir(parents=True)
    np.savez_compressed(artifact_dir / "data.npz", pos=np.zeros((1, 3), dtype=np.float32))
    (artifact_dir / "meta.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "encoder_name": "residue_point_cloud",
                "type": "cloud",
                "level": "residue",
                "params": {},
                "input": "legacy",
                "feature_keys": {},
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "results.csv"
    _write_results(
        manifest,
        [
            BatchResultRow(
                id="legacy",
                representation_name="residue_point_cloud",
                representation_family="explicit",
                artifact_dir=str(artifact_dir),
                status="success",
            )
        ],
    )

    with pytest.raises(ManifestValidationError, match="strict-valid Artifact Schema v2"):
        batch_export_module.batch_export(manifest, out=tmp_path / "exports")


def test_batch_export_rejects_corrupt_v2_artifacts(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts" / "corrupt" / "residue_point_cloud"
    artifact_dir.mkdir(parents=True)
    np.savez_compressed(artifact_dir / "data.npz", pos=np.zeros((1, 3), dtype=np.float32))
    (artifact_dir / "meta.json").write_text("{bad json", encoding="utf-8")
    manifest = tmp_path / "results.csv"
    _write_results(
        manifest,
        [
            BatchResultRow(
                id="corrupt",
                representation_name="residue_point_cloud",
                representation_family="explicit",
                artifact_dir=str(artifact_dir),
                status="success",
            )
        ],
    )

    with pytest.raises(ManifestValidationError):
        batch_export_module.batch_export(manifest, out=tmp_path / "exports")


def test_batch_export_rejects_result_artifact_representation_mismatch(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts" / "protein" / "residue_point_cloud"
    _save_cloud_artifact(artifact_dir, representation_name="atom_point_cloud")
    manifest = tmp_path / "results.csv"
    _write_results(
        manifest,
        [
            BatchResultRow(
                id="protein",
                representation_name="residue_point_cloud",
                representation_family="explicit",
                artifact_dir=str(artifact_dir),
                status="success",
            )
        ],
    )

    with pytest.raises(ManifestValidationError, match="representation_name"):
        batch_export_module.batch_export(manifest, out=tmp_path / "exports")


def test_batch_export_rejects_result_artifact_family_mismatch(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts" / "protein" / "residue_point_cloud"
    _save_cloud_artifact(artifact_dir)
    manifest = tmp_path / "results.csv"
    _write_results(
        manifest,
        [
            BatchResultRow(
                id="protein",
                representation_name="residue_point_cloud",
                representation_family="learned",
                artifact_dir=str(artifact_dir),
                status="success",
            )
        ],
    )

    with pytest.raises(ManifestValidationError, match="representation_family"):
        batch_export_module.batch_export(manifest, out=tmp_path / "exports")


def test_batch_export_rejects_duplicate_artifact_dirs(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts" / "shared" / "residue_point_cloud"
    _save_cloud_artifact(artifact_dir)
    manifest = tmp_path / "results.csv"
    _write_results(
        manifest,
        [
            BatchResultRow(
                id="first",
                representation_name="residue_point_cloud",
                representation_family="explicit",
                artifact_dir=str(artifact_dir),
                status="success",
            ),
            BatchResultRow(
                id="second",
                representation_name="residue_point_cloud",
                representation_family="explicit",
                artifact_dir=str(artifact_dir),
                status="success",
            ),
        ],
    )

    with pytest.raises(ManifestValidationError, match="duplicate artifact_dir"):
        batch_export_module.batch_export(manifest, out=tmp_path / "exports")


def test_batch_export_rejects_duplicate_id_representation_results(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts" / "protein" / "residue_point_cloud"
    _save_cloud_artifact(artifact_dir)
    row = BatchResultRow(
        id="protein",
        representation_name="residue_point_cloud",
        representation_family="explicit",
        artifact_dir=str(artifact_dir),
        status="success",
    )

    with pytest.raises(ManifestValidationError, match="duplicate result"):
        batch_export_module._load_successful_artifacts([row, row])
