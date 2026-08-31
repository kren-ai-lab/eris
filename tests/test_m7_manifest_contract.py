from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from eris.pipelines.manifest import (
    BatchResultRow,
    ManifestValidationError,
    StructureManifestRow,
    read_manifest,
    write_manifest,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_structure_manifest_uses_canonical_fields_and_normalizes_values() -> None:
    row = StructureManifestRow.from_dict(
        {"id": " protein ", "path": " data/1.pdb ", "chain": " A ", "model": "2"}
    )

    assert row == StructureManifestRow(id="protein", path="data/1.pdb", chain="A", model=2)
    assert row.to_dict() == {
        "id": "protein",
        "path": "data/1.pdb",
        "chain": "A",
        "model": 2,
    }


def test_structure_manifest_accepts_path_like_values() -> None:
    row = StructureManifestRow(id="protein", path=Path("data/protein.pdb"))

    assert row.path == str(Path("data/protein.pdb"))


def test_structure_manifest_requires_id_without_path_stem_fallback(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_csv(manifest, [{"path": "data/protein.pdb"}])

    with pytest.raises(ManifestValidationError, match="row 1: field 'id' is required"):
        read_manifest(manifest, StructureManifestRow)


@pytest.mark.parametrize("row", [{"id": "x"}, {"id": "x", "path": ""}])
def test_structure_manifest_rejects_missing_or_empty_path(
    tmp_path: Path, row: dict[str, str]
) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_csv(manifest, [row])

    with pytest.raises(ManifestValidationError, match="row 1: field 'path'"):
        read_manifest(manifest, StructureManifestRow)


@pytest.mark.parametrize("model_value", [None, ""])
def test_structure_manifest_defaults_missing_or_empty_model_to_zero(
    model_value: object,
) -> None:
    row = StructureManifestRow.from_dict({"id": "x", "path": "data.pdb", "model": model_value})

    assert row.model == 0


@pytest.mark.parametrize(
    "row,match",
    [
        ({"id": "x", "path": "data.pdb", "model": "bad"}, "row 1: field 'model'"),
        ({"id": "x", "path": "data.pdb", "model": "-1"}, "row 1: field 'model'"),
    ],
)
def test_structure_manifest_rejects_invalid_models(
    tmp_path: Path, row: dict[str, str], match: str
) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_csv(manifest, [row])

    with pytest.raises(ManifestValidationError, match=match):
        read_manifest(manifest, StructureManifestRow)


def test_structure_manifest_rejects_duplicate_ids(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_csv(
        manifest,
        [
            {"id": "dup", "path": "a.pdb"},
            {"id": "dup", "path": "b.pdb"},
        ],
    )

    with pytest.raises(ManifestValidationError, match="row 2: duplicate id 'dup'"):
        read_manifest(manifest, StructureManifestRow)


def test_structure_manifest_accepts_legacy_aliases_but_writes_canonical_fields(
    tmp_path: Path,
) -> None:
    row = StructureManifestRow.from_dict(
        {
            "sample_id": "protein",
            "input": "data/protein.pdb",
            "chain_id": "",
            "model_index": "",
        }
    )

    assert row == StructureManifestRow(id="protein", path="data/protein.pdb", chain=None, model=0)

    out = tmp_path / "manifest.csv"
    write_manifest(out, [row])

    assert out.read_text(encoding="utf-8").splitlines()[0] == "id,path,chain,model"
    assert "sample_id" not in out.read_text(encoding="utf-8")
    assert read_manifest(out, StructureManifestRow) == [row]


def test_structure_manifest_rejects_conflicting_aliases() -> None:
    with pytest.raises(ManifestValidationError, match="field 'path' conflicts with alias 'input'"):
        StructureManifestRow.from_dict(
            {"id": "protein", "path": "a.pdb", "input": "b.pdb"}, row_index=1
        )


def test_structure_manifest_roundtrips_all_supported_formats(tmp_path: Path) -> None:
    rows = [
        StructureManifestRow(id="a", path="a.pdb", chain=None, model=0),
        StructureManifestRow(id="b", path="b.pdb", chain="B", model=3),
    ]

    for suffix in (".csv", ".jsonl", ".json"):
        path = tmp_path / f"structures{suffix}"
        write_manifest(path, rows)

        assert read_manifest(path, StructureManifestRow) == rows


def test_batch_result_success_contract_and_canonical_order(tmp_path: Path) -> None:
    row = BatchResultRow(
        id="protein",
        representation_name="residue_point_cloud",
        representation_family="explicit",
        artifact_dir="out/protein/residue_point_cloud",
        status="success",
        error_type="",
        error_message=None,
    )

    assert row.error_type is None
    assert row.error_message is None
    out = tmp_path / "batch.csv"
    write_manifest(out, [row])

    expected_header = (
        "id,representation_name,representation_family,artifact_dir,status,error_type,error_message"
    )
    assert out.read_text(encoding="utf-8").splitlines()[0] == expected_header
    assert read_manifest(out, BatchResultRow) == [row]


def test_batch_result_jsonl_serialization_uses_only_canonical_fields(tmp_path: Path) -> None:
    row = BatchResultRow(
        id="protein",
        representation_name="gearnet_edge",
        representation_family="learned",
        artifact_dir="out/protein/gearnet_edge",
        status="success",
    )
    out = tmp_path / "batch.jsonl"

    write_manifest(out, [row])

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert list(payload) == [
        "id",
        "representation_name",
        "representation_family",
        "artifact_dir",
        "status",
        "error_type",
        "error_message",
    ]
    assert "sample_id" not in payload
    assert "input" not in payload


def test_batch_result_failed_contract_allows_missing_artifact_dir() -> None:
    row = BatchResultRow(
        id="protein",
        representation_name="gearnet_edge",
        representation_family="learned",
        artifact_dir="",
        status="failed",
        error_type="RuntimeError",
        error_message="boom",
    )

    assert row.artifact_dir is None
    assert row.to_dict() == {
        "id": "protein",
        "representation_name": "gearnet_edge",
        "representation_family": "learned",
        "artifact_dir": None,
        "status": "failed",
        "error_type": "RuntimeError",
        "error_message": "boom",
    }


@pytest.mark.parametrize(
    "row,match",
    [
        (
            {
                "id": "protein",
                "representation_name": "x",
                "representation_family": "explicit",
                "artifact_dir": "",
                "status": "success",
            },
            "artifact_dir",
        ),
        (
            {
                "id": "protein",
                "representation_name": "x",
                "representation_family": "explicit",
                "artifact_dir": "out/x",
                "status": "success",
                "error_type": "RuntimeError",
            },
            "success rows must not include error metadata",
        ),
        (
            {
                "id": "protein",
                "representation_name": "x",
                "representation_family": "explicit",
                "status": "failed",
                "error_type": "",
                "error_message": "boom",
            },
            "error_type",
        ),
        (
            {
                "id": "protein",
                "representation_name": "x",
                "representation_family": "other",
                "artifact_dir": "out/x",
                "status": "success",
            },
            "representation_family",
        ),
        (
            {
                "id": "protein",
                "representation_name": "x",
                "representation_family": "explicit",
                "artifact_dir": "out/x",
                "status": "cached",
            },
            "status",
        ),
    ],
)
def test_batch_result_rejects_invalid_rows(tmp_path: Path, row: dict[str, str], match: str) -> None:
    path = tmp_path / "batch.json"
    path.write_text(json.dumps([row]), encoding="utf-8")

    with pytest.raises(ManifestValidationError, match=match):
        read_manifest(path, BatchResultRow)


def test_batch_result_roundtrips_all_supported_formats(tmp_path: Path) -> None:
    rows = [
        BatchResultRow(
            id="a",
            representation_name="atom_point_cloud",
            representation_family="explicit",
            artifact_dir="out/a/atom_point_cloud",
            status="success",
        ),
        BatchResultRow(
            id="b",
            representation_name="gearnet_edge",
            representation_family="learned",
            artifact_dir=None,
            status="failed",
            error_type="ValueError",
            error_message="missing CA",
        ),
    ]

    for suffix in (".csv", ".jsonl", ".json"):
        path = tmp_path / f"batch{suffix}"
        write_manifest(path, rows)

        assert read_manifest(path, BatchResultRow) == rows
