from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import eris
import eris.cli.pack as pack_cli
from eris.cli.main import app
from eris.pipelines.manifest import BatchResultRow

runner = CliRunner()


def _success(name: str, family: str = "explicit") -> BatchResultRow:
    return BatchResultRow(
        id="sample",
        representation_name=name,
        representation_family=family,
        artifact_dir="artifacts/sample/" + name,
        status="success",
        error_type=None,
        error_message=None,
    )


def test_m7_workflow_public_entry_points_are_available() -> None:
    assert callable(eris.batch_encode)
    assert callable(eris.batch_embed)
    assert callable(eris.batch_export)
    assert callable(eris.pack_artifacts)
    assert callable(eris.pack)
    assert callable(eris.to_torch)
    assert callable(eris.to_pyg)


def test_batch_help_exposes_dedicated_hierarchy() -> None:
    result = runner.invoke(app, ["batch", "--help"])

    assert result.exit_code == 0
    assert "encode" in result.output
    assert "embed" in result.output
    assert runner.invoke(app, ["encode", "batch"]).exit_code == 2
    assert runner.invoke(app, ["embed", "batch"]).exit_code == 2


def test_batch_encode_cli_delegates_to_public_api(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("id,path,chain,model\nsample,protein.pdb,,0\n", encoding="utf-8")
    calls: dict[str, object] = {}

    def fake_batch_encode(**kwargs):
        calls.update(kwargs)
        return [_success("residue_point_cloud")]

    monkeypatch.setattr(eris, "batch_encode", fake_batch_encode)
    result = runner.invoke(
        app,
        [
            "batch",
            "encode",
            str(manifest),
            "--representation",
            "residue_point_cloud",
            "--out",
            str(tmp_path / "artifacts"),
            "--no-cache",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls["representation"] == "residue_point_cloud"
    assert calls["manifest_path"] == manifest
    assert calls["no_cache"] is True


def test_batch_embed_cli_delegates_to_public_api(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("id,path,chain,model\nsample,protein.pdb,,0\n", encoding="utf-8")
    calls: dict[str, object] = {}

    def fake_batch_embed(**kwargs):
        calls.update(kwargs)
        return [_success("gearnet_edge", family="learned")]

    monkeypatch.setattr(eris, "batch_embed", fake_batch_embed)
    result = runner.invoke(
        app,
        [
            "batch",
            "embed",
            str(manifest),
            "--representation",
            "gearnet_edge",
            "--out",
            str(tmp_path / "artifacts"),
            "--precision",
            "fp32",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls["representation"] == "gearnet_edge"
    assert calls["manifest_path"] == manifest
    assert calls["use_cache"] is True


def test_pack_cli_delegates_to_public_pack_boundary(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def fake_pack(root, output):
        calls["root"] = root
        calls["output"] = output
        return [_success("residue_point_cloud")]

    monkeypatch.setattr(pack_cli.eris, "pack_artifacts", fake_pack)
    result = runner.invoke(
        app,
        ["pack", str(tmp_path / "artifacts"), "--out", str(tmp_path / "index.csv")],
    )

    assert result.exit_code == 0, result.output
    assert calls["root"] == tmp_path / "artifacts"
    assert calls["output"] == tmp_path / "index.csv"
