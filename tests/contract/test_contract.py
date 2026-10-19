"""Shared ecosystem CLI contract: ``--config``, ``validate`` and the provenance sidecar."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from eris.cli.main import app
from tests.helpers import PDB

SCHEMAS = Path(__file__).parent
runner = CliRunner()


def _schema(name: str) -> Draft202012Validator:
    return Draft202012Validator(json.loads((SCHEMAS / name).read_text(encoding="utf-8")))


def _meta(artifact_dir: Path) -> dict:
    return json.loads((artifact_dir / "meta.json").read_text(encoding="utf-8"))


def _config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_config_params_and_precedence(tmp_path: Path) -> None:
    cfg = _config(tmp_path, "representation: residue_knn_graph\nparams:\n  k: 4\n")
    out = tmp_path / "a"
    assert runner.invoke(app, ["encode", str(PDB), "--config", str(cfg), "--out", str(out)]).exit_code == 0
    assert _meta(out)["parameters"]["k"] == 4
    out2 = tmp_path / "b"
    flags = ["--params", '{"k": 6}', "--out", str(out2)]
    assert runner.invoke(app, ["encode", str(PDB), "--config", str(cfg), *flags]).exit_code == 0
    assert _meta(out2)["parameters"]["k"] == 6


def test_bad_config_exits_2(tmp_path: Path) -> None:
    cfg = _config(tmp_path, "representation: residue_knn_graph\nparams:\n  cutoff: x\n")
    result = runner.invoke(app, ["encode", str(PDB), "--config", str(cfg), "--out", str(tmp_path / "a")])
    assert result.exit_code == 2
    assert not (tmp_path / "a").exists()


@pytest.mark.parametrize(
    ("body", "valid"),
    [
        ("representation: residue_knn_graph\nparams:\n  k: 4\n", True),
        ("representation: residue_knn_graph\nparams:\n  cutoff: x\n  nope: 1\nbad: 1\n", False),
    ],
)
def test_validate_json_conforms(tmp_path: Path, body: str, valid: bool) -> None:  # noqa: FBT001
    result = runner.invoke(app, ["validate", "encode", str(_config(tmp_path, body)), "--json"])
    assert result.exit_code == (0 if valid else 2)
    issues = json.loads(result.stdout)
    _schema("validate-errors.schema.json").validate(issues)
    assert (issues == []) is valid
    if not valid:
        assert {"params.cutoff", "params.nope", "bad"} <= {issue["path"] for issue in issues}


def test_encode_provenance(tmp_path: Path) -> None:
    out = tmp_path / "a"
    args = ["encode", str(PDB), "-r", "residue_knn_graph", "--out", str(out)]
    assert runner.invoke(app, args).exit_code == 0
    doc = json.loads((tmp_path / "a.provenance.json").read_text(encoding="utf-8"))
    _schema("provenance.schema.json").validate(doc)
    assert doc["command"] == "encode"
    assert doc["extensions"]["representation"] == "residue_knn_graph"
    assert "out" not in doc["parameters"]


def test_batch_from_folder_provenance_and_record_id(tmp_path: Path) -> None:
    folder = tmp_path / "pdbs"
    folder.mkdir()
    (folder / "good.pdb").write_text(PDB.read_text(encoding="utf-8"), encoding="utf-8")
    (folder / "bad.pdb").write_text("not a structure\n", encoding="utf-8")
    out = tmp_path / "arts"
    args = ["batch", "from-folder", str(folder), "-r", "residue_knn_graph", "--out", str(out)]
    assert runner.invoke(app, args).exit_code == 0
    doc = json.loads((tmp_path / "arts.provenance.json").read_text(encoding="utf-8"))
    _schema("provenance.schema.json").validate(doc)
    assert (doc["extensions"]["records_ok"], doc["extensions"]["records_failed"]) == (1, 1)
    metas = [_meta(p.parent) for p in out.rglob("meta.json")]
    assert [m["input"]["structure_id"] for m in metas] == ["good"]
    assert metas[0]["provenance"]["created_at"].endswith("Z")
