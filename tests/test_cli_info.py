from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

import eris
from eris.cli.main import app
from eris.core.artifact_schema_v2 import (
    build_artifact_metadata,
    build_feature_keys,
    build_input_metadata,
    build_provenance,
)
from eris.core.artifacts import StructureArtifact

runner = CliRunner()
PDB = Path("tests/data/1AQ2.pdb")


def _learned_artifact() -> StructureArtifact:
    data = {
        "node_embedding": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "global_embedding": np.array([0.5, 0.6], dtype=np.float32),
        "node_ids": np.array(["A:1", "A:2"], dtype=np.str_),
    }
    parameters = {"output_mode": "both"}
    metadata = build_artifact_metadata(
        data=data,
        representation_name="synthetic_learned_embedding",
        representation_family="learned",
        representation_type="embedding",
        granularity="mixed",
        coordinate_source="model_specific",
        encoder={"implementation": "tests.synthetic.LearnedAdapter", "version": "1"},
        input_metadata=build_input_metadata(
            source="memory://learned",
            source_sha256="a" * 64,
            input_format="pdb",
            model_index=0,
            chain_id="A",
        ),
        parameters=parameters,
        provenance=build_provenance(
            representation_name="synthetic_learned_embedding",
            parameters=parameters,
        ),
        output_declarations=[
            {"name": "node_embedding", "role": "node_embedding", "layout": "ND"},
            {"name": "global_embedding", "role": "global_embedding", "layout": "D"},
            {
                "name": "node_ids",
                "role": "node_ids",
                "granularity": "residue",
                "layout": "N",
            },
        ],
        feature_keys=build_feature_keys(),
        counts={"n_nodes": 2},
        model={
            "architecture": "SyntheticModel",
            "checkpoint": {
                "identifier": "synthetic.pth",
                "revision": "test-revision",
                "source": "memory://checkpoint",
                "sha256": "b" * 64,
            },
        },
    )
    return StructureArtifact(data=data, meta=metadata)


def _schema_v1_artifact(target: Path) -> None:
    target.mkdir(exist_ok=True)
    np.savez_compressed(target / "data.npz", pos=np.zeros((1, 3), dtype=np.float32))
    (target / "meta.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "encoder_name": "legacy_cloud",
                "encoder_version": "0.1.0",
                "encoder_fingerprint": "legacy",
                "type": "cloud",
                "level": "residue",
                "params": {},
                "input": "legacy",
                "feature_keys": {},
                "shapes": {"pos": [1, 3]},
                "dtypes": {"pos": "float32"},
                "counts": {"n_nodes": 1},
            }
        ),
        encoding="utf-8",
    )


def test_info_help_exposes_show_and_validate_without_strict_flag():
    info_help = runner.invoke(app, ["info", "--help"])
    show_help = runner.invoke(app, ["info", "show", "--help"])
    validate_help = runner.invoke(app, ["info", "validate", "--help"])

    assert info_help.exit_code == 0
    assert show_help.exit_code == 0
    assert validate_help.exit_code == 0
    assert "show" in info_help.output
    assert "validate" in info_help.output
    assert "--strict" not in show_help.output


def test_info_show_displays_explicit_artifact_v2_metadata(tmp_path):
    artifact = eris.encode(
        str(PDB),
        representation="residue_contact_map",
        output=tmp_path,
        use_cache=False,
    )

    result = runner.invoke(app, ["info", "show", str(tmp_path)])

    assert result.exit_code == 0, result.output
    for expected in (
        "schema_version",
        "2",
        "representation_name",
        "residue_contact_map",
        "representation_family",
        "explicit",
        "representation_type",
        "matrix",
        "granularity",
        "residue",
        "coordinate_source",
        "ca",
        "parameters",
        "outputs",
        "feature_keys",
        "shapes",
        "dtypes",
        "counts",
        "provenance",
        "keys",
    ):
        assert expected in result.output
    assert artifact.meta.get("model") is None
    assert "checkpoint identifier" not in result.output


def test_info_show_displays_learned_model_metadata(tmp_path):
    _learned_artifact().save(tmp_path)

    result = runner.invoke(app, ["info", "show", str(tmp_path)])

    assert result.exit_code == 0, result.output
    for expected in (
        "synthetic_learned_embedding",
        "learned",
        "embedding",
        "model_specific",
        "SyntheticModel",
        "checkpoint identifier",
        "synthetic.pth",
        "checkpoint revision",
        "test-revision",
        "checkpoint source",
        "memory://checkpoint",
        "checkpoint sha256",
        "b" * 64,
    ):
        assert expected in result.output


def test_info_validate_accepts_valid_schema_v2_artifact(tmp_path):
    eris.encode(str(PDB), representation="residue_distogram", output=tmp_path, use_cache=False)

    result = runner.invoke(app, ["info", "validate", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "Artifact is valid (Schema v2): residue_distogram" in result.output


def test_info_commands_reject_absent_and_incomplete_artifacts_cleanly(tmp_path):
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "meta.json").write_text("{}", encoding="utf-8")

    for command, artifact_dir, expected in (
        ("show", tmp_path / "missing", "does not exist"),
        ("validate", partial, "Missing artifact files"),
    ):
        result = runner.invoke(app, ["info", command, str(artifact_dir)])
        assert result.exit_code == 2
        assert expected in result.output
        assert "Traceback" not in result.output


def test_info_commands_reject_schema_v1_cleanly(tmp_path):
    _schema_v1_artifact(tmp_path)

    for command in ("show", "validate"):
        result = runner.invoke(app, ["info", command, str(tmp_path)])
        assert result.exit_code == 2
        assert "Schema v1 is no longer supported" in result.output
        assert "Traceback" not in result.output


def test_info_commands_reject_invalid_v2_cleanly(tmp_path):
    eris.encode(str(PDB), representation="residue_distogram", output=tmp_path, use_cache=False)
    meta_path = tmp_path / "meta.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["counts"]["n_nodes"] = 999
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")

    for command in ("show", "validate"):
        result = runner.invoke(app, ["info", command, str(tmp_path)])
        assert result.exit_code == 2
        assert "n_nodes" in result.output
        assert "Traceback" not in result.output


def test_info_read_only_commands_do_not_import_optional_frameworks():
    script = """
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import eris
from typer.testing import CliRunner
from eris.cli.main import app

with TemporaryDirectory() as tmpdir:
    output = Path(tmpdir) / "artifact"
    eris.encode(
        "tests/data/1AQ2.pdb",
        representation="residue_distogram",
        output=output,
        use_cache=False,
    )
    runner = CliRunner()
    for args in (
        ["info", "--help"],
        ["info", "validate", "--help"],
        ["info", "show", str(output)],
        ["info", "validate", str(output)],
    ):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, (args, result.output)

blocked = {"torch", "torchdrug", "torch_scatter", "torch_cluster", "rdkit"}
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise SystemExit(f"optional frameworks loaded: {loaded}")
"""
    subprocess.run([sys.executable, "-c", script], check=True)
