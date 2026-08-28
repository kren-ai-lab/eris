from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

import eris
import eris.cli.encode as encode_cli
from eris.cli.main import app
from eris.core.artifacts import StructureArtifact

PDB = Path("tests/data/1AQ2.pdb")
CANONICAL_REPRESENTATIONS = [
    "atom_point_cloud",
    "residue_point_cloud",
    "atom_knn_graph",
    "residue_knn_graph",
    "residue_contact_graph",
    "residue_distogram",
    "residue_distogram_bins",
    "residue_contact_map",
    "residue_contact_map_bins",
    "residue_voxel_grid",
]

runner = CliRunner()


def _assert_artifacts_equivalent(left: StructureArtifact, right: StructureArtifact) -> None:
    for key in (
        "representation_name",
        "parameters",
        "input",
        "feature_keys",
        "shapes",
        "dtypes",
        "counts",
    ):
        assert left.meta[key] == right.meta[key]
    assert list(left.data) == list(right.data)
    for name in left.data:
        np.testing.assert_array_equal(left.data[name], right.data[name])


def test_encode_help_exposes_public_single_command_only():
    help_result = runner.invoke(app, ["encode", "--help"])

    assert help_result.exit_code == 0
    for expected in (
        "INPUT",
        "--representation",
        "--chain",
        "--model",
        "--out",
        "--no-cache",
        "--params",
    ):
        assert expected in help_result.output
    for historical in (
        "single",
        "batch",
        "--encoder",
        "--id",
        "--allow-unknown-params",
    ):
        assert historical not in help_result.output


def test_direct_encode_command_delegates_to_public_api(monkeypatch, tmp_path):
    calls = []

    def fake_encode(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(meta={"representation_name": kwargs["representation"]})

    monkeypatch.setattr(encode_cli.eris, "encode", fake_encode)

    result = runner.invoke(
        app,
        [
            "encode",
            "input.pdb",
            "--representation",
            "residue_knn_graph",
            "--chain",
            "A",
            "--model",
            "1",
            "--out",
            str(tmp_path),
            "--no-cache",
            "--params",
            '{"k": 8}',
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        (
            (),
            {
                "input": "input.pdb",
                "representation": "residue_knn_graph",
                "output": tmp_path,
                "use_cache": False,
                "k": 8,
                "chain_id": "A",
                "model_index": 1,
            },
        )
    ]


def test_absent_chain_and_model_are_not_forwarded(monkeypatch, tmp_path):
    captured = {}

    def fake_encode(*args, **kwargs):
        del args
        captured.update(kwargs)
        return SimpleNamespace(meta={"representation_name": kwargs["representation"]})

    monkeypatch.setattr(encode_cli.eris, "encode", fake_encode)

    result = runner.invoke(
        app,
        [
            "encode",
            "input.pdb",
            "--representation",
            "residue_contact_map",
            "--out",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert captured["use_cache"] is True
    assert "chain_id" not in captured
    assert "model_index" not in captured


@pytest.mark.parametrize("representation", CANONICAL_REPRESENTATIONS)
def test_all_public_representations_encode_from_cli(representation, tmp_path):
    output = tmp_path / representation

    result = runner.invoke(
        app,
        [
            "encode",
            str(PDB),
            "--representation",
            representation,
            "--out",
            str(output),
            "--no-cache",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output / "meta.json").is_file()
    assert (output / "data.npz").is_file()
    artifact = StructureArtifact.load(output, strict=True)
    assert artifact.meta["representation_name"] == representation


def test_cli_artifact_is_equivalent_to_public_api_artifact(tmp_path):
    cli_output = tmp_path / "cli"
    api_output = tmp_path / "api"

    result = runner.invoke(
        app,
        [
            "encode",
            str(PDB),
            "--representation",
            "residue_knn_graph",
            "--out",
            str(cli_output),
            "--no-cache",
            "--params",
            '{"k": 8}',
        ],
    )

    assert result.exit_code == 0, result.output
    api_artifact = eris.encode(
        str(PDB),
        representation="residue_knn_graph",
        output=api_output,
        use_cache=False,
        k=8,
    )
    cli_artifact = StructureArtifact.load(cli_output, strict=True)
    reloaded_api_artifact = StructureArtifact.load(api_output, strict=True)
    _assert_artifacts_equivalent(cli_artifact, api_artifact)
    _assert_artifacts_equivalent(cli_artifact, reloaded_api_artifact)


def test_cli_custom_param_is_reflected_in_artifact(tmp_path):
    output = tmp_path / "custom"

    result = runner.invoke(
        app,
        [
            "encode",
            str(PDB),
            "--representation",
            "residue_knn_graph",
            "--out",
            str(output),
            "--no-cache",
            "--params",
            '{"k": 8}',
        ],
    )

    assert result.exit_code == 0, result.output
    artifact = StructureArtifact.load(output, strict=True)
    assert artifact.meta["parameters"]["k"] == 8
    expected = eris.encode(str(PDB), representation="residue_knn_graph", use_cache=False, k=8)
    _assert_artifacts_equivalent(artifact, expected)


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ("{bad json", "Invalid JSON for --params"),
        ("[1, 2]", "--params must be a JSON object"),
        ('{"kk": 8}', "Unknown parameters"),
        ('{"k": "wide"}', "Invalid parameters"),
    ],
)
def test_params_errors_are_clean_usage_errors(params, expected, tmp_path):
    result = runner.invoke(
        app,
        [
            "encode",
            str(PDB),
            "--representation",
            "residue_knn_graph",
            "--out",
            str(tmp_path),
            "--params",
            params,
        ],
    )

    assert result.exit_code == 2
    assert expected in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("representation", ["missing", "gearnet_edge"])
def test_unknown_and_learned_names_are_rejected_cleanly(representation, tmp_path):
    result = runner.invoke(
        app,
        [
            "encode",
            str(PDB),
            "--representation",
            representation,
            "--out",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 2
    assert "Unknown representation" in result.output
    assert "Traceback" not in result.output


def test_missing_representation_is_required(tmp_path):
    result = runner.invoke(app, ["encode", str(PDB), "--out", str(tmp_path)])

    assert result.exit_code == 2
    assert "--representation" in result.output


def test_structure_load_error_is_clean_usage_error(tmp_path):
    result = runner.invoke(
        app,
        [
            "encode",
            "missing.pdb",
            "--representation",
            "residue_contact_map",
            "--out",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 2
    assert "File not found" in result.output or "Failed to parse structure" in result.output
    assert "Traceback" not in result.output


def test_historical_encode_single_and_batch_are_not_public_help_targets():
    root_help = runner.invoke(app, ["--help"])
    encode_help = runner.invoke(app, ["encode", "--help"])

    assert root_help.exit_code == 0
    assert encode_help.exit_code == 0
    assert "encoders" not in root_help.output
    assert "single" not in encode_help.output
    assert "batch" not in encode_help.output
    assert runner.invoke(app, ["encode", "batch"]).exit_code == 2


def test_explicit_encode_cli_does_not_import_optional_learned_frameworks():
    script = """
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typer.testing import CliRunner
from eris.cli.main import app

runner = CliRunner()
with TemporaryDirectory() as tmpdir:
    output = Path(tmpdir) / 'artifact'
    for args in (
        ['encode', '--help'],
        [
            'encode',
            'tests/data/1AQ2.pdb',
            '--representation',
            'residue_contact_map',
            '--out',
            str(output),
            '--no-cache',
        ],
    ):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
blocked = {'torch', 'torchdrug', 'torch_scatter', 'torch_cluster', 'rdkit'}
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise SystemExit(f'optional frameworks loaded: {loaded}')
"""

    subprocess.run([sys.executable, "-c", script], check=True)
