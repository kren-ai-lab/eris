from typer.testing import CliRunner

from eris.cli.main import app
from eris.core.artifacts import StructureArtifact


def test_cli_runs():
    runner = CliRunner()
    result = runner.invoke(app, ["representations", "list"])

    assert result.exit_code == 0
    assert "residue_distogram" in result.output
    assert "distogram_residue" not in result.output


def test_cli_verbose_lists_derived_defaults():
    runner = CliRunner()
    result = runner.invoke(app, ["representations", "info", "residue_distogram"])

    assert result.exit_code == 0
    assert "defaults" in result.output
    assert "params_schema" not in result.output


def test_cli_encode_uses_canonical_default_representation(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "encode",
            "tests/data/1AQ2.pdb",
            "--representation",
            "residue_distogram",
            "--out",
            str(tmp_path),
            "--no-cache",
        ],
    )

    assert result.exit_code == 0
    artifact = StructureArtifact.load(tmp_path)
    assert artifact.meta["schema_version"] == 2

    info = runner.invoke(app, ["info", "show", str(tmp_path)])
    assert info.exit_code == 0
    assert "representation_name" in info.output
    assert "residue_distogram" in info.output
    assert "representation_type" in info.output
    assert "matrix" in info.output
    assert "granularity" in info.output
    assert "residue" in info.output
    assert "coordinate_source" in info.output
    assert "ca" in info.output


def test_cli_encode_help_no_longer_exposes_pre_release_options():
    runner = CliRunner()

    help_result = runner.invoke(app, ["encode", "--help"])

    assert help_result.exit_code == 0
    assert "--node-emb" not in help_result.output
    assert "--node-emb-key" not in help_result.output
    assert "--encoder" not in help_result.output
    assert "--id" not in help_result.output
    assert "--allow-unknown-params" not in help_result.output
