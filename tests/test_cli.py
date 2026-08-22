from typer.testing import CliRunner

from eris.cli.main import app
from eris.core.artifacts import StructureArtifact


def test_cli_runs():
    runner = CliRunner()
    result = runner.invoke(app, ["encoders", "list"])

    assert result.exit_code == 0
    assert "residue_distogram" in result.output
    assert "distogram_residue" not in result.output


def test_cli_verbose_lists_derived_defaults():
    runner = CliRunner()
    result = runner.invoke(app, ["encoders", "list", "--verbose"])

    assert result.exit_code == 0
    assert "defaults" in result.output
    assert "params_schema" not in result.output


def test_cli_single_uses_canonical_default_encoder(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "encode",
            "single",
            "--input",
            "tests/data/1AQ2.pdb",
            "--out",
            str(tmp_path),
            "--no-cache",
        ],
    )

    assert result.exit_code == 0
    artifact_dir = tmp_path / "1AQ2" / "residue_distogram"
    artifact = StructureArtifact.load(artifact_dir)
    assert artifact.meta["schema_version"] == 2

    info = runner.invoke(app, ["info", "show", str(artifact_dir)])
    assert info.exit_code == 0
    assert "representation: residue_distogram" in info.output
    assert "type=matrix granularity=residue coordinate_source=ca" in info.output


def test_cli_single_no_longer_exposes_post_encode_node_embedding_options():
    runner = CliRunner()

    help_result = runner.invoke(app, ["encode", "single", "--help"])

    assert help_result.exit_code == 0
    assert "--node-emb" not in help_result.output
    assert "--node-emb-key" not in help_result.output
