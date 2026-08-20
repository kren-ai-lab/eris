from typer.testing import CliRunner

from eris.cli.main import app


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
    assert (tmp_path / "1AQ2" / "residue_distogram" / "meta.json").is_file()
