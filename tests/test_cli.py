from typer.testing import CliRunner
from morpheo.cli.main import app

def test_cli_runs():
    runner = CliRunner()
    result = runner.invoke(app, ["encoders"])
    assert result.exit_code == 0
