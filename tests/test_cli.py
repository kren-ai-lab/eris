from typer.testing import CliRunner
from eris.cli.main import app

def test_cli_runs():
    runner = CliRunner()
    result = runner.invoke(app, ["encoders", "list"])
    assert result.exit_code == 0
