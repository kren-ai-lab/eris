from __future__ import annotations

import json
import subprocess
import sys

from typer.testing import CliRunner

import eris
import eris.cli.discovery as discovery_cli
from eris.cli.main import app

runner = CliRunner()


def test_version_uses_public_version():
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == eris.__version__


def test_public_root_hierarchy_and_help():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "representations" in result.stdout
    assert "embedders" in result.stdout
    assert "encoders" not in result.stdout


def test_all_discovery_help_commands_succeed_without_execution():
    commands = [
        ["representations", "--help"],
        ["representations", "list", "--help"],
        ["representations", "info", "--help"],
        ["embedders", "--help"],
        ["embedders", "list", "--help"],
        ["embedders", "info", "--help"],
    ]

    for command in commands:
        result = runner.invoke(app, command)
        assert result.exit_code == 0, (command, result.stdout)


def test_representation_list_delegates_to_public_api(monkeypatch):
    calls = []
    monkeypatch.setattr(
        discovery_cli.eris,
        "list_representations",
        lambda: calls.append("list") or ["one", "two"],
    )

    result = runner.invoke(app, ["representations", "list"])

    assert result.exit_code == 0
    assert calls == ["list"]
    assert "one" in result.stdout and "two" in result.stdout


def test_representation_info_delegates_and_renders_nested_json(monkeypatch):
    info = {
        "name": "synthetic",
        "family": "explicit",
        "type": "graph",
        "granularity": "residue",
        "coordinate_source": "ca",
        "description": "A graph",
        "defaults": {"k": 10},
        "parameters": [{"name": "k", "default": 10}],
    }
    calls = []
    monkeypatch.setattr(
        discovery_cli.eris,
        "get_representation_info",
        lambda name: calls.append(name) or info,
    )

    result = runner.invoke(app, ["representations", "info", "synthetic"])

    assert result.exit_code == 0
    assert calls == ["synthetic"]
    for value in ("synthetic", "explicit", "graph", "residue", "defaults", '"k": 10'):
        assert value in result.stdout


def test_embedder_list_delegates_without_availability_probe(monkeypatch):
    calls = []
    monkeypatch.setattr(
        discovery_cli.eris,
        "list_embedders",
        lambda: calls.append("list") or ["fake_embedder"],
    )

    result = runner.invoke(app, ["embedders", "list"])

    assert result.exit_code == 0
    assert calls == ["list"]
    assert "fake_embedder" in result.stdout


def test_embedder_info_delegates_and_renders_availability(monkeypatch):
    info = {
        "name": "fake_embedder",
        "family": "learned",
        "type": "embedding",
        "coordinate_source": "model_specific",
        "backend_id": "fake",
        "encoder_version": "1",
        "output_modes": ["both"],
        "granularity_by_output_mode": {"both": "mixed"},
        "parameters": [],
        "availability": {"available": False, "reason": "Install dependencies."},
    }
    calls = []
    monkeypatch.setattr(
        discovery_cli.eris,
        "get_embedder_info",
        lambda name: calls.append(name) or info,
    )

    result = runner.invoke(app, ["embedders", "info", "fake_embedder"])

    assert result.exit_code == 0
    assert calls == ["fake_embedder"]
    assert "available" in result.stdout
    assert "false" in result.stdout.lower()
    assert "Install dependencies." in result.stdout


def test_unknown_names_return_usage_error_without_traceback():
    for command in (
        ["representations", "info", "missing"],
        ["embedders", "info", "missing"],
    ):
        result = runner.invoke(app, command)
        assert result.exit_code == 2
        assert "Unknown" in result.output
        assert "Traceback" not in result.output


def test_historical_encoders_command_is_not_public():
    result = runner.invoke(app, ["encoders", "list"])

    assert result.exit_code == 2


def test_core_discovery_cli_does_not_import_optional_frameworks():
    script = """
import sys
from typer.testing import CliRunner
from eris.cli.main import app

runner = CliRunner()
for args in (['--version'], ['representations', 'list'], ['embedders', 'list']):
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.stdout
blocked = {'torch', 'torchdrug', 'torch_scatter', 'torch_cluster', 'rdkit'}
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise SystemExit(f'optional frameworks loaded: {loaded}')
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_real_embedder_info_is_core_only():
    result = runner.invoke(app, ["embedders", "info", "gearnet_edge"])

    assert result.exit_code == 0
    assert "gearnet_edge" in result.stdout
    assert "availability" in result.stdout
    json.dumps(eris.get_embedder_info("gearnet_edge"))
