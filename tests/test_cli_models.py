from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

import eris.cli.models as models_cli
from eris.cli.main import app
from eris.core.hashing import sha256_file
from eris.core.learned import (
    BackendAvailability,
    CheckpointCache,
    CheckpointDescriptor,
    EmbedderRegistry,
    StructuralEmbedderConfig,
    StructuralModelDescriptor,
)

runner = CliRunner()
CHECKPOINT_SHA256 = "6dc11e2db6eff3c90f90672af227c647125adb7476de18a2bb860457bdb598ec"


def _candidate(tmp_path: Path, content: bytes = b"synthetic checkpoint") -> Path:
    path = tmp_path / "checkpoint.pth"
    path.write_bytes(content)
    return path


def _registry(descriptor: CheckpointDescriptor) -> EmbedderRegistry:
    registry = EmbedderRegistry()
    registry.register(
        representation_name="gearnet_edge",
        backend_id="gearnet-edge",
        implementation="tests.fake.GearNetEdgeBackend",
        encoder_version="1",
        config_type=StructuralEmbedderConfig,
        model_descriptor=StructuralModelDescriptor(
            architecture="GearNet-Edge",
            checkpoint=descriptor,
        ),
        backend_factory=lambda: None,
        availability_probe=lambda: BackendAvailability(available=True),
    )
    return registry


def test_models_help_and_list_expose_only_public_model_name():
    help_result = runner.invoke(app, ["models", "--help"])
    list_help = runner.invoke(app, ["models", "list", "--help"])
    result = runner.invoke(app, ["models", "list"])

    assert help_result.exit_code == 0
    assert list_help.exit_code == 0
    assert result.exit_code == 0
    assert "gearnet_edge" in result.output
    assert "GearNet-Edge" not in result.output
    assert "mc_gearnet_edge" not in result.output
    assert "download" not in help_result.output
    assert "import" not in help_result.output
    assert "remove" not in help_result.output
    assert "status" not in help_result.output


def test_models_info_displays_frozen_official_identity_without_runtime_availability(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(models_cli, "user_cache_dir", lambda app_name: str(tmp_path / "cache"))

    result = runner.invoke(app, ["models", "info", "gearnet_edge"])

    assert result.exit_code == 0, result.output
    for expected in (
        "representation/embedder",
        "gearnet_edge",
        "architecture",
        "GearNet-Edge",
        "checkpoint identifier",
        "mc_gearnet_edge.pth",
        "checkpoint revision",
        "Zenodo record v2",
        "checkpoint source",
        "https://zenodo.org/records/7723075",
        "checkpoint sha256",
        CHECKPOINT_SHA256,
        "checkpoint cached",
        "false",
    ):
        assert expected in result.output
    assert "availability" not in result.output
    assert "output_modes" not in result.output


def test_models_info_reports_valid_checkpoint_cache_hit(monkeypatch, tmp_path):
    candidate = _candidate(tmp_path)
    descriptor = CheckpointDescriptor(
        identifier="synthetic.pth",
        sha256=sha256_file(candidate),
        revision="test-revision",
        source="memory://checkpoint",
    )
    CheckpointCache(cache_dir=tmp_path / "cache").put(candidate, descriptor)
    monkeypatch.setattr(models_cli, "learned_registry", _registry(descriptor))
    monkeypatch.setattr(models_cli, "user_cache_dir", lambda app_name: str(tmp_path / "cache"))

    result = runner.invoke(app, ["models", "info", "gearnet_edge"])

    assert result.exit_code == 0, result.output
    assert "checkpoint cached" in result.output
    assert "true" in result.output
    assert str(tmp_path / "cache") not in result.output


def test_models_info_treats_corrupt_checkpoint_cache_as_uncached(monkeypatch, tmp_path):
    candidate = _candidate(tmp_path)
    descriptor = CheckpointDescriptor(
        identifier="synthetic.pth",
        sha256=sha256_file(candidate),
        revision="test-revision",
        source="memory://checkpoint",
    )
    cached = CheckpointCache(cache_dir=tmp_path / "cache").put(candidate, descriptor)
    cached.path.write_bytes(b"corrupt")
    monkeypatch.setattr(models_cli, "learned_registry", _registry(descriptor))
    monkeypatch.setattr(models_cli, "user_cache_dir", lambda app_name: str(tmp_path / "cache"))

    result = runner.invoke(app, ["models", "info", "gearnet_edge"])

    assert result.exit_code == 0, result.output
    assert "checkpoint cached" in result.output
    assert "false" in result.output


def test_models_info_unknown_name_is_clean_usage_error():
    result = runner.invoke(app, ["models", "info", "missing"])

    assert result.exit_code == 2
    assert "Unknown model/embedder 'missing'" in result.output
    assert "Traceback" not in result.output


def test_models_read_only_commands_do_not_import_optional_frameworks():
    script = """
import sys
from typer.testing import CliRunner
from eris.cli.main import app

runner = CliRunner()
for args in (
    ["models", "--help"],
    ["models", "list"],
    ["models", "info", "gearnet_edge"],
):
    result = runner.invoke(app, args)
    assert result.exit_code == 0, (args, result.output)

blocked = {"torch", "torchdrug", "torch_scatter", "torch_cluster", "rdkit"}
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise SystemExit(f"optional frameworks loaded: {loaded}")
"""
    subprocess.run([sys.executable, "-c", script], check=True)
