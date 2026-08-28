from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

import eris.cli.cache as cache_cli
from eris.cli.main import app
from eris.core.hashing import sha256_file
from eris.core.learned import CheckpointCache, CheckpointDescriptor

runner = CliRunner()


def _candidate(tmp_path: Path, content: bytes = b"synthetic checkpoint") -> Path:
    path = tmp_path / "checkpoint.pth"
    path.write_bytes(content)
    return path


def _write_artifact_entry(cache_root: Path, key: str = "a" * 64) -> Path:
    entry = cache_root / "v2" / key
    entry.mkdir(parents=True)
    np.savez_compressed(entry / "data.npz", value=np.array([1], dtype=np.int64))
    (entry / "meta.json").write_text("{}", encoding="utf-8")
    return entry


def test_cache_info_help_exists_without_changing_clear_help():
    cache_help = runner.invoke(app, ["cache", "--help"])
    info_help = runner.invoke(app, ["cache", "info", "--help"])
    clear_help = runner.invoke(app, ["cache", "clear", "--help"])

    assert cache_help.exit_code == 0
    assert info_help.exit_code == 0
    assert clear_help.exit_code == 0
    assert "info" in cache_help.output
    assert "clear" in cache_help.output


def test_cache_info_absent_root_does_not_create_directory(monkeypatch, tmp_path):
    cache_root = tmp_path / "absent-cache"
    monkeypatch.setattr(cache_cli, "user_cache_dir", lambda app_name: str(cache_root))

    result = runner.invoke(app, ["cache", "info"])

    assert result.exit_code == 0, result.output
    assert "artifact cache" in result.output
    assert "absent" in result.output
    assert "artifact v2 entries" in result.output
    assert "0" in result.output
    assert "checkpoint cache" in result.output
    assert "gearnet_edge checkpoint cached" in result.output
    assert "false" in result.output
    assert not cache_root.exists()


def test_cache_info_counts_only_artifact_v2_entries_and_keeps_checkpoint_separate(
    monkeypatch,
    tmp_path,
):
    cache_root = tmp_path / "cache"
    _write_artifact_entry(cache_root)
    (cache_root / "v2" / "incomplete").mkdir()
    checkpoint_root = cache_root / "checkpoints" / ("b" * 64)
    checkpoint_root.mkdir(parents=True)
    (checkpoint_root / "checkpoint").write_bytes(b"not official")
    monkeypatch.setattr(cache_cli, "user_cache_dir", lambda app_name: str(cache_root))

    result = runner.invoke(app, ["cache", "info"])

    assert result.exit_code == 0, result.output
    assert "artifact cache" in result.output
    assert "present" in result.output
    assert "artifact v2 entries" in result.output
    assert "1" in result.output
    assert "checkpoint cache" in result.output
    assert "gearnet_edge checkpoint cached" in result.output
    assert "false" in result.output


def test_cache_info_reports_valid_official_checkpoint_status_without_path(
    monkeypatch,
    tmp_path,
):
    candidate = _candidate(tmp_path)
    descriptor = CheckpointDescriptor(
        identifier="synthetic.pth",
        sha256=sha256_file(candidate),
        revision="test-revision",
        source="memory://checkpoint",
    )
    CheckpointCache(cache_dir=tmp_path / "cache").put(candidate, descriptor)
    monkeypatch.setattr(cache_cli, "GEARNET_EDGE_CHECKPOINT_DESCRIPTOR", descriptor)
    monkeypatch.setattr(cache_cli, "user_cache_dir", lambda app_name: str(tmp_path / "cache"))

    result = runner.invoke(app, ["cache", "info"])

    assert result.exit_code == 0, result.output
    assert "gearnet_edge checkpoint cached" in result.output
    assert "true" in result.output
    assert str(tmp_path / "cache") not in result.output


def test_cache_info_is_read_only_for_existing_cache(monkeypatch, tmp_path):
    cache_root = tmp_path / "cache"
    artifact_entry = _write_artifact_entry(cache_root)
    before = sorted(path.relative_to(cache_root) for path in cache_root.rglob("*"))
    monkeypatch.setattr(cache_cli, "user_cache_dir", lambda app_name: str(cache_root))

    result = runner.invoke(app, ["cache", "info"])

    after = sorted(path.relative_to(cache_root) for path in cache_root.rglob("*"))
    assert result.exit_code == 0, result.output
    assert before == after
    assert artifact_entry.is_dir()


def test_cache_info_does_not_import_optional_frameworks():
    script = """
import sys
from typer.testing import CliRunner
from eris.cli.main import app

result = CliRunner().invoke(app, ["cache", "info"])
assert result.exit_code == 0, result.output
blocked = {"torch", "torchdrug", "torch_scatter", "torch_cluster", "rdkit"}
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise SystemExit(f"optional frameworks loaded: {loaded}")
"""
    subprocess.run([sys.executable, "-c", script], check=True)
