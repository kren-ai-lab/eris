from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

import eris.cli.cache as cache_cli
from eris.cli.main import app
from eris.core.artifact_schema_v2 import (
    build_artifact_metadata,
    build_feature_keys,
    build_input_metadata,
    build_provenance,
)
from eris.core.artifacts import StructureArtifact
from eris.core.cache import DiskCache
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


def _valid_artifact() -> StructureArtifact:
    data = {"pos": np.array([[0.0, 0.0, 0.0]], dtype=np.float32)}
    parameters = {"center": True}
    meta = build_artifact_metadata(
        data=data,
        representation_name="synthetic_cache_artifact",
        representation_family="explicit",
        representation_type="cloud",
        granularity="residue",
        coordinate_source="ca",
        encoder={"implementation": "tests.synthetic.CacheEncoder", "version": "1.0"},
        input_metadata=build_input_metadata(
            source="memory://cache-test",
            source_sha256="a" * 64,
            input_format="pdb",
            model_index=0,
            chain_id=None,
        ),
        parameters=parameters,
        provenance=build_provenance(
            representation_name="synthetic_cache_artifact",
            parameters=parameters,
        ),
        output_declarations=[
            {
                "name": "pos",
                "role": "coordinates",
                "granularity": "residue",
                "layout": "N3",
            }
        ],
        feature_keys=build_feature_keys(),
        counts={"n_nodes": 1},
    )
    return StructureArtifact(data=data, meta=meta)


def _write_valid_artifact_cache(cache_root: Path, key: str = "1" * 64) -> Path:
    DiskCache(cache_root).set_v2(key, _valid_artifact())
    return cache_root / "v2" / key


def _checkpoint_descriptor_for(path: Path) -> CheckpointDescriptor:
    return CheckpointDescriptor(
        identifier="synthetic.pth",
        sha256=sha256_file(path),
        revision="test-revision",
        source="memory://checkpoint",
    )


def _write_checkpoint_cache(
    cache_root: Path,
    tmp_path: Path,
    content: bytes = b"synthetic checkpoint",
) -> CheckpointDescriptor:
    candidate = _candidate(tmp_path, content)
    descriptor = _checkpoint_descriptor_for(candidate)
    CheckpointCache(cache_dir=cache_root).put(candidate, descriptor)
    return descriptor


def test_cache_info_help_exists_without_changing_clear_help():
    cache_help = runner.invoke(app, ["cache", "--help"])
    info_help = runner.invoke(app, ["cache", "info", "--help"])
    clear_help = runner.invoke(app, ["cache", "clear", "--help"])

    assert cache_help.exit_code == 0
    assert info_help.exit_code == 0
    assert clear_help.exit_code == 0
    assert "info" in cache_help.output
    assert "clear" in cache_help.output
    assert "--target" in clear_help.output
    assert "artifacts" in clear_help.output
    assert "checkpoints" in clear_help.output
    assert "all" in clear_help.output
    assert "--yes" in clear_help.output


def test_cache_clear_requires_target_and_does_not_mutate(monkeypatch, tmp_path):
    cache_root = tmp_path / "cache"
    artifact_entry = _write_artifact_entry(cache_root)
    checkpoint_dir = cache_root / "checkpoints" / ("b" * 64)
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "checkpoint").write_bytes(b"checkpoint")
    monkeypatch.setattr(cache_cli, "user_cache_dir", lambda app_name: str(cache_root))

    result = runner.invoke(app, ["cache", "clear", "--yes"])

    assert result.exit_code == 2
    assert artifact_entry.is_dir()
    assert checkpoint_dir.is_dir()


def test_cache_clear_rejects_invalid_target_and_does_not_mutate(monkeypatch, tmp_path):
    cache_root = tmp_path / "cache"
    artifact_entry = _write_artifact_entry(cache_root)
    monkeypatch.setattr(cache_cli, "user_cache_dir", lambda app_name: str(cache_root))

    result = runner.invoke(app, ["cache", "clear", "--target", "foo", "--yes"])

    assert result.exit_code == 2
    assert artifact_entry.is_dir()


def test_cache_clear_absent_root_is_noop_and_does_not_create_root(monkeypatch, tmp_path):
    cache_root = tmp_path / "absent-cache"
    monkeypatch.setattr(cache_cli, "user_cache_dir", lambda app_name: str(cache_root))

    for target in ("artifacts", "checkpoints", "all"):
        result = runner.invoke(app, ["cache", "clear", "--target", target, "--yes"])

        assert result.exit_code == 0, result.output
        assert "nothing to clear" in result.output
        assert not cache_root.exists()


def test_cache_clear_without_yes_can_be_cancelled(monkeypatch, tmp_path):
    cache_root = tmp_path / "cache"
    artifact_entry = _write_artifact_entry(cache_root)
    monkeypatch.setattr(cache_cli, "user_cache_dir", lambda app_name: str(cache_root))

    result = runner.invoke(app, ["cache", "clear", "--target", "artifacts"], input="n\n")

    assert result.exit_code == 0, result.output
    assert "Cache clear cancelled" in result.output
    assert artifact_entry.is_dir()


def test_cache_clear_without_yes_can_be_confirmed(monkeypatch, tmp_path):
    cache_root = tmp_path / "cache"
    artifact_entry = _write_artifact_entry(cache_root)
    monkeypatch.setattr(cache_cli, "user_cache_dir", lambda app_name: str(cache_root))

    result = runner.invoke(app, ["cache", "clear", "--target", "artifacts"], input="y\n")

    assert result.exit_code == 0, result.output
    assert "Cleared Eris artifacts cache namespaces: artifact" in result.output
    assert not artifact_entry.exists()


def test_cache_clear_artifacts_yes_preserves_checkpoint(monkeypatch, tmp_path):
    cache_root = tmp_path / "cache"
    artifact_entry = _write_artifact_entry(cache_root)
    content = b"checkpoint survives artifact clear"
    descriptor = _write_checkpoint_cache(cache_root, tmp_path, content)
    checkpoint = CheckpointCache(cache_dir=cache_root).get(descriptor)
    assert checkpoint is not None
    checkpoint_path = checkpoint.path
    checkpoint_sha = checkpoint.sha256
    monkeypatch.setattr(cache_cli, "user_cache_dir", lambda app_name: str(cache_root))

    result = runner.invoke(app, ["cache", "clear", "--target", "artifacts", "--yes"])

    assert result.exit_code == 0, result.output
    assert not artifact_entry.exists()
    assert checkpoint_path.is_file()
    assert checkpoint_path.read_bytes() == content
    cached = CheckpointCache(cache_dir=cache_root).get(descriptor)
    assert cached is not None
    assert cached.path == checkpoint_path
    assert cached.sha256 == checkpoint_sha


def test_cache_clear_checkpoints_yes_preserves_artifact(monkeypatch, tmp_path):
    cache_root = tmp_path / "cache"
    artifact_key = "1" * 64
    artifact_entry = _write_valid_artifact_cache(cache_root, artifact_key)
    descriptor = _write_checkpoint_cache(cache_root, tmp_path)
    checkpoint = CheckpointCache(cache_dir=cache_root).get(descriptor)
    assert checkpoint is not None
    monkeypatch.setattr(cache_cli, "user_cache_dir", lambda app_name: str(cache_root))

    result = runner.invoke(app, ["cache", "clear", "--target", "checkpoints", "--yes"])

    assert result.exit_code == 0, result.output
    assert not checkpoint.path.exists()
    assert artifact_entry.is_dir()
    loaded = DiskCache(cache_root).get_v2(artifact_key)
    assert loaded is not None
    np.testing.assert_array_equal(loaded.data["pos"], _valid_artifact().data["pos"])


def test_cache_clear_all_clears_known_namespaces_and_preserves_unknown_content(
    monkeypatch,
    tmp_path,
):
    cache_root = tmp_path / "cache"
    artifact_entry = _write_artifact_entry(cache_root)
    descriptor = _write_checkpoint_cache(cache_root, tmp_path)
    checkpoint = CheckpointCache(cache_dir=cache_root).get(descriptor)
    assert checkpoint is not None
    unknown = cache_root / "unknown.txt"
    unknown.write_text("keep me", encoding="utf-8")
    monkeypatch.setattr(cache_cli, "user_cache_dir", lambda app_name: str(cache_root))

    result = runner.invoke(app, ["cache", "clear", "--target", "all", "--yes"])

    assert result.exit_code == 0, result.output
    assert not artifact_entry.exists()
    assert not checkpoint.path.exists()
    assert unknown.read_text(encoding="utf-8") == "keep me"
    assert cache_root.is_dir()


def test_cache_clear_all_removes_empty_root_after_known_namespaces(monkeypatch, tmp_path):
    cache_root = tmp_path / "cache"
    _write_artifact_entry(cache_root)
    _write_checkpoint_cache(cache_root, tmp_path)
    monkeypatch.setattr(cache_cli, "user_cache_dir", lambda app_name: str(cache_root))

    result = runner.invoke(app, ["cache", "clear", "--target", "all", "--yes"])

    assert result.exit_code == 0, result.output
    assert "Removed empty Eris cache root" in result.output
    assert not cache_root.exists()


def test_cache_clear_namespace_file_is_clean_usage_error(monkeypatch, tmp_path):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    artifact_namespace = cache_root / "v2"
    artifact_namespace.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(cache_cli, "user_cache_dir", lambda app_name: str(cache_root))

    result = runner.invoke(app, ["cache", "clear", "--target", "artifacts", "--yes"])

    assert result.exit_code == 2
    assert "Expected Eris artifact cache namespace to be a directory" in result.output
    assert artifact_namespace.is_file()


def test_cache_clear_preserves_legacy_flat_cache_as_unknown_content(monkeypatch, tmp_path):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    legacy_key = "2" * 64
    legacy_npz = cache_root / f"{legacy_key}.npz"
    legacy_json = cache_root / f"{legacy_key}.json"
    np.savez_compressed(legacy_npz, value=np.array([1], dtype=np.int64))
    legacy_json.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cache_cli, "user_cache_dir", lambda app_name: str(cache_root))

    result = runner.invoke(app, ["cache", "clear", "--target", "all", "--yes"])

    assert result.exit_code == 0, result.output
    assert legacy_npz.is_file()
    assert legacy_json.is_file()
    assert cache_root.is_dir()


def test_cache_info_reflects_target_aware_clear_operations(monkeypatch, tmp_path):
    cache_root = tmp_path / "cache"
    descriptor = _write_checkpoint_cache(cache_root, tmp_path)
    monkeypatch.setattr(cache_cli, "GEARNET_EDGE_CHECKPOINT_DESCRIPTOR", descriptor)
    monkeypatch.setattr(cache_cli, "user_cache_dir", lambda app_name: str(cache_root))

    _write_artifact_entry(cache_root)
    artifact_result = runner.invoke(app, ["cache", "clear", "--target", "artifacts", "--yes"])
    artifact_info = runner.invoke(app, ["cache", "info"])

    assert artifact_result.exit_code == 0, artifact_result.output
    assert artifact_info.exit_code == 0, artifact_info.output
    assert "artifact cache" in artifact_info.output
    assert "absent" in artifact_info.output
    assert "gearnet_edge checkpoint cached" in artifact_info.output
    assert "true" in artifact_info.output

    _write_artifact_entry(cache_root)
    checkpoint_result = runner.invoke(
        app,
        ["cache", "clear", "--target", "checkpoints", "--yes"],
    )
    checkpoint_info = runner.invoke(app, ["cache", "info"])

    assert checkpoint_result.exit_code == 0, checkpoint_result.output
    assert checkpoint_info.exit_code == 0, checkpoint_info.output
    assert "artifact cache" in checkpoint_info.output
    assert "present" in checkpoint_info.output
    assert "gearnet_edge checkpoint cached" in checkpoint_info.output
    assert "false" in checkpoint_info.output


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
