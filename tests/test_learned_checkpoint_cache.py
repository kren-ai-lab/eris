from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

import eris.core.learned.checkpoint_cache as checkpoint_cache_module
from eris.core.cache import compute_cache_key_v2
from eris.core.hashing import sha256_file
from eris.core.learned import (
    CheckpointCache,
    CheckpointDescriptor,
    StructuralModelDescriptor,
    VerifiedCheckpoint,
)
from eris.exceptions import CacheError, ExecutionError

CHECKPOINT_BYTES = b"fake checkpoint bytes"


def _candidate(tmp_path: Path, name: str = "weights.pt", content: bytes = CHECKPOINT_BYTES) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _descriptor(path: Path, **overrides: object) -> CheckpointDescriptor:
    values = {
        "identifier": "synthetic-checkpoint",
        "sha256": sha256_file(path),
        "revision": "revision-1",
        "source": "https://models.example/weights.pt",
    }
    values.update(overrides)
    return CheckpointDescriptor(**values)


def _canonical(cache_root: Path, descriptor: CheckpointDescriptor) -> Path:
    return cache_root / "checkpoints" / descriptor.sha256 / "checkpoint"


def _stage_files(entry_dir: Path) -> list[Path]:
    return list(entry_dir.glob(".eris-checkpoint-*")) if entry_dir.exists() else []


def _artifact_cache_key(model: StructuralModelDescriptor) -> str:
    return compute_cache_key_v2(
        representation_name="synthetic_structure_embedding",
        encoder_implementation="tests.fake_backend.FakeStructuralEncoder",
        encoder_version="1.0",
        config_fingerprint="a" * 64,
        source_sha256="b" * 64,
        parser_hint=".pdb",
        model_index=0,
        chain_id=None,
        model=model.to_cache_identity(),
    )


def test_empty_get_is_a_lazy_side_effect_free_miss(tmp_path):
    cache = CheckpointCache(cache_dir=tmp_path)
    descriptor = CheckpointDescriptor(identifier="missing", sha256="a" * 64)

    assert cache.get(descriptor) is None
    assert not (tmp_path / "checkpoints").exists()


def test_valid_put_and_get_use_exact_canonical_layout(tmp_path):
    candidate = _candidate(tmp_path / "input")
    descriptor = _descriptor(candidate)
    cache_root = tmp_path / "cache"
    cache = CheckpointCache(cache_dir=cache_root)
    expected_path = _canonical(cache_root, descriptor).resolve()

    inserted = cache.put(candidate, descriptor)
    loaded = cache.get(descriptor)

    assert inserted == VerifiedCheckpoint(
        path=expected_path,
        descriptor=descriptor,
        sha256=descriptor.sha256,
    )
    assert loaded == inserted
    assert inserted.path.is_file()
    assert inserted.path.read_bytes() == CHECKPOINT_BYTES
    assert inserted.path.name == "checkpoint"
    assert inserted.path.suffix == ""
    assert list(inserted.path.parent.iterdir()) == [inserted.path]
    assert not (cache_root / "v2").exists()


def test_candidate_path_and_filename_do_not_affect_physical_entry(tmp_path):
    first = _candidate(tmp_path / "first", "weights.pt")
    second = _candidate(tmp_path / "second", "model.ckpt")
    descriptor = _descriptor(first)
    cache = CheckpointCache(cache_dir=tmp_path / "cache")

    first_result = cache.put(first, descriptor)
    second_result = cache.put(second, descriptor)

    assert first_result.path == second_result.path
    assert first_result.path.name == "checkpoint"


def test_logical_descriptors_with_same_sha_share_blob_but_preserve_caller_context(tmp_path):
    candidate = _candidate(tmp_path / "input")
    first = _descriptor(
        candidate,
        identifier="checkpoint-a",
        revision="revision-a",
        source="https://mirror-a.example/weights",
    )
    second = _descriptor(
        candidate,
        identifier="checkpoint-b",
        revision="revision-b",
        source="https://mirror-b.example/weights",
    )
    cache = CheckpointCache(cache_dir=tmp_path / "cache")

    first_result = cache.put(candidate, first)
    second_result = cache.get(second)

    assert second_result is not None
    assert first_result.path == second_result.path
    assert first_result.descriptor is first
    assert second_result.descriptor is second


def test_physical_blob_is_shared_while_revision_changes_artifact_cache_identity(tmp_path):
    candidate = _candidate(tmp_path / "input")
    first_checkpoint = _descriptor(candidate, revision="revision-a")
    second_checkpoint = _descriptor(candidate, revision="revision-b")
    first_model = StructuralModelDescriptor(
        architecture="synthetic-model",
        checkpoint=first_checkpoint,
    )
    second_model = StructuralModelDescriptor(
        architecture="synthetic-model",
        checkpoint=second_checkpoint,
    )
    cache = CheckpointCache(cache_dir=tmp_path / "cache")

    first_result = cache.put(candidate, first_checkpoint)
    second_result = cache.get(second_checkpoint)

    assert second_result is not None
    assert first_result.path == second_result.path
    assert _artifact_cache_key(first_model) != _artifact_cache_key(second_model)


def test_different_sha_values_use_different_entry_paths(tmp_path):
    first = _candidate(tmp_path / "input", "first.bin", b"first")
    second = _candidate(tmp_path / "input", "second.bin", b"second")
    first_descriptor = _descriptor(first)
    second_descriptor = _descriptor(second)
    cache = CheckpointCache(cache_dir=tmp_path / "cache")

    first_result = cache.put(first, first_descriptor)
    second_result = cache.put(second, second_descriptor)

    assert first_result.path != second_result.path
    assert first_result.path.parent.name == first_descriptor.sha256
    assert second_result.path.parent.name == second_descriptor.sha256


def test_get_rehashes_and_leaves_corrupt_entry_intact(tmp_path):
    candidate = _candidate(tmp_path / "input")
    descriptor = _descriptor(candidate)
    cache = CheckpointCache(cache_dir=tmp_path / "cache")
    inserted = cache.put(candidate, descriptor)
    inserted.path.write_bytes(b"corrupt")

    assert cache.get(descriptor) is None
    assert inserted.path.is_file()
    assert inserted.path.read_bytes() == b"corrupt"


def test_valid_put_repairs_corrupt_canonical_entry(tmp_path):
    candidate = _candidate(tmp_path / "input")
    descriptor = _descriptor(candidate)
    cache = CheckpointCache(cache_dir=tmp_path / "cache")
    canonical = cache.put(candidate, descriptor).path
    canonical.write_bytes(b"corrupt")

    repaired = cache.put(candidate, descriptor)

    assert repaired.path == canonical
    assert repaired.path.read_bytes() == CHECKPOINT_BYTES
    assert cache.get(descriptor) == repaired


def test_wrong_candidate_is_rejected_without_canonical_or_temp(tmp_path):
    expected = _candidate(tmp_path / "expected", content=b"expected")
    wrong = _candidate(tmp_path / "input", content=b"wrong")
    descriptor = _descriptor(expected)
    cache_root = tmp_path / "cache"
    cache = CheckpointCache(cache_dir=cache_root)
    canonical = _canonical(cache_root, descriptor)

    with pytest.raises(ExecutionError, match="SHA-256 mismatch"):
        cache.put(wrong, descriptor)

    assert not canonical.exists()
    assert _stage_files(canonical.parent) == []


def test_failed_repair_preserves_corrupt_canonical(tmp_path):
    expected = _candidate(tmp_path / "expected", content=b"expected")
    wrong = _candidate(tmp_path / "input", content=b"wrong candidate")
    descriptor = _descriptor(expected)
    cache_root = tmp_path / "cache"
    canonical = _canonical(cache_root, descriptor)
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"existing corruption")
    cache = CheckpointCache(cache_dir=cache_root)

    with pytest.raises(ExecutionError, match="SHA-256 mismatch"):
        cache.put(wrong, descriptor)

    assert canonical.read_bytes() == b"existing corruption"
    assert _stage_files(canonical.parent) == []


def test_valid_hit_short_circuits_missing_candidate_and_copy(tmp_path, monkeypatch):
    candidate = _candidate(tmp_path / "input")
    descriptor = _descriptor(candidate)
    cache = CheckpointCache(cache_dir=tmp_path / "cache")
    existing = cache.put(candidate, descriptor)
    candidate.unlink()
    copy_spy = Mock(side_effect=AssertionError("candidate must not be copied"))
    monkeypatch.setattr(checkpoint_cache_module.shutil, "copyfileobj", copy_spy)

    reused = cache.put(candidate, descriptor)

    assert reused == existing
    copy_spy.assert_not_called()


@pytest.mark.parametrize("candidate_kind", ["missing", "directory"])
def test_invalid_candidate_path_raises_execution_error(tmp_path, candidate_kind):
    expected = _candidate(tmp_path / "expected")
    descriptor = _descriptor(expected)
    candidate = tmp_path / candidate_kind
    if candidate_kind == "directory":
        candidate.mkdir()
    cache = CheckpointCache(cache_dir=tmp_path / "cache")

    with pytest.raises(ExecutionError):
        cache.put(candidate, descriptor)

    assert not (tmp_path / "cache" / "checkpoints").exists()


def test_sha_named_poisoned_entry_is_not_trusted(tmp_path):
    expected = _candidate(tmp_path / "expected")
    descriptor = _descriptor(expected)
    cache_root = tmp_path / "cache"
    canonical = _canonical(cache_root, descriptor)
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"poison")

    assert CheckpointCache(cache_dir=cache_root).get(descriptor) is None
    assert canonical.read_bytes() == b"poison"


def test_operational_cached_read_error_is_wrapped_as_cache_error(tmp_path, monkeypatch):
    expected = _candidate(tmp_path / "expected")
    descriptor = _descriptor(expected)
    cache_root = tmp_path / "cache"
    canonical = _canonical(cache_root, descriptor)
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(CHECKPOINT_BYTES)
    failure = OSError("synthetic cache read failure")

    def fail_verification(path, supplied_descriptor):
        del path, supplied_descriptor
        raise ExecutionError("wrapped read failure") from failure

    monkeypatch.setattr(checkpoint_cache_module, "verify_checkpoint", fail_verification)

    with pytest.raises(CacheError, match="Unable to read cached checkpoint") as captured:
        CheckpointCache(cache_dir=cache_root).get(descriptor)

    assert captured.value.__cause__ is failure


def test_copy_failure_is_cache_error_and_cleans_temp(tmp_path, monkeypatch):
    candidate = _candidate(tmp_path / "input")
    descriptor = _descriptor(candidate)
    cache_root = tmp_path / "cache"
    canonical = _canonical(cache_root, descriptor)
    failure = OSError("synthetic copy failure")
    monkeypatch.setattr(
        checkpoint_cache_module.shutil,
        "copyfileobj",
        Mock(side_effect=failure),
    )

    with pytest.raises(CacheError, match="Unable to stage checkpoint") as captured:
        CheckpointCache(cache_dir=cache_root).put(candidate, descriptor)

    assert captured.value.__cause__ is failure
    assert not canonical.exists()
    assert _stage_files(canonical.parent) == []


def test_cache_directory_creation_failure_is_cache_error(tmp_path):
    candidate = _candidate(tmp_path / "input")
    descriptor = _descriptor(candidate)
    cache_root = tmp_path / "cache"
    cache_root.write_bytes(b"not a directory")

    with pytest.raises(CacheError, match="Unable to stage checkpoint") as captured:
        CheckpointCache(cache_dir=cache_root).put(candidate, descriptor)

    assert isinstance(captured.value.__cause__, OSError)
    assert cache_root.read_bytes() == b"not a directory"


def test_fsync_failure_is_cache_error_and_prevents_replace(tmp_path, monkeypatch):
    candidate = _candidate(tmp_path / "input")
    descriptor = _descriptor(candidate)
    cache_root = tmp_path / "cache"
    canonical = _canonical(cache_root, descriptor)
    failure = OSError("synthetic fsync failure")
    replace_spy = Mock()
    monkeypatch.setattr(checkpoint_cache_module.os, "fsync", Mock(side_effect=failure))
    monkeypatch.setattr(checkpoint_cache_module.os, "replace", replace_spy)

    with pytest.raises(CacheError, match="Unable to stage checkpoint") as captured:
        CheckpointCache(cache_dir=cache_root).put(candidate, descriptor)

    assert captured.value.__cause__ is failure
    replace_spy.assert_not_called()
    assert not canonical.exists()
    assert _stage_files(canonical.parent) == []


def test_replace_failure_preserves_canonical_and_cleans_temp(tmp_path, monkeypatch):
    candidate = _candidate(tmp_path / "input")
    descriptor = _descriptor(candidate)
    cache_root = tmp_path / "cache"
    canonical = _canonical(cache_root, descriptor)
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"existing corruption")
    failure = OSError("synthetic replace failure")
    monkeypatch.setattr(checkpoint_cache_module.os, "replace", Mock(side_effect=failure))

    with pytest.raises(CacheError, match="Unable to install checkpoint") as captured:
        CheckpointCache(cache_dir=cache_root).put(candidate, descriptor)

    assert captured.value.__cause__ is failure
    assert canonical.read_bytes() == b"existing corruption"
    assert _stage_files(canonical.parent) == []


def test_successful_put_fsyncs_before_atomic_replace(tmp_path, monkeypatch):
    candidate = _candidate(tmp_path / "input")
    descriptor = _descriptor(candidate)
    cache = CheckpointCache(cache_dir=tmp_path / "cache")
    events: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def recording_fsync(file_descriptor):
        events.append("fsync")
        return real_fsync(file_descriptor)

    def recording_replace(source, destination):
        assert events == ["fsync"]
        events.append("replace")
        return real_replace(source, destination)

    monkeypatch.setattr(checkpoint_cache_module.os, "fsync", recording_fsync)
    monkeypatch.setattr(checkpoint_cache_module.os, "replace", recording_replace)

    cache.put(candidate, descriptor)

    assert events == ["fsync", "replace"]


def test_default_root_uses_eris_platform_cache_without_creating_it(tmp_path, monkeypatch):
    root = tmp_path / "platform-cache"
    monkeypatch.setattr(checkpoint_cache_module, "user_cache_dir", lambda app: str(root))

    cache = CheckpointCache()
    descriptor = CheckpointDescriptor(identifier="missing", sha256="a" * 64)

    assert cache.cache_dir == root
    assert cache.get(descriptor) is None
    assert not root.exists()


def test_internal_export_remains_offline_and_does_not_load_frameworks():
    script = """
import sys
import eris
from eris.core.learned import CheckpointCache

assert CheckpointCache.__name__ == 'CheckpointCache'
assert not hasattr(eris, 'CheckpointCache')
blocked = {'torch', 'torchdrug', 'torch_geometric', 'transformers', 'requests'}
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise SystemExit(f'optional frameworks loaded: {loaded}')
"""

    subprocess.run([sys.executable, "-c", script], check=True)
