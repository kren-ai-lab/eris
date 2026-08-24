from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from eris.core.artifact_schema_v2 import compute_config_fingerprint, validate_schema_v2
from eris.core.cache import compute_cache_key_v2
from eris.core.hashing import sha256_file
from eris.core.learned import (
    CheckpointDescriptor,
    StructuralModelDescriptor,
    VerifiedCheckpoint,
    verify_checkpoint,
)
from eris.exceptions import ConfigError, ExecutionError


def _checkpoint(**overrides: object) -> CheckpointDescriptor:
    values = {
        "identifier": "synthetic-checkpoint",
        "sha256": "a" * 64,
        "revision": "revision-1",
        "source": "https://models.example/checkpoint.bin",
    }
    values.update(overrides)
    return CheckpointDescriptor(**values)


def _model(**overrides: object) -> StructuralModelDescriptor:
    values = {
        "architecture": "synthetic-structural-model",
        "checkpoint": _checkpoint(),
    }
    values.update(overrides)
    return StructuralModelDescriptor(**values)


def _cache_key(model: StructuralModelDescriptor) -> str:
    return compute_cache_key_v2(
        representation_name="synthetic_structure_embedding",
        encoder_implementation="tests.fake_backend.FakeStructuralEncoder",
        encoder_version="1.0",
        config_fingerprint="b" * 64,
        source_sha256="c" * 64,
        parser_hint=".pdb",
        model_index=0,
        chain_id=None,
        model=model.to_cache_identity(),
    )


def _learned_artifact_metadata(model: StructuralModelDescriptor) -> tuple[dict, dict]:
    data = {"global_embedding": np.array([0.1, 0.2], dtype=np.float32)}
    parameters = {"output_mode": "global"}
    metadata = {
        "schema_version": 2,
        "representation_name": "synthetic_structure_embedding",
        "representation_family": "learned",
        "representation_type": "embedding",
        "granularity": "structure",
        "coordinate_source": "model_specific",
        "encoder": {
            "implementation": "tests.fake_backend.FakeStructuralEncoder",
            "version": "1.0",
        },
        "model": model.to_dict(),
        "input": {
            "source": "memory://synthetic-structure",
            "source_sha256": "c" * 64,
            "format": "pdb",
            "model_index": 0,
            "chain_id": None,
        },
        "parameters": parameters,
        "provenance": {
            "eris_version": "0.1.0",
            "config_fingerprint": compute_config_fingerprint(
                "synthetic_structure_embedding",
                parameters,
            ),
        },
        "outputs": [
            {
                "name": "global_embedding",
                "role": "global_embedding",
                "granularity": "structure",
                "layout": "D",
            }
        ],
        "feature_keys": {},
        "shapes": {"global_embedding": [2]},
        "dtypes": {"global_embedding": "float32"},
        "counts": {},
    }
    return metadata, data


def test_checkpoint_and_model_descriptors_are_frozen():
    checkpoint = _checkpoint()
    model = _model(checkpoint=checkpoint)

    with pytest.raises(FrozenInstanceError):
        checkpoint.identifier = "changed"
    with pytest.raises(FrozenInstanceError):
        model.architecture = "changed"


def test_minimal_descriptor_omits_optional_none_values():
    checkpoint = _checkpoint(revision=None, source=None)
    model = _model(checkpoint=checkpoint)

    assert checkpoint.to_dict() == {
        "identifier": "synthetic-checkpoint",
        "sha256": "a" * 64,
    }
    assert model.to_dict() == {
        "architecture": "synthetic-structural-model",
        "checkpoint": {
            "identifier": "synthetic-checkpoint",
            "sha256": "a" * 64,
        },
    }


def test_full_metadata_and_cache_identity_are_deterministic_and_distinct():
    model = _model()

    assert model.to_dict() == model.to_dict()
    assert model.to_dict()["checkpoint"] == {
        "identifier": "synthetic-checkpoint",
        "sha256": "a" * 64,
        "revision": "revision-1",
        "source": "https://models.example/checkpoint.bin",
    }
    assert model.to_cache_identity() == {
        "architecture": "synthetic-structural-model",
        "checkpoint": {
            "identifier": "synthetic-checkpoint",
            "revision": "revision-1",
            "sha256": "a" * 64,
        },
    }


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        ("identifier", "", "identifier"),
        ("identifier", "   ", "identifier"),
        ("identifier", " checkpoint ", "surrounding whitespace"),
        ("identifier", Path("checkpoint.bin"), "identifier must be a string"),
        ("sha256", None, "lowercase 64-character"),
        ("sha256", "a" * 63, "lowercase 64-character"),
        ("sha256", "g" * 64, "lowercase 64-character"),
        ("sha256", "A" * 64, "lowercase 64-character"),
        ("revision", "", "revision"),
        ("revision", 1, "revision must be a string"),
        ("revision", " revision ", "surrounding whitespace"),
        ("source", "", "source"),
        ("source", Path("checkpoint.bin"), "source must be a string"),
        ("source", " source ", "surrounding whitespace"),
    ],
)
def test_checkpoint_descriptor_rejects_invalid_values(field_name, value, match):
    with pytest.raises(ConfigError, match=match):
        _checkpoint(**{field_name: value})


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        ("architecture", "", "architecture"),
        ("architecture", None, "architecture must be a string"),
        ("architecture", " architecture ", "surrounding whitespace"),
        ("checkpoint", object(), "CheckpointDescriptor"),
    ],
)
def test_model_descriptor_rejects_invalid_values(field_name, value, match):
    with pytest.raises(ConfigError, match=match):
        _model(**{field_name: value})


def test_model_metadata_is_strict_schema_v2_compatible():
    metadata, data = _learned_artifact_metadata(_model())

    validated = validate_schema_v2(metadata, data, strict=True)

    assert validated["model"] == metadata["model"]


def test_source_mirror_changes_metadata_but_not_cache_identity_or_key():
    mirror_a = _model(checkpoint=_checkpoint(source="https://mirror-a.example/checkpoint.bin"))
    mirror_b = _model(checkpoint=_checkpoint(source="https://mirror-b.example/checkpoint.bin"))

    assert mirror_a.to_dict() != mirror_b.to_dict()
    assert mirror_a.to_cache_identity() == mirror_b.to_cache_identity()
    assert _cache_key(mirror_a) == _cache_key(mirror_b)


@pytest.mark.parametrize(
    "changed",
    [
        _model(architecture="other-architecture"),
        _model(checkpoint=_checkpoint(identifier="other-checkpoint")),
        _model(checkpoint=_checkpoint(revision="revision-2")),
        _model(checkpoint=_checkpoint(sha256="b" * 64)),
    ],
)
def test_scientific_model_identity_components_change_cache_key(changed):
    assert changed.to_cache_identity() != _model().to_cache_identity()
    assert _cache_key(changed) != _cache_key(_model())


def test_correct_checkpoint_hashes_once_and_returns_resolved_result(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.bin"
    path.write_bytes(b"fake checkpoint")
    expected_sha256 = sha256_file(path)
    descriptor = _checkpoint(sha256=expected_sha256)
    hash_spy = Mock(return_value=expected_sha256)
    monkeypatch.setattr("eris.core.learned.checkpoints.sha256_file", hash_spy)

    verified = verify_checkpoint(str(path), descriptor)

    hash_spy.assert_called_once_with(path)
    assert verified == VerifiedCheckpoint(
        path=path.resolve(),
        descriptor=descriptor,
        sha256=expected_sha256,
    )
    with pytest.raises(FrozenInstanceError):
        verified.sha256 = "b" * 64


def test_one_byte_checkpoint_mutation_is_rejected(tmp_path):
    path = tmp_path / "checkpoint.bin"
    path.write_bytes(b"fake checkpoint")
    descriptor = _checkpoint(sha256=sha256_file(path))
    path.write_bytes(b"fake checkpoinu")

    with pytest.raises(ExecutionError, match="SHA-256 mismatch"):
        verify_checkpoint(path, descriptor)


def test_wrong_expected_sha_reports_complete_mismatch_context(tmp_path):
    path = tmp_path / "checkpoint.bin"
    path.write_bytes(b"fake checkpoint")
    actual = sha256_file(path)
    expected = "f" * 64
    descriptor = _checkpoint(identifier="context-checkpoint", sha256=expected)

    with pytest.raises(ExecutionError) as captured:
        verify_checkpoint(path, descriptor)

    message = str(captured.value)
    assert "context-checkpoint" in message
    assert str(path) in message
    assert expected in message
    assert actual in message


def test_missing_and_non_file_checkpoint_paths_are_rejected(tmp_path):
    descriptor = _checkpoint()

    with pytest.raises(ExecutionError, match="does not exist"):
        verify_checkpoint(tmp_path / "missing.bin", descriptor)
    with pytest.raises(ExecutionError, match="not a regular file"):
        verify_checkpoint(tmp_path, descriptor)


def test_checkpoint_read_oserror_is_wrapped_with_cause(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.bin"
    path.write_bytes(b"fake checkpoint")
    failure = OSError("synthetic read failure")
    monkeypatch.setattr(
        "eris.core.learned.checkpoints.sha256_file",
        Mock(side_effect=failure),
    )

    with pytest.raises(ExecutionError, match="Unable to read checkpoint") as captured:
        verify_checkpoint(path, _checkpoint())

    assert captured.value.__cause__ is failure


def test_same_bytes_and_descriptor_verify_at_distinct_paths_without_changing_identity(tmp_path):
    first = tmp_path / "first" / "checkpoint.bin"
    second = tmp_path / "second" / "checkpoint.bin"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"fake checkpoint")
    second.write_bytes(b"fake checkpoint")
    descriptor = _checkpoint(sha256=sha256_file(first))
    model = _model(checkpoint=descriptor)

    first_verified = verify_checkpoint(first, descriptor)
    second_verified = verify_checkpoint(second, descriptor)

    assert first_verified.path != second_verified.path
    assert first_verified.descriptor is second_verified.descriptor
    assert first_verified.sha256 == second_verified.sha256
    first_model = _model(checkpoint=first_verified.descriptor)
    second_model = _model(checkpoint=second_verified.descriptor)
    assert first_model.to_cache_identity() == second_model.to_cache_identity()
    assert _cache_key(first_model) == _cache_key(second_model) == _cache_key(model)
