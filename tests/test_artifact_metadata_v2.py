from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import Mock

import pytest

import eris
from eris.core.artifact_schema_v2 import (
    ArtifactSchemaV2ValidationError,
    build_input_metadata,
    build_provenance,
    compute_config_fingerprint,
)
from eris.core.base_encoder import EncodingContext
from eris.core.hashing import sha256_file
from eris.core.types import StructureInput
from eris.encoders.clouds.residue_point_cloud import ResiduePointCloudCA

DATA = Path("tests/data/1AQ2.pdb")


class _RecordingCache:
    def __init__(self) -> None:
        self.get_keys: list[str] = []
        self.set_keys: list[str] = []

    def get_v2(self, key: str):
        self.get_keys.append(key)
        return None

    def set_v2(self, key: str, artifact) -> None:
        self.set_keys.append(key)

    def get(self, key: str):
        raise AssertionError("active encode must not consult legacy cache")

    def set(self, key: str, data, meta) -> None:
        raise AssertionError("active encode must not write legacy cache")


class _CapturingResiduePointCloud(ResiduePointCloudCA):
    captured_context: EncodingContext | None = None

    def _prepare_data(self, ctx: EncodingContext) -> None:
        self.captured_context = ctx
        super()._prepare_data(ctx)


def _input_metadata(**overrides):
    values = {
        "source": "relative/input.pdb",
        "source_sha256": "a" * 64,
        "input_format": "pdb",
        "model_index": 0,
        "chain_id": None,
    }
    values.update(overrides)
    return build_input_metadata(**values)


def test_sha256_file_matches_exact_source_bytes_and_is_lowercase(tmp_path):
    source = tmp_path / "source.bin"
    content = b"Eris\x00source\r\nbytes"
    source.write_bytes(content)

    digest = sha256_file(source)

    assert digest == hashlib.sha256(content).hexdigest()
    assert len(digest) == 64
    assert digest == digest.lower()


def test_sha256_file_changes_when_source_content_changes(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"first")
    first = sha256_file(source)
    source.write_bytes(b"second")

    assert sha256_file(source) != first


def test_input_builder_preserves_relative_locator_and_exact_five_fields():
    source = Path("relative") / ".." / "input.pdb"

    metadata = _input_metadata(source=source)

    assert metadata == {
        "source": str(source),
        "source_sha256": "a" * 64,
        "format": "pdb",
        "model_index": 0,
        "chain_id": None,
    }
    assert set(metadata) == {"source", "source_sha256", "format", "model_index", "chain_id"}


def test_input_builder_preserves_absolute_string_locator(tmp_path):
    source = str(tmp_path / "input.pdb")

    metadata = _input_metadata(source=source, chain_id="A", model_index=2)

    assert metadata["source"] == source
    assert metadata["chain_id"] == "A"
    assert metadata["model_index"] == 2


def test_input_builder_uses_explicit_format_without_suffix_inference():
    metadata = _input_metadata(source="structure.cif", input_format="parser-confirmed-format")

    assert metadata["format"] == "parser-confirmed-format"


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"source": ""}, "source must be a non-empty string"),
        ({"source_sha256": "A" * 64}, "lowercase 64-character"),
        ({"source_sha256": "a" * 63}, "lowercase 64-character"),
        ({"input_format": ""}, "format must be a non-empty string"),
        ({"model_index": -1}, "integer >= 0"),
        ({"model_index": True}, "integer >= 0"),
        ({"chain_id": 1}, "string or None"),
    ],
)
def test_input_builder_rejects_invalid_contract_values(overrides, match):
    with pytest.raises(ArtifactSchemaV2ValidationError, match=match):
        _input_metadata(**overrides)


def test_provenance_builder_emits_only_required_fields_from_authorities():
    parameters = {"k": 8, "selection": {"chain": None}}

    provenance = build_provenance(
        representation_name="residue_knn_graph",
        parameters=parameters,
    )

    assert provenance == {
        "eris_version": eris.__version__,
        "config_fingerprint": compute_config_fingerprint(
            "residue_knn_graph",
            parameters,
        ),
    }
    assert set(provenance) == {"eris_version", "config_fingerprint"}
    assert "environment" not in provenance
    assert "created_at" not in provenance
    assert "invocation" not in provenance


def test_provenance_builder_is_deterministic_and_parameter_order_independent():
    left = {"center": True, "nested": {"b": 2, "a": 1}}
    right = {"nested": {"a": 1, "b": 2}, "center": True}

    assert build_provenance(
        representation_name="residue_knn_graph", parameters=left
    ) == build_provenance(representation_name="residue_knn_graph", parameters=right)


def test_provenance_fingerprint_changes_with_semantic_configuration():
    baseline = build_provenance(
        representation_name="residue_knn_graph",
        parameters={"k": 8},
    )
    changed = build_provenance(
        representation_name="residue_knn_graph",
        parameters={"k": 9},
    )

    assert baseline["config_fingerprint"] != changed["config_fingerprint"]


@pytest.mark.parametrize("use_cache", [False, True])
def test_encode_hashes_once_reuses_digest_and_threads_context(monkeypatch, use_cache):
    digest = "f" * 64
    cache = _RecordingCache()
    encoder = _CapturingResiduePointCloud(use_cache=use_cache, cache=cache)
    inp = StructureInput(source=str(DATA), chain_id="A", model_index=0)
    hash_spy = Mock(return_value=digest)
    cache_key_spy = Mock(return_value="f" * 64)
    monkeypatch.setattr("eris.core.hashing.sha256_file", hash_spy)
    monkeypatch.setattr(
        "eris.core.base_encoder.base_encoder.compute_cache_key_v2",
        cache_key_spy,
    )

    encoder.encode(inp)

    hash_spy.assert_called_once_with(Path(inp.source))
    if use_cache:
        cache_key_spy.assert_called_once_with(
            representation_name=encoder.name,
            encoder_implementation=encoder.encoder_implementation,
            encoder_version=encoder.version,
            config_fingerprint=compute_config_fingerprint(
                encoder.name,
                encoder.config.to_dict(),
            ),
            source_sha256=digest,
            parser_hint=".pdb",
            model_index=inp.model_index,
            chain_id=inp.chain_id,
            model=None,
        )
    else:
        cache_key_spy.assert_not_called()
    assert encoder.captured_context is not None
    assert encoder.captured_context.source_sha256 == digest
    assert cache.get_keys == (["f" * 64] if use_cache else [])
    assert cache.set_keys == (["f" * 64] if use_cache else [])


def test_current_explicit_encoding_is_complete_schema_v2():
    encoder = ResiduePointCloudCA(use_cache=False)
    artifact = encoder.encode(StructureInput(source=str(DATA), chain_id="A", model_index=0))
    artifact.validate(strict=True)
    encoder_parameters = encoder.config.to_dict()

    assert artifact.meta["schema_version"] == 2
    assert artifact.meta["parameters"] == encoder_parameters
    assert artifact.meta["provenance"]["config_fingerprint"] == (
        compute_config_fingerprint("residue_point_cloud", encoder_parameters)
    )
    assert artifact.meta["input"]["source_sha256"] == sha256_file(DATA)
    assert {"params", "encoder_name", "encoder_fingerprint", "type", "level"}.isdisjoint(
        artifact.meta
    )
