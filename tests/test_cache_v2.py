from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

import eris.core.cache as cache_module
from eris.core.artifact_schema_v2 import (
    ArtifactSchemaV2ValidationError,
    build_artifact_metadata,
    build_feature_keys,
    build_input_metadata,
    build_provenance,
    compute_config_fingerprint,
)
from eris.core.artifacts import ArtifactValidationError, StructureArtifact
from eris.core.cache import DiskCache, compute_cache_key_v2, make_cache_key
from eris.core.hashing import sha256_text
from eris.core.types import StructureInput
from eris.encoders.clouds.residue_point_cloud import ResiduePointCloudCA


def _artifact() -> StructureArtifact:
    data = {"pos": np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)}
    parameters = {"center": True}
    meta = build_artifact_metadata(
        data=data,
        representation_name="synthetic_cloud",
        representation_family="explicit",
        representation_type="cloud",
        granularity="residue",
        coordinate_source="ca",
        encoder={"implementation": "tests.synthetic.Encoder", "version": "1.0"},
        input_metadata=build_input_metadata(
            source="memory://synthetic",
            source_sha256="a" * 64,
            input_format="pdb",
            model_index=0,
            chain_id=None,
        ),
        parameters=parameters,
        provenance=build_provenance(
            representation_name="synthetic_cloud",
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
        counts={"n_nodes": 2},
    )
    return StructureArtifact(data=data, meta=meta)


def _legacy_artifact() -> StructureArtifact:
    artifact = StructureArtifact(
        data={"pos": np.zeros((1, 3), dtype=np.float32)},
        meta={
            "encoder_name": "legacy",
            "encoder_version": "0.1.0",
            "encoder_fingerprint": "legacy-fingerprint",
            "type": "cloud",
            "level": "residue",
            "params": {},
            "input": "source=memory://legacy|chain=None|model=0",
            "feature_keys": {},
        },
    )
    artifact.finalize()
    return artifact


def _identity(**overrides) -> dict:
    values = {
        "representation_name": "residue_point_cloud",
        "encoder_implementation": "eris.encoders.clouds.ResiduePointCloudCA",
        "encoder_version": "0.1.0",
        "config_fingerprint": "b" * 64,
        "source_sha256": "a" * 64,
        "model_index": 0,
        "chain_id": None,
        "model": None,
    }
    values.update(overrides)
    return values


def _key(**overrides) -> str:
    return compute_cache_key_v2(**_identity(**overrides))


def _entry(cache_root, key: str):
    return cache_root / "v2" / key


def _write_entry(cache_root, key: str, *, meta: dict, data: dict[str, np.ndarray]) -> None:
    entry = _entry(cache_root, key)
    entry.mkdir(parents=True)
    (entry / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    np.savez_compressed(entry / "data.npz", **data)


def test_compute_cache_key_v2_is_deterministic_and_has_explicit_inputs():
    first = _key()
    second = _key()
    parameters = inspect.signature(compute_cache_key_v2).parameters

    assert first == second
    assert len(first) == 64
    assert "source" not in parameters
    assert "source_locator" not in parameters
    assert "eris_version" not in parameters
    assert "input_format" not in parameters
    assert "parameters" not in parameters


def test_cache_key_payload_contains_exact_namespaces_and_identity(monkeypatch):
    monkeypatch.setattr(cache_module, "sha256_text", lambda value: value)

    payload = json.loads(_key())

    assert payload == {
        "cache_contract": 2,
        "artifact_schema_version": 2,
        "representation_name": "residue_point_cloud",
        "encoder": {
            "implementation": "eris.encoders.clouds.ResiduePointCloudCA",
            "version": "0.1.0",
        },
        "config_fingerprint": "b" * 64,
        "source_sha256": "a" * 64,
        "input_selection": {"model_index": 0, "chain_id": None},
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("representation_name", "other_representation"),
        ("encoder_implementation", "eris.encoders.OtherEncoder"),
        ("encoder_version", "0.2.0"),
        ("config_fingerprint", "c" * 64),
        ("source_sha256", "d" * 64),
        ("chain_id", "A"),
        ("model_index", 1),
        (
            "model",
            {
                "architecture": "synthetic-model",
                "checkpoint": {"identifier": "checkpoint-a"},
            },
        ),
    ],
)
def test_each_cache_identity_component_changes_key(field, value):
    assert _key(**{field: value}) != _key()


def test_equivalent_model_mapping_order_produces_same_key():
    left = {
        "architecture": "synthetic-model",
        "checkpoint": {"identifier": "checkpoint-a", "revision": "v1"},
    }
    right = {
        "checkpoint": {"revision": "v1", "identifier": "checkpoint-a"},
        "architecture": "synthetic-model",
    }

    assert _key(model=left) == _key(model=right)
    assert _key(model=left) != _key(
        model={
            "architecture": "synthetic-model",
            "checkpoint": {"identifier": "checkpoint-b", "revision": "v1"},
        }
    )


def test_parameter_mapping_order_is_abstracted_by_config_fingerprint():
    left = compute_config_fingerprint("residue_point_cloud", {"a": 1, "b": 2})
    right = compute_config_fingerprint("residue_point_cloud", {"b": 2, "a": 1})

    assert left == right
    assert _key(config_fingerprint=left) == _key(config_fingerprint=right)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("representation_name", ""),
        ("encoder_implementation", "not-dotted"),
        ("encoder_version", ""),
        ("config_fingerprint", "A" * 64),
        ("source_sha256", "short"),
        ("model_index", True),
        ("model_index", -1),
        ("chain_id", 1),
        ("model", []),
    ],
)
def test_compute_cache_key_v2_rejects_invalid_identity(field, value):
    with pytest.raises((ValueError, ArtifactSchemaV2ValidationError)):
        _key(**{field: value})


@pytest.mark.parametrize("key", ["short", "A" * 64, "../" + "a" * 61])
def test_v2_cache_rejects_unsafe_or_noncanonical_keys(tmp_path, key):
    cache = DiskCache(tmp_path)

    with pytest.raises(ValueError):
        cache.get_v2(key)
    with pytest.raises(ValueError):
        cache.set_v2(key, _artifact())


def test_v2_cache_namespace_is_isolated_from_legacy_flat_files(tmp_path):
    cache = DiskCache(tmp_path)
    key = _key()
    legacy = _legacy_artifact()

    cache.set(key, legacy.data, legacy.meta)

    assert (tmp_path / f"{key}.npz").is_file()
    assert (tmp_path / f"{key}.json").is_file()
    assert cache.get_v2(key) is None
    assert not _entry(tmp_path, key).exists()


def test_v2_cache_round_trip_returns_strict_valid_artifact(tmp_path):
    cache = DiskCache(tmp_path)
    key = _key()
    artifact = _artifact()

    cache.set_v2(key, artifact)
    loaded = cache.get_v2(key)

    assert loaded is not None
    assert loaded.meta == artifact.meta
    assert loaded.meta["schema_version"] == 2
    np.testing.assert_array_equal(loaded.data["pos"], artifact.data["pos"])
    assert (_entry(tmp_path, key) / "data.npz").is_file()
    assert (_entry(tmp_path, key) / "meta.json").is_file()


@pytest.mark.parametrize(
    "case",
    [
        "v1",
        "versionless",
        "future",
        "bad-json",
        "bad-npz",
        "missing-file",
        "shape",
        "dtype",
        "feature",
        "count",
        "object",
    ],
)
def test_stale_or_corrupt_v2_cache_entry_is_a_non_destructive_miss(tmp_path, case):
    cache = DiskCache(tmp_path)
    key = _key()
    artifact = _artifact()
    entry = _entry(tmp_path, key)

    if case == "v1":
        _legacy_artifact().save(entry)
    elif case == "versionless":
        legacy = _legacy_artifact()
        del legacy.meta["schema_version"]
        _write_entry(tmp_path, key, meta=legacy.meta, data=legacy.data)
    else:
        _write_entry(tmp_path, key, meta=artifact.meta, data=artifact.data)
        if case == "future":
            meta = dict(artifact.meta)
            meta["schema_version"] = 3
            (entry / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        elif case == "bad-json":
            (entry / "meta.json").write_text("{bad", encoding="utf-8")
        elif case == "bad-npz":
            (entry / "data.npz").write_bytes(b"bad npz")
        elif case == "missing-file":
            (entry / "meta.json").unlink()
        elif case in {"shape", "dtype", "feature", "count"}:
            meta = json.loads((entry / "meta.json").read_text(encoding="utf-8"))
            if case == "shape":
                meta["shapes"]["pos"] = [3, 3]
            elif case == "dtype":
                meta["dtypes"]["pos"] = "float64"
            elif case == "feature":
                meta["feature_keys"]["node_attr_keys"] = ["unexpected"]
            else:
                meta["counts"]["n_nodes"] = 99
            (entry / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        elif case == "object":
            np.savez_compressed(
                entry / "data.npz",
                pos=np.array([object()], dtype=object),
            )

    assert cache.get_v2(key) is None
    assert entry.exists()


@pytest.mark.parametrize("case", ["v1", "invalid-v2", "unsafe-v2"])
def test_invalid_v2_cache_write_raises_without_usable_entry(tmp_path, case):
    cache = DiskCache(tmp_path)
    key = _key()
    artifact = _legacy_artifact() if case == "v1" else _artifact()
    if case == "invalid-v2":
        artifact.meta["counts"]["n_nodes"] = 99
    elif case == "unsafe-v2":
        artifact.data["pos"] = np.array([object()], dtype=object)

    with pytest.raises(ArtifactValidationError):
        cache.set_v2(key, artifact)

    assert cache.get_v2(key) is None
    assert not _entry(tmp_path, key).exists()


def test_legacy_cache_get_set_and_key_behavior_remain_unchanged(tmp_path):
    cache = DiskCache(tmp_path)
    data = {"value": np.array([1, 2], dtype=np.int64)}
    meta = {"legacy": True}
    key = make_cache_key("one", "two")

    cache.set(key, data, meta)
    loaded = cache.get(key)

    assert key == sha256_text("one|two")
    assert loaded is not None
    loaded_data, loaded_meta = loaded
    np.testing.assert_array_equal(loaded_data["value"], data["value"])
    assert loaded_meta == meta
    assert (tmp_path / f"{key}.npz").is_file()
    assert (tmp_path / f"{key}.json").is_file()


class _LegacyRecordingCache:
    def __init__(self) -> None:
        self.get_keys: list[str] = []

    def get(self, key: str):
        self.get_keys.append(key)
        return None


def test_base_encoder_still_uses_legacy_cache_api():
    cache = _LegacyRecordingCache()
    encoder = ResiduePointCloudCA(use_cache=True, cache=cache)
    source_sha256 = "f" * 64
    inp = StructureInput(source="synthetic.pdb", chain_id="A", model_index=1)

    key, returned_cache, artifact = encoder._check_cache(inp, source_sha256)

    assert key == make_cache_key(
        encoder.name,
        encoder.fingerprint(),
        source_sha256,
        inp.to_string(),
    )
    assert cache.get_keys == [key]
    assert returned_cache is cache
    assert artifact is None
