from __future__ import annotations

import json
from typing import Any, cast

import numpy as np
import pytest

import eris.core.cache as cache_module
from eris.core.artifact_schema import (
    build_artifact_metadata,
    build_feature_keys,
    build_provenance,
    compute_config_fingerprint,
)
from eris.core.artifacts import ArtifactValidationError, StructureArtifact
from eris.core.cache import (
    DiskCache,
    compute_cache_key,
    parser_hint_from_source,
    parser_hint_matches_format,
)
from eris.core.hashing import sha256_file
from eris.core.io import StructureLoadError
from eris.core.types import StructureInput
from eris.encoders.clouds.atom_point_cloud import AtomPointCloud
from eris.encoders.clouds.residue_point_cloud import ResiduePointCloudCA
from eris.encoders.graphs.residue_knn_graph import ResidueKNNGraph
from tests.helpers import PDB, input_metadata


def _artifact() -> StructureArtifact:
    data = {
        "pos": np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        "chain_id": np.array(["A", "A"], dtype=np.str_),
    }
    parameters = {"center": True}
    meta = build_artifact_metadata(
        data=data,
        representation_name="synthetic_cloud",
        representation_family="explicit",
        representation_type="cloud",
        granularity="residue",
        coordinate_source="ca",
        encoder={"implementation": "tests.synthetic.Encoder", "version": "1.0"},
        input_metadata=input_metadata(),
        parameters=parameters,
        provenance=build_provenance(
            representation_name="synthetic_cloud",
            parameters=parameters,
            chain_ids=["A"],
        ),
        output_declarations=[
            {
                "name": "pos",
                "role": "coordinates",
                "granularity": "residue",
                "layout": "N3",
            },
            {"name": "chain_id", "role": "chain_id", "granularity": "residue", "layout": "N"},
        ],
        feature_keys=build_feature_keys(),
        counts={"n_nodes": 2},
    )
    return StructureArtifact(data=data, meta=meta)


def _identity(**overrides) -> dict:
    values = {
        "representation_name": "residue_point_cloud",
        "encoder_implementation": "eris.encoders.clouds.ResiduePointCloudCA",
        "encoder_version": "0.1.0",
        "config_fingerprint": "b" * 64,
        "source_sha256": "a" * 64,
        "parser_hint": ".pdb",
        "model_index": 0,
        "chain_ids": ["A"],
        "model": None,
    }
    values.update(overrides)
    return values


def _key(**overrides) -> str:
    return compute_cache_key(**_identity(**overrides))


def _entry(cache_root, key: str):
    return cache_root / "artifacts" / key


def _write_entry(cache_root, key: str, *, meta: dict, data: dict[str, np.ndarray]) -> None:
    entry = _entry(cache_root, key)
    entry.mkdir(parents=True)
    (entry / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    np.savez_compressed(entry / "data.npz", **cast("dict[str, Any]", data))


def test_compute_cache_key_is_deterministic_sha256_hex():
    key = _key()

    assert key == _key()
    assert len(key) == 64
    assert set(key) <= set("0123456789abcdef")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("foo.PDB", ".pdb"),
        ("foo.pdb.gz", ".pdb.gz"),
        ("foo", "<no-suffix>"),
    ],
)
def test_parser_hint_uses_only_normalized_parser_relevant_suffix(source, expected):
    assert parser_hint_from_source(source) == expected


def test_parser_hint_is_path_independent_and_conservative():
    assert parser_hint_from_source("first/place/protein.pdb") == parser_hint_from_source(
        "second/place/protein.PDB"
    )
    assert _key(parser_hint=parser_hint_from_source("first/place/protein.pdb")) == _key(
        parser_hint=parser_hint_from_source("second/place/protein.PDB")
    )
    assert _key(parser_hint=".pdb") != _key(parser_hint=".cif")
    assert _key(parser_hint=".pdb") != _key(parser_hint=".ent")
    assert _key(parser_hint=".cif") != _key(parser_hint=".mmcif")
    assert _key(parser_hint=".pdb.gz") != _key(parser_hint=".pdb")


@pytest.mark.parametrize(
    ("parser_hint", "input_format", "expected"),
    [
        (".cif", "mmcif", True),
        (".pdb", "mmcif", False),
    ],
)
def test_parser_hint_checks_known_confirmed_formats(parser_hint, input_format, expected):
    assert parser_hint_matches_format(parser_hint, input_format) is expected


def test_optional_execution_identity_is_precision_sensitive():
    assert _key(execution={"precision": "fp32"}) != _key(execution={"precision": "fp16"})
    assert _key(execution={"precision": "fp32"}) != _key()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("encoder_version", "0.2.0"),
        ("config_fingerprint", "c" * 64),
        ("chain_ids", ["B"]),
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


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("config_fingerprint", "A" * 64, "config_fingerprint must be a lowercase"),
        ("model_index", -1, "model_index must be a plain integer"),
    ],
)
def test_compute_cache_key_rejects_invalid_identity(field, value, match):
    with pytest.raises(ValueError, match=match):
        _key(**{field: value})


def test_cache_rejects_unsafe_keys(tmp_path):
    cache = DiskCache(tmp_path)
    key = "../" + "a" * 61

    with pytest.raises(ValueError, match="cache key must be"):
        cache.get(key)
    with pytest.raises(ValueError, match="cache key must be"):
        cache.set(key, _artifact())


def test_cache_round_trip_returns_strict_valid_artifact(tmp_path):
    cache = DiskCache(tmp_path)
    key = _key()
    artifact = _artifact()

    cache.set(key, artifact)
    loaded = cache.get(key)

    assert loaded is not None
    assert loaded.meta == artifact.meta
    np.testing.assert_array_equal(loaded.data["pos"], artifact.data["pos"])


@pytest.mark.parametrize(
    "case",
    [
        "bad-json",
        "shape",
        "object",
        "missing-member",
    ],
)
def test_stale_or_corrupt_cache_entry_is_a_non_destructive_miss(tmp_path, case):
    cache = DiskCache(tmp_path)
    key = _key()
    artifact = _artifact()
    entry = _entry(tmp_path, key)

    _write_entry(tmp_path, key, meta=artifact.meta, data=artifact.data)
    if case == "bad-json":
        (entry / "meta.json").write_text("{bad", encoding="utf-8")
    elif case == "shape":
        meta = json.loads((entry / "meta.json").read_text(encoding="utf-8"))
        meta["shapes"]["pos"] = [3, 3]
        (entry / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    elif case == "object":
        np.savez_compressed(
            entry / "data.npz",
            pos=np.array([object(), object()], dtype=object),
            chain_id=artifact.data["chain_id"],
        )
    elif case == "missing-member":
        np.savez_compressed(entry / "data.npz", pos=artifact.data["pos"])

    assert cache.get(key) is None
    assert entry.exists()


def test_invalid_cache_write_raises_without_usable_entry(tmp_path):
    cache = DiskCache(tmp_path)
    key = _key()
    artifact = _artifact()
    artifact.data["pos"] = np.array([object()], dtype=object)

    with pytest.raises(ArtifactValidationError, match="prohibited dtype"):
        cache.set(key, artifact)

    assert cache.get(key) is None
    assert not _entry(tmp_path, key).exists()


class _RecordingDiskCache(DiskCache):
    def __init__(self, cache_dir) -> None:
        super().__init__(cache_dir)
        self.get_keys: list[str] = []
        self.set_keys: list[str] = []

    def get(self, key: str):
        self.get_keys.append(key)
        return super().get(key)

    def set(self, key: str, artifact: StructureArtifact) -> None:
        self.set_keys.append(key)
        super().set(key, artifact)


def test_active_encoder_cache_miss_then_hit_uses_only_api(
    tmp_path,
    monkeypatch,
):
    cache = _RecordingDiskCache(tmp_path)
    inp = StructureInput(source=str(PDB))
    first_encoder = ResidueKNNGraph(use_cache=True, cache=cache)
    first = first_encoder.encode(inp)
    second_encoder = ResidueKNNGraph(use_cache=True, cache=cache)

    def fail_recompute(ctx):  # noqa: ARG001
        raise AssertionError("a valid cache hit must not recompute arrays")

    monkeypatch.setattr(second_encoder, "_create_data", fail_recompute)
    second = second_encoder.encode(inp)

    assert first.meta == second.meta
    assert cache.get_keys == [cache.set_keys[0], cache.set_keys[0]]
    assert cache.set_keys == [cache.get_keys[0]]
    assert set(first.data) == set(second.data)
    for name, expected in first.data.items():
        np.testing.assert_array_equal(second.data[name], expected)


def test_cache_hit_refreshes_current_source_without_rewriting_cached_metadata(
    tmp_path,
    monkeypatch,
):
    source_a = tmp_path / "dir_a" / "protein.pdb"
    source_b = tmp_path / "dir_b" / "protein.pdb"
    source_a.parent.mkdir()
    source_b.parent.mkdir()
    source_bytes = PDB.read_bytes()
    source_a.write_bytes(source_bytes)
    source_b.write_bytes(source_bytes)
    cache = _RecordingDiskCache(tmp_path / "cache")

    first = ResiduePointCloudCA(use_cache=True, cache=cache).encode(StructureInput(source=str(source_a)))
    second_encoder = ResiduePointCloudCA(use_cache=True, cache=cache)

    def fail_recompute(ctx):  # noqa: ARG001
        raise AssertionError("same bytes and suffix must hit the cache")

    monkeypatch.setattr(second_encoder, "_create_data", fail_recompute)
    second = second_encoder.encode(StructureInput(source=str(source_b)))

    assert first.meta["input"]["source"] == str(source_a)
    assert second.meta["input"]["source"] == str(source_b)
    assert second.meta["input"]["format"] == "pdb"
    assert cache.get_keys[0] == cache.get_keys[1]
    cached_meta = json.loads(
        (_entry(cache.cache_dir, cache.get_keys[0]) / "meta.json").read_text(encoding="utf-8")
    )
    assert cached_meta["input"]["source"] == str(source_a)
    for name, expected in first.data.items():
        np.testing.assert_array_equal(second.data[name], expected)


def test_parser_hint_prevents_pdb_cache_from_satisfying_cif_parse(tmp_path):
    pdb_source = tmp_path / "protein.pdb"
    cif_source = tmp_path / "protein.cif"
    source_bytes = PDB.read_bytes()
    pdb_source.write_bytes(source_bytes)
    cif_source.write_bytes(source_bytes)
    cache = _RecordingDiskCache(tmp_path / "cache")

    ResiduePointCloudCA(use_cache=True, cache=cache).encode(StructureInput(source=str(pdb_source)))
    with pytest.raises(StructureLoadError, match="Failed to parse structure"):
        ResiduePointCloudCA(use_cache=True, cache=cache).encode(StructureInput(source=str(cif_source)))

    # Chain resolution parses the structure before cache lookup, so the mislabeled
    # coordinate file fails without consulting the PDB cache entry.
    assert len(cache.get_keys) == 1
    assert len(list((cache.cache_dir / "artifacts").iterdir())) == 1


def test_strict_valid_cache_artifact_with_wrong_identity_is_a_miss(tmp_path):
    cache = _RecordingDiskCache(tmp_path)
    inp = StructureInput(source=str(PDB))
    expected_encoder = ResiduePointCloudCA(use_cache=True, cache=cache)
    parameters = expected_encoder.config.to_dict()
    key = compute_cache_key(
        representation_name=expected_encoder.name,
        encoder_implementation=expected_encoder.encoder_implementation,
        encoder_version=expected_encoder.version,
        config_fingerprint=compute_config_fingerprint(expected_encoder.name, parameters, chain_ids=["A"]),
        source_sha256=sha256_file(PDB),
        parser_hint=".pdb",
        model_index=0,
        chain_ids=["A"],
    )
    wrong = AtomPointCloud(use_cache=False).encode(inp)
    cache.set(key, wrong)
    cache.set_keys.clear()

    result = expected_encoder.encode(inp)

    assert result.meta["representation_name"] == "residue_point_cloud"
    assert cache.get_keys == [key]
    assert cache.set_keys == [key]
    cached = cache.get(key)
    assert cached is not None
    assert cached.meta["representation_name"] == "residue_point_cloud"


def test_use_cache_false_does_not_instantiate_or_compute_cache(monkeypatch):
    def fail_cache_construction():
        raise AssertionError("use_cache=False must not instantiate DiskCache")

    def fail_cache_key(**kwargs):  # noqa: ARG001
        raise AssertionError("use_cache=False must not compute a cache key")

    monkeypatch.setattr(cache_module, "DiskCache", fail_cache_construction)
    monkeypatch.setattr(
        "eris.core.base_encoder.base_encoder.DiskCache",
        fail_cache_construction,
    )
    monkeypatch.setattr(
        "eris.core.base_encoder.base_encoder.compute_cache_key",
        fail_cache_key,
    )

    ResiduePointCloudCA(use_cache=False).encode(StructureInput(source=str(PDB)))
