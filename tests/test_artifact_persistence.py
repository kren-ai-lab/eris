from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import eris.core.artifacts as artifacts_module
from eris.core.artifact_schema import (
    ArtifactValidationError,
    build_artifact_metadata,
    build_feature_keys,
    build_input_metadata,
    build_provenance,
)
from eris.core.artifacts import ArtifactValidationError, StructureArtifact
from eris.exceptions import ConfigError
from eris.pooling import pool


def _input_metadata(digest: str = "a") -> dict:
    return build_input_metadata(
        source="memory://synthetic",
        source_sha256=digest * 64,
        input_format="pdb",
        model_index=0,
        chain_ids=["A"],
    )


def _explicit_artifact() -> StructureArtifact:
    data = {
        "edge_index": np.array([[0, 1], [1, 0]], dtype=np.int64),
        "pos": np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        "node_attr": np.array([[1, 0], [0, 1]], dtype=np.float32),
        "chain_id": np.array(["A", "A"], dtype=np.str_),
    }
    parameters = {"features": ["mass", "charge"]}
    meta = build_artifact_metadata(
        data=data,
        representation_name="synthetic_explicit_graph",
        representation_family="explicit",
        representation_type="graph",
        granularity="residue",
        coordinate_source="ca",
        encoder={
            "implementation": "tests.synthetic.ExplicitEncoder",
            "version": "1.0",
        },
        input_metadata=_input_metadata(),
        parameters=parameters,
        provenance=build_provenance(
            representation_name="synthetic_explicit_graph",
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
            {
                "name": "node_attr",
                "role": "node_features",
                "granularity": "residue",
                "layout": "NF",
            },
            {"name": "edge_index", "role": "edge_index", "layout": "2E"},
            {"name": "chain_id", "role": "chain_id", "granularity": "residue", "layout": "N"},
        ],
        feature_keys=build_feature_keys(node_attr_keys=["mass", "charge"]),
        counts={"n_nodes": 2, "n_edges": 2, "n_self_loops": 0},
        extensions={"tests.schema": {"reviewed": True}},
    )
    return StructureArtifact(data=data, meta=meta)


def _learned_artifact() -> StructureArtifact:
    data = {
        "valid_mask": np.array([True, False], dtype=np.bool_),
        "node_ids": np.array(["A:1", "A:2"], dtype=np.str_),
        "chain_id": np.array(["A", "A"], dtype=np.str_),
        "global_embedding": np.array([0.2, 0.3], dtype=np.float32),
        "node_embedding": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
    }
    parameters = {"pooling": "mean"}
    meta = build_artifact_metadata(
        data=data,
        representation_name="synthetic_learned_embedding",
        representation_family="learned",
        representation_type="embedding",
        granularity="mixed",
        coordinate_source="model_specific",
        encoder={
            "implementation": "tests.synthetic.LearnedEncoder",
            "version": "1.0",
        },
        input_metadata=_input_metadata("b"),
        parameters=parameters,
        provenance=build_provenance(
            representation_name="synthetic_learned_embedding",
            parameters=parameters,
            chain_ids=["A"],
        ),
        output_declarations=[
            {"name": "node_embedding", "role": "node_embedding", "layout": "ND"},
            {
                "name": "global_embedding",
                "role": "global_embedding",
                "layout": "D",
            },
            {
                "name": "node_ids",
                "role": "node_ids",
                "granularity": "residue",
                "layout": "N",
            },
            {"name": "chain_id", "role": "chain_id", "granularity": "residue", "layout": "N"},
            {
                "name": "valid_mask",
                "role": "valid_mask",
                "granularity": "residue",
                "layout": "N",
            },
        ],
        feature_keys=build_feature_keys(),
        counts={"n_nodes": 2},
        model={
            "architecture": "synthetic-structural-model",
            "checkpoint": {
                "identifier": "synthetic-checkpoint",
                "revision": "v1",
                "source": "memory://synthetic-checkpoint",
                "sha256": "c" * 64,
            },
        },
    )
    return StructureArtifact(data=data, meta=meta)


def _assert_data_unchanged(
    artifact: StructureArtifact,
    before: dict[str, np.ndarray],
) -> None:
    assert list(artifact.data) == list(before)
    for name, expected in before.items():
        np.testing.assert_array_equal(artifact.data[name], expected)


@pytest.mark.parametrize(
    ("mode", "expected", "expected_keys"),
    [
        ("mean", [2.0, np.nan], ["pool.mean.mass", "pool.mean.charge"]),
        ("sum", [4.0, np.nan], ["pool.sum.mass", "pool.sum.charge"]),
        ("max", [3.0, np.nan], ["pool.max.mass", "pool.max.charge"]),
        (
            "mean_std",
            [2.0, np.nan, 1.0, np.nan],
            [
                "pool.mean.mass",
                "pool.mean.charge",
                "pool.std.mass",
                "pool.std.charge",
            ],
        ),
    ],
)
def test_residue_pool_math_nan_omit_and_schema_vector(mode, expected, expected_keys):
    source = _explicit_artifact()
    source.data["node_attr"] = np.asarray([[1.0, np.nan], [3.0, np.nan]], dtype=np.float32)
    meta_before = deepcopy(source.meta)
    data_before = {name: array.copy() for name, array in source.data.items()}

    pooled = pool(source, mode=mode)

    np.testing.assert_equal(pooled.data["pooled_attr"], np.asarray(expected, dtype=np.float64))
    assert pooled.meta["representation_name"] == "residue_pool"
    assert pooled.meta["representation_family"] == "explicit"
    assert pooled.meta["representation_type"] == "vector"
    assert pooled.meta["granularity"] == "structure"
    assert pooled.meta["feature_keys"]["pooled_attr_keys"] == expected_keys
    assert pooled.meta["outputs"] == [
        {
            "name": "pooled_attr",
            "role": "pooled_features",
            "granularity": "structure",
            "layout": "F",
        }
    ]
    pooled.validate(strict=True)
    assert source.meta == meta_before
    _assert_data_unchanged(source, data_before)


def test_residue_pool_nan_policies_round_trip_and_fingerprints(tmp_path: Path):
    source = _explicit_artifact()
    source.data["node_attr"] = np.asarray([[1.0, np.nan], [3.0, np.nan]], dtype=np.float32)

    omit = pool(source, mode="mean", nan_policy="omit")
    repeat = pool(source, mode="mean", nan_policy="omit")
    propagate = pool(source, mode="mean", nan_policy="propagate")
    summed = pool(source, mode="sum", nan_policy="omit")

    assert (
        omit.meta["provenance"]["config_fingerprint"]
        == repeat.meta["provenance"]["config_fingerprint"]
    )
    assert (
        omit.meta["parameters"]["source_artifact_fingerprint"]
        == repeat.meta["parameters"]["source_artifact_fingerprint"]
    )
    assert (
        omit.meta["provenance"]["config_fingerprint"]
        != propagate.meta["provenance"]["config_fingerprint"]
    )
    assert (
        omit.meta["provenance"]["config_fingerprint"]
        != summed.meta["provenance"]["config_fingerprint"]
    )
    reordered = _explicit_artifact()
    reordered.data["node_attr"] = source.data["node_attr"].copy()
    reordered.meta["feature_keys"]["node_attr_keys"] = ["charge", "mass"]
    reordered_pool = pool(reordered)
    assert (
        reordered_pool.meta["parameters"]["source_artifact_fingerprint"]
        != omit.meta["parameters"]["source_artifact_fingerprint"]
    )
    np.testing.assert_equal(propagate.data["pooled_attr"], [2.0, np.nan])
    with pytest.raises(ConfigError, match="nan_policy='error'"):
        pool(source, nan_policy="error")

    source.data["node_attr"][0, 0] = 2.0
    changed = pool(source)
    assert (
        changed.meta["parameters"]["source_artifact_fingerprint"]
        != omit.meta["parameters"]["source_artifact_fingerprint"]
    )

    source_target = tmp_path / "source"
    source.save(source_target)
    from_path = pool(source_target, output=tmp_path / "pooled-from-path")
    assert from_path.meta == changed.meta
    np.testing.assert_equal(from_path.data["pooled_attr"], changed.data["pooled_attr"])

    target = tmp_path / "pooled"
    omit.save(target)
    loaded = StructureArtifact.load(target, strict=True)
    assert loaded.meta == omit.meta
    np.testing.assert_equal(loaded.data["pooled_attr"], omit.data["pooled_attr"])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"source": "missing"}, "not present"),
        ({"mode": "median"}, "mode must be one of"),
        ({"nan_policy": "zero"}, "nan_policy must be one of"),
    ],
)
def test_residue_pool_rejects_invalid_requests(kwargs, match):
    with pytest.raises(ConfigError, match=match):
        pool(_explicit_artifact(), **kwargs)


def test_residue_pool_rejects_incompatible_granularity_and_malformed_source():
    learned = _learned_artifact()
    with pytest.raises(ConfigError, match="residue granularity"):
        pool(learned, source="node_embedding")

    malformed = _explicit_artifact()
    malformed.data["node_attr"] = malformed.data["node_attr"].reshape(-1)
    with pytest.raises(ArtifactValidationError, match="shape|dimensions"):
        pool(malformed)

    missing_keys = _explicit_artifact()
    missing_keys.meta["feature_keys"] = {}
    with pytest.raises(ConfigError, match="one ordered key"):
        pool(missing_keys)

    infinite = _explicit_artifact()
    infinite.data["node_attr"][0, 0] = np.inf
    with pytest.raises(ConfigError, match="infinities"):
        pool(infinite)

    empty = _explicit_artifact()
    empty.data["pos"] = np.zeros((0, 3), dtype=np.float32)
    empty.data["node_attr"] = np.zeros((0, 2), dtype=np.float32)
    empty.data["edge_index"] = np.zeros((2, 0), dtype=np.int64)
    empty.data["chain_id"] = np.asarray([], dtype=np.str_)
    empty.meta["shapes"] = {name: list(array.shape) for name, array in empty.data.items()}
    empty.meta["dtypes"] = {name: str(array.dtype) for name, array in empty.data.items()}
    empty.meta["counts"] = {"n_nodes": 0, "n_edges": 0, "n_self_loops": 0}
    with pytest.raises(ConfigError, match="N,D > 0"):
        pool(empty)


def test_explicit_finalize_is_idempotent_and_save_preserves_contract(tmp_path):
    artifact = _explicit_artifact()
    meta_before = deepcopy(artifact.meta)
    data_before = {name: array.copy() for name, array in artifact.data.items()}

    artifact.validate()
    assert artifact.meta == meta_before
    artifact.finalize()
    meta_after_first = deepcopy(artifact.meta)
    artifact.finalize()

    assert artifact.meta == meta_after_first == meta_before
    _assert_data_unchanged(artifact, data_before)

    target = tmp_path / "explicit"
    artifact.save(target)

    assert {path.name for path in target.iterdir()} == {"data.npz", "meta.json"}
    serialized = (target / "meta.json").read_text(encoding="utf-8")
    assert not serialized.endswith("\n")
    stored_meta = json.loads(serialized)
    for field in (
        "schema_version",
        "input",
        "outputs",
        "feature_keys",
        "counts",
        "shapes",
        "dtypes",
        "extensions",
    ):
        assert stored_meta[field] == artifact.meta[field]
    assert stored_meta["input"]["source_sha256"] == "a" * 64

    expected_order = [output["name"] for output in artifact.meta["outputs"]]
    with np.load(target / "data.npz", allow_pickle=False) as stored_data:
        assert stored_data.files == expected_order
        for name in expected_order:
            np.testing.assert_array_equal(stored_data[name], artifact.data[name])


def test_learned_validate_finalize_and_save(tmp_path):
    artifact = _learned_artifact()
    data_before = {name: array.copy() for name, array in artifact.data.items()}

    artifact.validate(strict=False)
    artifact.finalize()
    artifact.save(tmp_path / "learned")

    stored_meta = json.loads((tmp_path / "learned" / "meta.json").read_text(encoding="utf-8"))
    assert stored_meta["schema_version"] == 1
    assert stored_meta["representation_family"] == "learned"
    assert stored_meta["representation_type"] == "embedding"
    assert stored_meta["coordinate_source"] == "model_specific"
    assert stored_meta["model"]["checkpoint"]["identifier"] == "synthetic-checkpoint"
    assert [output["name"] for output in stored_meta["outputs"]] == [
        "node_embedding",
        "global_embedding",
        "node_ids",
        "chain_id",
        "valid_mask",
    ]
    _assert_data_unchanged(artifact, data_before)


def test_valid_finalize_canonicalizes_only_after_validation():
    artifact = _explicit_artifact()
    artifact.meta["extensions"]["tests.schema"]["labels"] = ("first", "second")
    original_meta = artifact.meta
    data_before = {name: array.copy() for name, array in artifact.data.items()}

    artifact.validate()
    assert artifact.meta is original_meta
    assert artifact.meta["extensions"]["tests.schema"]["labels"] == (
        "first",
        "second",
    )

    artifact.finalize()
    assert artifact.meta is not original_meta
    assert artifact.meta["extensions"]["tests.schema"]["labels"] == [
        "first",
        "second",
    ]
    first = deepcopy(artifact.meta)
    artifact.finalize()
    assert artifact.meta == first
    _assert_data_unchanged(artifact, data_before)


@pytest.mark.parametrize(
    "unsafe",
    [
        np.array([object()], dtype=object),
        np.array([1 + 2j], dtype=np.complex64),
    ],
    ids=["object", "complex"],
)
def test_unsafe_payload_fails_before_filesystem_mutation(tmp_path, unsafe):
    artifact = _explicit_artifact()
    artifact.data["node_attr"] = unsafe
    meta_before = deepcopy(artifact.meta)
    target = tmp_path / "unsafe"

    with pytest.raises(ArtifactValidationError) as raised:
        artifact.save(target)

    assert artifact.meta == meta_before
    assert artifact.data["node_attr"] is unsafe
    assert not target.exists()


@pytest.mark.parametrize("bad_value", [float("nan")], ids=["nan"])
def test_non_finite_metadata_fails_without_mutation(tmp_path, bad_value):
    artifact = _explicit_artifact()
    artifact.meta["extensions"]["tests.schema"]["non_finite"] = bad_value
    original_meta = artifact.meta
    data_before = {name: array.copy() for name, array in artifact.data.items()}
    target = tmp_path / "non-finite"

    with pytest.raises(ArtifactValidationError) as raised:
        artifact.save(target)

    assert artifact.meta is original_meta
    assert artifact.meta["extensions"]["tests.schema"]["non_finite"] is bad_value
    _assert_data_unchanged(artifact, data_before)
    assert not target.exists()


def test_save_enforces_strict_validation(tmp_path):
    artifact = _explicit_artifact()
    artifact.meta["counts"]["n_nodes"] = 99
    meta_before = deepcopy(artifact.meta)
    target = tmp_path / "invalid-counts"

    with pytest.raises(ArtifactValidationError, match="n_nodes"):
        artifact.save(target)

    assert artifact.meta == meta_before
    assert not target.exists()


@pytest.mark.parametrize("operation", ["finalize"])
def test_invalid_operations_do_not_mutate_artifact(tmp_path, operation):
    artifact = _explicit_artifact()
    artifact.meta["counts"]["n_nodes"] = 99
    meta_before = deepcopy(artifact.meta)
    data_before = {name: array.copy() for name, array in artifact.data.items()}
    target = tmp_path / "invalid"

    with pytest.raises(ArtifactValidationError, match="n_nodes"):
        if operation == "validate":
            artifact.validate(strict=False)
        elif operation == "finalize":
            artifact.finalize()
        else:
            artifact.save(target)

    assert artifact.meta == meta_before
    _assert_data_unchanged(artifact, data_before)
    assert not target.exists()


@pytest.mark.parametrize(
    "version",
    [True, "2", 3],
    ids=["bool", "string", "future"],
)
@pytest.mark.parametrize("operation", ["validate"])
def test_invalid_schema_versions_reject_without_mutation(
    tmp_path,
    version,
    operation,
):
    meta = {"schema_version": version}
    artifact = StructureArtifact(data={}, meta=meta)
    target = tmp_path / "invalid-version"

    with pytest.raises(ArtifactValidationError):
        if operation == "save":
            artifact.save(target)
        else:
            getattr(artifact, operation)()

    assert artifact.meta is meta
    assert artifact.data == {}
    assert not target.exists()


def test_versionless_metadata_is_rejected_by_the_schema_boundary():
    artifact = StructureArtifact(
        data={},
        meta={"encoder_name": "legacy", "representation_name": "mixed_identity"},
    )

    with pytest.raises(ArtifactValidationError, match="missing required fields"):
        artifact.finalize()


def test_staging_failure_preserves_existing_artifact_and_cleans_temps(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "existing"
    target.mkdir()
    old_data = b"old data"
    old_meta = b"old metadata"
    (target / "data.npz").write_bytes(old_data)
    (target / "meta.json").write_bytes(old_meta)
    (target / "unrelated.txt").write_text("keep", encoding="utf-8")

    def fail_json_write(path: Path, serialized_meta: str) -> None:
        raise OSError("controlled staging failure")

    monkeypatch.setattr(artifacts_module, "_write_staged_json", fail_json_write)

    with pytest.raises(OSError, match="controlled staging failure"):
        _explicit_artifact().save(target)

    assert (target / "data.npz").read_bytes() == old_data
    assert (target / "meta.json").read_bytes() == old_meta
    assert (target / "unrelated.txt").read_text(encoding="utf-8") == "keep"
    assert not [path for path in target.iterdir() if path.name.startswith(".eris-")]


def test_save_replaces_canonical_files_and_leaves_unrelated_files(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    (target / "data.npz").write_bytes(b"old data")
    (target / "meta.json").write_bytes(b"old metadata")
    (target / "unrelated.txt").write_text("keep", encoding="utf-8")

    artifact = _explicit_artifact()
    artifact.save(target)

    assert json.loads((target / "meta.json").read_text(encoding="utf-8"))["schema_version"] == 1
    with np.load(target / "data.npz", allow_pickle=False) as stored_data:
        assert stored_data.files == [output["name"] for output in artifact.meta["outputs"]]
    assert (target / "unrelated.txt").read_text(encoding="utf-8") == "keep"
    assert not [path for path in target.iterdir() if path.name.startswith(".eris-")]


def test_load_uses_persisted_contract(tmp_path):
    target = tmp_path / "artifacts"
    artifact = _explicit_artifact()
    artifact.save(target)

    loaded = StructureArtifact.load(target)

    assert loaded.meta == artifact.meta
    assert list(loaded.data) == [output["name"] for output in artifact.meta["outputs"]]
