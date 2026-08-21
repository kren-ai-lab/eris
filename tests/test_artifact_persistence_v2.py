from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import eris.core.artifacts as artifacts_module
from eris.core.artifact_schema_v2 import (
    ArtifactSchemaV2ValidationError,
    build_artifact_metadata,
    build_feature_keys,
    build_input_metadata,
    build_provenance,
)
from eris.core.artifacts import ArtifactValidationError, StructureArtifact


def _input_metadata(digest: str = "a") -> dict:
    return build_input_metadata(
        source="memory://synthetic",
        source_sha256=digest * 64,
        input_format="pdb",
        model_index=0,
        chain_id=None,
    )


def _explicit_artifact() -> StructureArtifact:
    data = {
        "edge_index": np.array([[0, 1], [1, 0]], dtype=np.int64),
        "pos": np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        "node_attr": np.array([[1, 0], [0, 1]], dtype=np.float32),
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
        ],
        feature_keys=build_feature_keys(node_attr_keys=["mass", "charge"]),
        counts={"n_nodes": 2, "n_edges": 2, "n_self_loops": 0},
        extensions={"tests.m3_5": {"reviewed": True}},
    )
    return StructureArtifact(data=data, meta=meta)


def _learned_artifact() -> StructureArtifact:
    data = {
        "valid_mask": np.array([True, False], dtype=np.bool_),
        "node_ids": np.array(["A:1", "A:2"], dtype=np.str_),
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


def _provisional_v1_artifact() -> StructureArtifact:
    return StructureArtifact(
        data={"pos": np.zeros((2, 3), dtype=np.float32)},
        meta={
            "encoder_name": "legacy_cloud",
            "encoder_version": "0.1.0",
            "encoder_fingerprint": "legacy-fingerprint",
            "type": "cloud",
            "level": "residue",
            "params": {},
            "input": "source=memory://legacy|chain=None|model=0",
            "feature_keys": {},
        },
    )


def _assert_data_unchanged(
    artifact: StructureArtifact,
    before: dict[str, np.ndarray],
) -> None:
    assert list(artifact.data) == list(before)
    for name, expected in before.items():
        np.testing.assert_array_equal(artifact.data[name], expected)


def test_explicit_v2_finalize_is_idempotent_and_save_preserves_contract(tmp_path):
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


def test_learned_v2_validate_finalize_and_save(tmp_path):
    artifact = _learned_artifact()
    data_before = {name: array.copy() for name, array in artifact.data.items()}

    artifact.validate(strict=False)
    artifact.finalize()
    artifact.save(tmp_path / "learned")

    stored_meta = json.loads((tmp_path / "learned" / "meta.json").read_text(encoding="utf-8"))
    assert stored_meta["schema_version"] == 2
    assert stored_meta["representation_family"] == "learned"
    assert stored_meta["representation_type"] == "embedding"
    assert stored_meta["coordinate_source"] == "model_specific"
    assert stored_meta["model"]["checkpoint"]["identifier"] == "synthetic-checkpoint"
    assert [output["name"] for output in stored_meta["outputs"]] == [
        "node_embedding",
        "global_embedding",
        "node_ids",
        "valid_mask",
    ]
    _assert_data_unchanged(artifact, data_before)


def test_valid_v2_finalize_canonicalizes_only_after_validation():
    artifact = _explicit_artifact()
    artifact.meta["extensions"]["tests.m3_5"]["labels"] = ("first", "second")
    original_meta = artifact.meta
    data_before = {name: array.copy() for name, array in artifact.data.items()}

    artifact.validate()
    assert artifact.meta is original_meta
    assert artifact.meta["extensions"]["tests.m3_5"]["labels"] == (
        "first",
        "second",
    )

    artifact.finalize()
    assert artifact.meta is not original_meta
    assert artifact.meta["extensions"]["tests.m3_5"]["labels"] == [
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
        np.array([b"raw"], dtype="S3"),
        np.array([1 + 2j], dtype=np.complex64),
    ],
    ids=["object", "bytes", "complex"],
)
def test_unsafe_v2_payload_fails_before_filesystem_mutation(tmp_path, unsafe):
    artifact = _explicit_artifact()
    artifact.data["node_attr"] = unsafe
    meta_before = deepcopy(artifact.meta)
    target = tmp_path / "unsafe"

    with pytest.raises(ArtifactValidationError) as raised:
        artifact.save(target)

    assert isinstance(raised.value.__cause__, ArtifactSchemaV2ValidationError)
    assert artifact.meta == meta_before
    assert artifact.data["node_attr"] is unsafe
    assert not target.exists()


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_non_finite_v2_metadata_fails_without_mutation(tmp_path, bad_value):
    artifact = _explicit_artifact()
    artifact.meta["extensions"]["tests.m3_5"]["non_finite"] = bad_value
    original_meta = artifact.meta
    data_before = {name: array.copy() for name, array in artifact.data.items()}
    target = tmp_path / "non-finite"

    with pytest.raises(ArtifactValidationError) as raised:
        artifact.save(target)

    assert isinstance(raised.value.__cause__, ArtifactSchemaV2ValidationError)
    assert artifact.meta is original_meta
    assert artifact.meta["extensions"]["tests.m3_5"]["non_finite"] is bad_value
    _assert_data_unchanged(artifact, data_before)
    assert not target.exists()


def test_v2_save_validate_false_cannot_bypass_strict_validation(tmp_path):
    artifact = _explicit_artifact()
    artifact.meta["counts"]["n_nodes"] = 99
    meta_before = deepcopy(artifact.meta)
    target = tmp_path / "invalid-counts"

    with pytest.raises(ArtifactValidationError, match="n_nodes"):
        artifact.save(target, validate=False)

    assert artifact.meta == meta_before
    assert not target.exists()


@pytest.mark.parametrize("operation", ["validate", "finalize", "save"])
def test_invalid_v2_operations_do_not_mutate_artifact(tmp_path, operation):
    artifact = _explicit_artifact()
    artifact.meta["counts"]["n_nodes"] = 99
    meta_before = deepcopy(artifact.meta)
    data_before = {name: array.copy() for name, array in artifact.data.items()}
    target = tmp_path / "invalid-v2"

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
    [True, False, "1", "2", None, np.int64(2), 3],
    ids=["true", "false", "string-one", "string-two", "none", "numpy-int", "future"],
)
@pytest.mark.parametrize("operation", ["validate", "finalize", "save"])
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


@pytest.mark.parametrize(
    "meta",
    [
        {"encoder_name": "incomplete"},
        {
            **_provisional_v1_artifact().meta,
            "representation_name": "mixed_identity",
        },
    ],
    ids=["incomplete", "mixed-v1-v2"],
)
def test_invalid_versionless_metadata_is_not_treated_as_v1(meta):
    artifact = StructureArtifact(data={}, meta=meta)

    with pytest.raises(ArtifactValidationError, match="Versionless metadata"):
        artifact.finalize()


def test_provisional_v1_requires_finalize_and_save_remains_v1(tmp_path):
    artifact = _provisional_v1_artifact()

    with pytest.raises(ArtifactValidationError, match="must be finalized"):
        artifact.validate()

    target = tmp_path / "provisional-v1"
    artifact.save(target)

    assert artifact.meta["schema_version"] == 1
    assert "representation_name" not in artifact.meta
    stored_meta = json.loads((target / "meta.json").read_text(encoding="utf-8"))
    assert stored_meta["schema_version"] == 1
    assert "parameters" not in stored_meta
    loaded = StructureArtifact.load(target, validate=True, strict=True)
    assert loaded.meta == stored_meta
    np.testing.assert_array_equal(loaded.data["pos"], artifact.data["pos"])


def test_finalized_v1_validate_and_save_without_conversion(tmp_path):
    artifact = _provisional_v1_artifact()
    artifact.finalize()
    meta_before = deepcopy(artifact.meta)

    artifact.validate(strict=True)
    artifact.save(tmp_path / "finalized-v1")

    assert artifact.meta == meta_before
    assert artifact.meta["schema_version"] == 1
    assert "representation_name" not in artifact.meta


def test_v1_save_validate_false_preserves_historical_bypass(tmp_path):
    artifact = StructureArtifact(
        data={"value": np.array([1], dtype=np.float32)},
        meta={"schema_version": 1},
    )

    artifact.save(tmp_path / "v1-unvalidated", validate=False)

    assert (tmp_path / "v1-unvalidated" / "data.npz").is_file()
    assert artifact.meta["schema_version"] == 1


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


def test_v2_save_replaces_canonical_files_and_leaves_unrelated_files(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    (target / "data.npz").write_bytes(b"old data")
    (target / "meta.json").write_bytes(b"old metadata")
    (target / "unrelated.txt").write_text("keep", encoding="utf-8")

    artifact = _explicit_artifact()
    artifact.save(target)

    assert json.loads((target / "meta.json").read_text(encoding="utf-8"))["schema_version"] == 2
    with np.load(target / "data.npz", allow_pickle=False) as stored_data:
        assert stored_data.files == [output["name"] for output in artifact.meta["outputs"]]
    assert (target / "unrelated.txt").read_text(encoding="utf-8") == "keep"
    assert not [path for path in target.iterdir() if path.name.startswith(".eris-")]


def test_v2_load_is_explicitly_deferred(tmp_path):
    target = tmp_path / "v2"
    _explicit_artifact().save(target)

    with pytest.raises(ArtifactValidationError, match="v2 loading is not supported"):
        StructureArtifact.load(target)
