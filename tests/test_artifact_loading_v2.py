from __future__ import annotations

import io
import json
import zipfile
from copy import deepcopy

import numpy as np
import pytest

from eris.core.artifact_schema_v2 import (
    build_artifact_metadata,
    build_feature_keys,
    build_input_metadata,
    build_provenance,
)
from eris.core.artifacts import ArtifactValidationError, StructureArtifact


def _metadata(
    data: dict[str, np.ndarray],
    *,
    learned: bool = False,
) -> dict:
    representation_name = "synthetic_embedding" if learned else "synthetic_graph"
    parameters = {"mode": "synthetic"}
    if learned:
        declarations = [
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
        ]
        model = {
            "architecture": "synthetic-model",
            "checkpoint": {
                "identifier": "synthetic-checkpoint",
                "revision": "v1",
                "sha256": "c" * 64,
            },
        }
    else:
        declarations = [
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
        ]
        model = None

    return build_artifact_metadata(
        data=data,
        representation_name=representation_name,
        representation_family="learned" if learned else "explicit",
        representation_type="embedding" if learned else "graph",
        granularity="mixed" if learned else "residue",
        coordinate_source="model_specific" if learned else "ca",
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
            representation_name=representation_name,
            parameters=parameters,
        ),
        output_declarations=declarations,
        feature_keys=(
            build_feature_keys()
            if learned
            else build_feature_keys(node_attr_keys=["mass", "charge"])
        ),
        counts={"n_nodes": 2} if learned else {"n_nodes": 2, "n_edges": 2},
        model=model,
        extensions={"tests.loading": {"canonical": True}},
    )


def _explicit_artifact() -> StructureArtifact:
    data = {
        "edge_index": np.array([[0, 1], [1, 0]], dtype=np.int64),
        "node_attr": np.array([[1, 0], [0, 1]], dtype=np.float32),
        "pos": np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
    }
    return StructureArtifact(data=data, meta=_metadata(data))


def _learned_artifact() -> StructureArtifact:
    data = {
        "valid_mask": np.array([True, False], dtype=np.bool_),
        "node_ids": np.array(["A:1", "A:2"], dtype=np.str_),
        "global_embedding": np.array([0.2, 0.3], dtype=np.float32),
        "node_embedding": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
    }
    return StructureArtifact(data=data, meta=_metadata(data, learned=True))


def _write_meta(path, meta: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def _write_npz(path, data: dict[str, np.ndarray]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "data.npz", **data)


def _legacy_artifact() -> StructureArtifact:
    return StructureArtifact(
        data={"pos": np.zeros((2, 3), dtype=np.float32)},
        meta={
            "schema_version": 1,
            "encoder_name": "legacy_cloud",
            "encoder_version": "0.1.0",
            "encoder_fingerprint": "legacy-fingerprint",
            "type": "cloud",
            "level": "residue",
            "params": {},
            "input": "source=memory://legacy|chain=None|model=0",
            "feature_keys": {},
            "shapes": {"pos": [2, 3]},
            "dtypes": {"pos": "float32"},
            "counts": {"n_nodes": 2},
        },
    )


def test_explicit_v2_round_trip_is_canonical_and_ordered(tmp_path):
    artifact = _explicit_artifact()
    target = tmp_path / "explicit"
    artifact.save(target)

    loaded = StructureArtifact.load(target)

    assert loaded.meta == artifact.meta
    assert loaded.meta["schema_version"] == 2
    assert loaded.meta["input"]["source_sha256"] == "a" * 64
    for field in ("outputs", "feature_keys", "counts", "shapes", "dtypes"):
        assert loaded.meta[field] == artifact.meta[field]
    expected_order = [output["name"] for output in artifact.meta["outputs"]]
    assert list(loaded.data) == expected_order
    for name in expected_order:
        np.testing.assert_array_equal(loaded.data[name], artifact.data[name])


def test_learned_v2_round_trip_preserves_model_and_alignment(tmp_path):
    artifact = _learned_artifact()
    target = tmp_path / "learned"
    artifact.save(target)

    loaded = StructureArtifact.load(target, validate=False, strict=False)

    assert loaded.meta == artifact.meta
    assert loaded.meta["model"]["checkpoint"]["identifier"] == "synthetic-checkpoint"
    assert list(loaded.data) == [
        "node_embedding",
        "global_embedding",
        "node_ids",
        "valid_mask",
    ]
    for name, expected in artifact.data.items():
        np.testing.assert_array_equal(loaded.data[name], expected)


@pytest.mark.parametrize(
    ("load_kwargs", "field", "value", "match"),
    [
        ({"validate": False}, "counts", {"n_nodes": 99, "n_edges": 2}, "n_nodes"),
        ({"strict": False}, "dtypes", {"pos": "float64"}, "dtypes"),
    ],
)
def test_v2_load_flags_cannot_disable_strict_validation(
    tmp_path,
    load_kwargs,
    field,
    value,
    match,
):
    artifact = _explicit_artifact()
    target = tmp_path / "strict"
    artifact.save(target)
    meta = json.loads((target / "meta.json").read_text(encoding="utf-8"))
    if field == "dtypes":
        meta[field].update(value)
    else:
        meta[field] = value
    _write_meta(target, meta)

    with pytest.raises(ArtifactValidationError, match=match):
        StructureArtifact.load(target, **load_kwargs)


def test_saved_v1_load_is_rejected_without_migration(tmp_path):
    artifact = _legacy_artifact()
    target = tmp_path / "v1"
    _write_meta(target, artifact.meta)
    _write_npz(target, artifact.data)

    with pytest.raises(ArtifactValidationError, match="Schema v1 is no longer supported"):
        StructureArtifact.load(target, validate=True, strict=True)


def test_versionless_persisted_artifact_is_rejected(tmp_path):
    artifact = _legacy_artifact()
    target = tmp_path / "versionless"
    meta = deepcopy(artifact.meta)
    del meta["schema_version"]
    _write_meta(target, meta)
    _write_npz(target, artifact.data)

    with pytest.raises(ArtifactValidationError, match="Schema v1 is no longer supported"):
        StructureArtifact.load(target)


@pytest.mark.parametrize(
    "version",
    [3, True, "2", None],
    ids=["future", "bool", "string", "none"],
)
def test_future_and_invalid_persisted_versions_are_rejected(tmp_path, version):
    artifact = _explicit_artifact()
    target = tmp_path / "invalid-version"
    artifact.save(target)
    meta = json.loads((target / "meta.json").read_text(encoding="utf-8"))
    meta["schema_version"] = version
    _write_meta(target, meta)

    with pytest.raises(ArtifactValidationError):
        StructureArtifact.load(target)


def test_malformed_json_is_wrapped_as_artifact_error(tmp_path):
    target = tmp_path / "bad-json"
    target.mkdir()
    (target / "meta.json").write_text("{bad", encoding="utf-8")
    np.savez_compressed(target / "data.npz", value=np.array([1]))

    with pytest.raises(ArtifactValidationError) as raised:
        StructureArtifact.load(target)

    assert isinstance(raised.value.__cause__, json.JSONDecodeError)


def test_corrupt_npz_is_wrapped_as_artifact_error(tmp_path):
    artifact = _explicit_artifact()
    target = tmp_path / "bad-npz"
    artifact.save(target)
    (target / "data.npz").write_bytes(b"not an npz")

    with pytest.raises(ArtifactValidationError) as raised:
        StructureArtifact.load(target)

    assert raised.value.__cause__ is not None


def test_object_npz_is_rejected_without_pickle_fallback(tmp_path):
    artifact = _explicit_artifact()
    target = tmp_path / "object"
    _write_meta(target, artifact.meta)
    unsafe = dict(artifact.data)
    unsafe["node_attr"] = np.array([object()], dtype=object)
    _write_npz(target, unsafe)

    with pytest.raises(ArtifactValidationError) as raised:
        StructureArtifact.load(target)

    assert isinstance(raised.value.__cause__, ValueError)


def test_missing_path_non_directory_and_missing_files_are_clear(tmp_path):
    with pytest.raises(ArtifactValidationError, match="does not exist"):
        StructureArtifact.load(tmp_path / "absent")

    file_path = tmp_path / "file"
    file_path.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="not a directory"):
        StructureArtifact.load(file_path)

    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "meta.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="Missing artifact files"):
        StructureArtifact.load(partial)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_v2_npz_members_must_exactly_match_outputs(tmp_path, change):
    artifact = _explicit_artifact()
    target = tmp_path / change
    _write_meta(target, artifact.meta)
    data = dict(artifact.data)
    if change == "missing":
        del data["node_attr"]
    else:
        data["hidden"] = np.array([1], dtype=np.int64)
    _write_npz(target, data)

    with pytest.raises(ArtifactValidationError, match="exactly match"):
        StructureArtifact.load(target)


def test_duplicate_npz_member_is_rejected(tmp_path):
    artifact = _explicit_artifact()
    target = tmp_path / "duplicate"
    _write_meta(target, artifact.meta)
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(target / "data.npz", "w") as archive:
            for name in ("pos", "node_attr", "edge_index", "pos"):
                buffer = io.BytesIO()
                np.save(buffer, artifact.data[name])
                archive.writestr(f"{name}.npy", buffer.getvalue())

    with pytest.raises(ArtifactValidationError, match="duplicate"):
        StructureArtifact.load(target)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda meta: meta["shapes"].__setitem__("pos", [3, 3]), "shapes"),
        (lambda meta: meta["dtypes"].__setitem__("node_attr", "float64"), "dtypes"),
        (
            lambda meta: meta["feature_keys"].__setitem__("node_attr_keys", ["mass"]),
            "feature_keys",
        ),
        (lambda meta: meta["counts"].__setitem__("n_edges", 99), "n_edges"),
    ],
)
def test_v2_strict_integrity_mismatches_are_rejected(tmp_path, mutation, match):
    artifact = _explicit_artifact()
    target = tmp_path / "mismatch"
    artifact.save(target)
    meta = deepcopy(artifact.meta)
    mutation(meta)
    _write_meta(target, meta)

    with pytest.raises(ArtifactValidationError, match=match):
        StructureArtifact.load(target)


def test_npz_physical_order_is_ignored_and_runtime_uses_outputs_order(tmp_path):
    artifact = _explicit_artifact()
    target = tmp_path / "physical-order"
    _write_meta(target, artifact.meta)
    physical_order = {
        "edge_index": artifact.data["edge_index"],
        "node_attr": artifact.data["node_attr"],
        "pos": artifact.data["pos"],
    }
    _write_npz(target, physical_order)

    loaded = StructureArtifact.load(target)

    assert list(loaded.data) == ["pos", "node_attr", "edge_index"]


def test_structurally_different_generation_is_rejected(tmp_path):
    artifact = _explicit_artifact()
    target = tmp_path / "different-shape"
    _write_meta(target, artifact.meta)
    changed = dict(artifact.data)
    changed["pos"] = np.zeros((3, 3), dtype=np.float32)
    _write_npz(target, changed)

    with pytest.raises(ArtifactValidationError, match="shape"):
        StructureArtifact.load(target)


def test_same_shape_numeric_generation_mismatch_is_currently_undetectable(tmp_path):
    artifact = _explicit_artifact()
    target = tmp_path / "numeric-generation"
    _write_meta(target, artifact.meta)
    changed = {name: array.copy() for name, array in artifact.data.items()}
    changed["pos"] = changed["pos"] + np.float32(10.0)
    _write_npz(target, changed)

    loaded = StructureArtifact.load(target)

    np.testing.assert_array_equal(loaded.data["pos"], changed["pos"])
    assert not np.array_equal(loaded.data["pos"], artifact.data["pos"])
