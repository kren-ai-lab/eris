from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from eris.core.artifact_schema import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactValidationError,
    compute_config_fingerprint,
    normalize_metadata,
    validate_schema,
)
from tests.helpers import learned_artifact


def _maps(data: dict[str, np.ndarray]) -> tuple[dict[str, list[int]], dict[str, str]]:
    return (
        {name: list(array.shape) for name, array in data.items()},
        {name: str(array.dtype) for name, array in data.items()},
    )


def _common_meta(data: dict[str, np.ndarray]) -> dict:
    shapes, dtypes = _maps(data)
    parameters = {"center": True, "selection": {"chain": None}, "scales": [1.0, 2.0]}
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "representation_name": "residue_knn_graph",
        "representation_family": "explicit",
        "representation_type": "graph",
        "granularity": "structure",
        "coordinate_source": "ca",
        "encoder": {
            "implementation": "eris.encoders.graphs.ResidueKNNGraph",
            "version": "0.1.0",
        },
        "input": {
            "source": "memory://tiny-structure",
            "source_sha256": "a" * 64,
            "format": "pdb",
            "model_index": 0,
            "chain_ids": ["A"],
        },
        "parameters": parameters,
        "provenance": {
            "eris_version": "0.1.0",
            "config_fingerprint": compute_config_fingerprint(
                "residue_knn_graph", parameters, chain_ids=["A"]
            ),
        },
        "outputs": [],
        "feature_keys": {},
        "shapes": shapes,
        "dtypes": dtypes,
        "counts": {},
    }


def _explicit_graph() -> tuple[dict, dict[str, np.ndarray]]:
    data = {
        "pos": np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32),
        "node_attr": np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32),
        "edge_index": np.array([[0, 1, 2], [1, 0, 2]], dtype=np.int64),
        "edge_attr": np.array([[1.0], [1.0], [0.0]], dtype=np.float32),
        "graph_attr": np.array([3.0, 3.0], dtype=np.float32),
        "chain_id": np.array(["A", "A", "A"], dtype=np.str_),
    }
    meta = _common_meta(data)
    meta["granularity"] = "residue"
    meta["outputs"] = [
        {"name": "pos", "role": "positions", "granularity": "residue", "layout": "N3"},
        {"name": "node_attr", "role": "node_features", "granularity": "residue"},
        {"name": "edge_index", "role": "edge_index", "layout": "2E"},
        {"name": "edge_attr", "role": "edge_features", "layout": "EF"},
        {"name": "graph_attr", "role": "graph_features"},
        {"name": "chain_id", "role": "chain_id", "granularity": "residue"},
    ]
    meta["feature_keys"] = {
        "node_attr_keys": ["residue.mass", "residue.charge"],
        "edge_attr_keys": ["edge.distance"],
        "graph_attr_keys": ["graph.n_nodes", "graph.n_edges"],
    }
    meta["counts"] = {"n_nodes": 3, "n_edges": 3, "n_self_loops": 1}
    return meta, data


def _learned_embedding() -> tuple[dict, dict[str, np.ndarray]]:
    artifact = learned_artifact(with_mask=True)
    return deepcopy(artifact.meta), {name: array.copy() for name, array in artifact.data.items()}


def _pooled_vector() -> tuple[dict, dict[str, np.ndarray]]:
    data = {"pooled_attr": np.asarray([1.0, np.nan], dtype=np.float64)}
    meta = _common_meta(data)
    parameters = {
        "mode": "mean",
        "nan_policy": "omit",
        "source": "node_attr",
        "source_artifact_fingerprint": "b" * 64,
        "source_feature_keys": ["mass", "charge"],
    }
    meta.update(
        {
            "representation_name": "residue_pool",
            "representation_type": "vector",
            "granularity": "structure",
            "parameters": parameters,
            "outputs": [
                {
                    "name": "pooled_attr",
                    "role": "pooled_features",
                    "granularity": "structure",
                    "layout": "F",
                }
            ],
            "feature_keys": {"pooled_attr_keys": ["pool.mean.mass", "pool.mean.charge"]},
            "counts": {"n_features": 2},
        }
    )
    meta["provenance"]["config_fingerprint"] = compute_config_fingerprint(
        "residue_pool",
        parameters,
        chain_ids=["A"],
    )
    return meta, data


def _assert_invalid(meta: dict, data: dict[str, np.ndarray], match: str) -> None:
    with pytest.raises(ArtifactValidationError, match=match):
        validate_schema(meta, data, strict=True)


def test_explicit_synthetic_graph_is_strict_valid_and_preserves_output_order():
    meta, data = _explicit_graph()

    normalized = validate_schema(meta, data, strict=True)

    assert [output["name"] for output in normalized["outputs"]] == list(data)
    assert "model" not in normalized


def test_learned_synthetic_embedding_is_strict_valid():
    meta, data = _learned_embedding()

    normalized = validate_schema(meta, data, strict=True)

    assert normalized["model"] == meta["model"]
    assert normalized["feature_keys"] == {}


def test_explicit_pooled_vector_is_strict_valid_and_preserves_nan():
    meta, data = _pooled_vector()

    normalized = validate_schema(meta, data, strict=True)

    assert normalized["representation_type"] == "vector"
    assert normalized["feature_keys"]["pooled_attr_keys"] == [
        "pool.mean.mass",
        "pool.mean.charge",
    ]
    assert np.isnan(data["pooled_attr"][1])


@pytest.mark.parametrize(
    "field",
    ["schema_version", "encoder", "counts"],
)
def test_each_required_top_level_field_is_enforced(field):
    meta, data = _explicit_graph()
    del meta[field]

    _assert_invalid(meta, data, "missing required fields")


def test_unknown_top_level_field_is_rejected():
    meta, data = _explicit_graph()
    meta["surprise"] = True

    _assert_invalid(meta, data, "unknown fields")


@pytest.mark.parametrize("schema_version", [True, "1", 2], ids=["bool", "string", "future"])
def test_only_integer_schema_version_one_is_accepted(schema_version):
    meta, data = _explicit_graph()
    meta["schema_version"] = schema_version

    _assert_invalid(meta, data, "must equal exactly 1")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("representation_family", "hybrid"),
        ("representation_type", "embedding"),
        ("coordinate_source", "all_atom"),
    ],
)
def test_invalid_identity_values_and_combinations_fail(field, value):
    meta, data = _explicit_graph()
    meta[field] = value

    _assert_invalid(meta, data, field)


@pytest.mark.parametrize(
    ("granularity", "coordinate_source"),
    [("atom", "ca"), ("residue", "model_specific")],
)
def test_explicit_granularity_and_coordinate_source_must_agree(granularity, coordinate_source):
    meta, data = _explicit_graph()
    meta["granularity"] = granularity
    meta["coordinate_source"] = coordinate_source

    _assert_invalid(meta, data, "requires coordinate_source")


def test_mixed_learned_coordinates_are_model_specific():
    meta, data = _learned_embedding()
    meta["coordinate_source"] = "ca"

    _assert_invalid(meta, data, "model_specific")


def test_encoder_implementation_is_required_and_non_empty():
    meta, data = _explicit_graph()
    del meta["encoder"]["implementation"]

    _assert_invalid(meta, data, "meta.encoder")

    meta, data = _explicit_graph()
    meta["encoder"]["implementation"] = ""
    _assert_invalid(meta, data, "encoder.implementation")


def test_encoder_implementation_must_be_import_path_like():
    meta, data = _explicit_graph()
    meta["encoder"]["implementation"] = "not a path"

    _assert_invalid(meta, data, "import-path-like")


def test_explicit_model_is_forbidden_and_learned_model_is_required():
    explicit_meta, explicit_data = _explicit_graph()
    explicit_meta["model"] = {
        "architecture": "not-allowed",
        "checkpoint": {"identifier": "not-allowed"},
    }
    _assert_invalid(explicit_meta, explicit_data, "must be absent")

    learned_meta, learned_data = _learned_embedding()
    del learned_meta["model"]
    _assert_invalid(learned_meta, learned_data, "is required")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda model: model.pop("architecture"), "missing required"),
        (lambda model: model["checkpoint"].update({"sha256": "ABC"}), "lowercase"),
    ],
)
def test_learned_model_and_checkpoint_are_strict(mutation, match):
    meta, data = _learned_embedding()
    mutation(meta["model"])

    _assert_invalid(meta, data, match)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update({"source_sha256": "A" * 64}), "lowercase"),
        (lambda value: value.update({"model_index": -1}), "integer >= 0"),
        (lambda value: value.update({"chain_ids": []}), "non-empty"),
        (lambda value: value.update({"chain_ids": ["A", "A"]}), "unique"),
    ],
)
def test_input_contract_rejects_invalid_values(mutation, match):
    meta, data = _explicit_graph()
    mutation(meta["input"])

    _assert_invalid(meta, data, match)


def test_optional_input_and_provenance_fields_are_accepted():
    meta, data = _explicit_graph()
    meta["input"].update({"original_filename": "tiny.pdb", "structure_id": "tiny"})
    meta["provenance"].update(
        {"created_at": "2026-08-21T12:30:00Z", "invocation": {"command": "synthetic-test"}}
    )

    validate_schema(meta, data, strict=True)


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (float("nan"), "finite"),
        (np.int64(1), "NumPy"),
    ],
)
def test_parameters_reject_non_json_or_numpy_values(value, match):
    meta, data = _explicit_graph()
    meta["parameters"] = {"bad": value}

    _assert_invalid(meta, data, match)


def test_output_declarations_preserve_order_and_optional_fields():
    meta, data = _explicit_graph()
    meta["outputs"][0]["description"] = "Centered residue coordinates"

    normalized = validate_schema(meta, data, strict=True)

    assert normalized["outputs"] == meta["outputs"]


def test_duplicate_output_names_fail():
    meta, data = _explicit_graph()
    meta["outputs"][1]["name"] = "pos"

    _assert_invalid(meta, data, "must be unique")


def test_output_names_must_exactly_equal_data_keys():
    meta, data = _explicit_graph()
    del data["pos"]

    _assert_invalid(meta, data, "exactly match data keys")


def test_output_and_data_key_syntax_is_restricted():
    name = "bad-name"
    meta, data = _explicit_graph()
    old_name = meta["outputs"][0]["name"]
    meta["outputs"][0]["name"] = name
    data[name] = data.pop(old_name)
    meta["shapes"][name] = meta["shapes"].pop(old_name)
    meta["dtypes"][name] = meta["dtypes"].pop(old_name)

    _assert_invalid(meta, data, "must match")


def test_shapes_and_dtypes_maps_must_have_exact_keys():
    meta, data = _explicit_graph()
    del meta["shapes"]["pos"]
    _assert_invalid(meta, data, "shapes keys")

    meta, data = _explicit_graph()
    meta["dtypes"]["extra"] = "float32"
    _assert_invalid(meta, data, "dtypes keys")


def test_strict_validation_checks_shape_and_dtype_values():
    meta, data = _explicit_graph()
    meta["shapes"]["pos"] = [99, 3]
    _assert_invalid(meta, data, "does not match")

    meta, data = _explicit_graph()
    meta["dtypes"]["pos"] = "float64"
    _assert_invalid(meta, data, "does not match")


def test_pickle_dependent_data_dtypes_are_rejected():
    data = {"value": np.array([object()], dtype=object)}
    meta = _common_meta(data)
    meta["outputs"] = [{"name": "value", "role": "custom"}]

    _assert_invalid(meta, data, "prohibited dtype")


def test_data_values_must_be_ndarrays():
    meta, data = _explicit_graph()
    cast("dict[str, Any]", data)["pos"] = [[0.0, 0.0, 0.0]]

    _assert_invalid(meta, data, "numpy.ndarray")


def test_feature_keys_accept_only_core_known_ordered_lists():
    meta, data = _explicit_graph()
    normalized = validate_schema(meta, data, strict=True)
    assert normalized["feature_keys"]["node_attr_keys"] == [
        "residue.mass",
        "residue.charge",
    ]

    meta["feature_keys"]["matrix_key"] = "matrix"
    _assert_invalid(meta, data, "unknown")


def test_feature_key_cardinality_is_checked():
    meta, data = _explicit_graph()
    meta["feature_keys"]["node_attr_keys"] = []

    _assert_invalid(meta, data, "node_attr_keys")


def test_counts_require_non_negative_plain_integers_and_allow_empty_mapping():
    meta, data = _explicit_graph()
    meta["counts"] = {}
    validate_schema(meta, data, strict=True)

    meta["counts"] = {"n_nodes": -1}
    _assert_invalid(meta, data, "integer >= 0")

    meta["counts"] = {"n_nodes": True}
    _assert_invalid(meta, data, "integer >= 0")


def test_strict_graph_counts_are_checked_against_roles():
    meta, data = _explicit_graph()
    meta["counts"]["n_self_loops"] = 0

    _assert_invalid(meta, data, "n_self_loops")


def test_extensions_are_namespaced_finite_json_mappings():
    meta, data = _explicit_graph()
    meta["extensions"] = {"org.example.analysis": {"score": 1.25}}
    validate_schema(meta, data, strict=True)

    meta["extensions"] = {"": {}}
    _assert_invalid(meta, data, "namespaces")

    meta["extensions"] = {"org.example.analysis": []}
    _assert_invalid(meta, data, "must be a mapping")


def test_config_fingerprint_is_key_order_independent_and_identity_sensitive():
    left = {"center": True, "nested": {"b": 2, "a": 1}}
    right = {"nested": {"a": 1, "b": 2}, "center": True}
    assert compute_config_fingerprint(
        "residue_knn_graph", left, chain_ids=["A"]
    ) == compute_config_fingerprint("residue_knn_graph", right, chain_ids=["A"])

    baseline = compute_config_fingerprint("residue_knn_graph", {"k": 8}, chain_ids=["A"])

    assert baseline != compute_config_fingerprint("residue_knn_graph", {"k": 9}, chain_ids=["A"])
    assert baseline != compute_config_fingerprint("atom_knn_graph", {"k": 8}, chain_ids=["A"])


def test_strict_validation_checks_config_fingerprint_against_name_and_parameters():
    meta, data = _explicit_graph()
    meta["provenance"]["config_fingerprint"] = "0" * 64

    _assert_invalid(meta, data, "does not match representation_name")


def test_normalization_converts_numpy_scalars_and_tuples_without_mutating_input():
    original = {"values": (np.int64(1), np.float32(2.5)), "enabled": np.bool_(True)}  # noqa: FBT003
    snapshot = deepcopy(original)

    normalized = normalize_metadata(original)

    assert normalized == {"values": [1, 2.5], "enabled": True}
    assert isinstance(original["values"], tuple)
    assert original["values"][0] == snapshot["values"][0]


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (Path("artifact"), "Path"),
        (float("nan"), "finite"),
    ],
)
def test_normalization_rejects_unsupported_metadata(value, match):
    with pytest.raises(ArtifactValidationError, match=match):
        normalize_metadata(value)


def test_learned_node_ids_must_align_be_unique_and_use_safe_unicode():
    meta, data = _learned_embedding()
    data["node_ids"] = np.array(["A:1::ALA"], dtype=np.str_)
    meta["shapes"]["node_ids"] = [1]
    _assert_invalid(meta, data, "must align")

    meta, data = _learned_embedding()
    data["node_ids"] = np.array(["A:1::ALA", "A:1::ALA"], dtype=np.str_)
    _assert_invalid(meta, data, "unique")

    meta, data = _learned_embedding()
    data["node_ids"] = np.array(["A:1", object()], dtype=object)
    meta["dtypes"]["node_ids"] = "object"
    _assert_invalid(meta, data, "prohibited dtype")


def test_learned_valid_mask_must_be_bool_and_align():
    meta, data = _learned_embedding()
    data["valid_mask"] = np.array([1, 1], dtype=np.int8)
    meta["dtypes"]["valid_mask"] = "int8"
    _assert_invalid(meta, data, "bool dtype")

    meta, data = _learned_embedding()
    data["valid_mask"] = np.array([True], dtype=np.bool_)
    meta["shapes"]["valid_mask"] = [1]
    _assert_invalid(meta, data, "must align")


def test_learned_global_embedding_is_canonical_one_dimensional():
    meta, data = _learned_embedding()
    data["global_embedding"] = data["global_embedding"][None, :]
    meta["shapes"]["global_embedding"] = [1, 3]

    _assert_invalid(meta, data, "requires 1 dimensions")


def test_learned_embeddings_must_be_floating_finite_and_non_empty():
    meta, data = _learned_embedding()
    data["node_embedding"][0, 0] = np.nan
    _assert_invalid(meta, data, "finite")

    meta, data = _learned_embedding()
    data["global_embedding"] = np.array([], dtype=np.float32)
    meta["shapes"]["global_embedding"] = [0]
    _assert_invalid(meta, data, "D > 0")


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("residue_ids", np.array([], dtype=str), "no atoms/residues"),
        ("pos", np.array([[np.nan, 0, 0]], dtype=np.float32), "'pos' contains non-finite"),
        ("edge_attr", np.array([[np.inf]], dtype=np.float32), "'edge_attr' contains infinite"),
        ("edge_index", np.array([[0], [1]], dtype=np.int64), "out-of-range"),
    ],
)
def test_encoder_output_check_rejects_bad_values(key, value, message):
    from eris.core.base_encoder.base_encoder import _check_output_values

    data = {
        "residue_ids": np.array(["A:1::GLY"]),
        "pos": np.zeros((1, 3), dtype=np.float32),
        "edge_index": np.array([[0], [0]], dtype=np.int64),
        "edge_attr": np.array([[np.nan]], dtype=np.float32),
    }
    _check_output_values(data)
    data[key] = value

    with pytest.raises(Exception, match=message):
        _check_output_values(data)
