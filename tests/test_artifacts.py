from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from eris.core.artifact_schema_v2 import (
    ARTIFACT_SCHEMA_V2_VERSION,
    ArtifactSchemaV2ValidationError,
    compute_config_fingerprint,
    normalize_metadata,
    validate_schema_v2,
)


def _maps(data: dict[str, np.ndarray]) -> tuple[dict[str, list[int]], dict[str, str]]:
    return (
        {name: list(array.shape) for name, array in data.items()},
        {name: str(array.dtype) for name, array in data.items()},
    )


def _common_meta(data: dict[str, np.ndarray]) -> dict:
    shapes, dtypes = _maps(data)
    parameters = {"center": True, "selection": {"chain": None}, "scales": [1.0, 2.0]}
    return {
        "schema_version": ARTIFACT_SCHEMA_V2_VERSION,
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
    data = {
        "node_embedding": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
        "global_embedding": np.array([0.25, 0.35, 0.45], dtype=np.float32),
        "node_ids": np.array(["A:1", "A:2"], dtype=np.str_),
        "valid_mask": np.array([True, True], dtype=np.bool_),
        "chain_id": np.array(["A", "A"], dtype=np.str_),
    }
    meta = _common_meta(data)
    meta.update(
        {
            "representation_name": "synthetic_structure_embedding",
            "representation_family": "learned",
            "representation_type": "embedding",
            "granularity": "mixed",
            "coordinate_source": "model_specific",
            "encoder": {
                "implementation": "tests.fake_backend.FakeStructuralEncoder",
                "version": "1.0",
            },
            "model": {
                "architecture": "fake-structural-model",
                "checkpoint": {
                    "identifier": "fake-checkpoint",
                    "revision": "v1",
                    "source": "memory://fake-checkpoint",
                    "sha256": "b" * 64,
                },
            },
            "parameters": {"pooling": "mean"},
            "outputs": [
                {"name": "node_embedding", "role": "node_embedding", "layout": "ND"},
                {"name": "global_embedding", "role": "global_embedding", "layout": "D"},
                {"name": "node_ids", "role": "node_ids", "granularity": "residue"},
                {"name": "chain_id", "role": "chain_id", "granularity": "residue"},
                {"name": "valid_mask", "role": "valid_mask", "granularity": "residue"},
            ],
            "feature_keys": {},
            "counts": {"n_nodes": 2},
        }
    )
    meta["provenance"]["config_fingerprint"] = compute_config_fingerprint(
        meta["representation_name"], meta["parameters"], chain_ids=["A"]
    )
    return meta, data


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
    with pytest.raises(ArtifactSchemaV2ValidationError, match=match):
        validate_schema_v2(meta, data, strict=True)


def test_explicit_synthetic_graph_is_strict_valid_and_preserves_output_order():
    meta, data = _explicit_graph()

    normalized = validate_schema_v2(meta, data, strict=True)

    assert [output["name"] for output in normalized["outputs"]] == list(data)
    assert "model" not in normalized


def test_learned_synthetic_embedding_is_strict_valid():
    meta, data = _learned_embedding()

    normalized = validate_schema_v2(meta, data, strict=True)

    assert normalized["model"]["checkpoint"]["identifier"] == "fake-checkpoint"
    assert normalized["feature_keys"] == {}


def test_explicit_pooled_vector_is_strict_valid_and_preserves_nan():
    meta, data = _pooled_vector()

    normalized = validate_schema_v2(meta, data, strict=True)

    assert normalized["representation_type"] == "vector"
    assert normalized["feature_keys"]["pooled_attr_keys"] == [
        "pool.mean.mass",
        "pool.mean.charge",
    ]
    assert np.isnan(data["pooled_attr"][1])


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "representation_name",
        "representation_family",
        "representation_type",
        "granularity",
        "coordinate_source",
        "encoder",
        "input",
        "parameters",
        "provenance",
        "outputs",
        "feature_keys",
        "shapes",
        "dtypes",
        "counts",
    ],
)
def test_each_required_top_level_field_is_enforced(field):
    meta, data = _explicit_graph()
    del meta[field]

    _assert_invalid(meta, data, "missing required fields")


def test_optional_top_level_fields_are_accepted():
    meta, data = _explicit_graph()
    meta["extensions"] = {"org.example.audit": {"reviewed": True}}

    validate_schema_v2(meta, data, strict=True)


def test_unknown_top_level_field_is_rejected():
    meta, data = _explicit_graph()
    meta["surprise"] = True

    _assert_invalid(meta, data, "unknown fields")


@pytest.mark.parametrize("schema_version", ["2"])
def test_only_integer_schema_version_two_is_accepted(schema_version):
    meta, data = _explicit_graph()
    meta["schema_version"] = schema_version

    _assert_invalid(meta, data, "must equal exactly 2")


@pytest.mark.parametrize("representation_type", ["cloud", "graph", "matrix", "grid"])
def test_explicit_representation_types_are_valid(representation_type):
    meta, data = _explicit_graph()
    meta["representation_type"] = representation_type

    validate_schema_v2(meta, data)


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


def test_learned_type_must_be_embedding():
    meta, data = _learned_embedding()
    meta["representation_type"] = "graph"

    _assert_invalid(meta, data, "incompatible")


@pytest.mark.parametrize(
    ("granularity", "coordinate_source"),
    [
        ("atom", "ca"),
        ("residue", "model_specific"),
    ],
)
def test_explicit_granularity_and_coordinate_source_must_agree(granularity, coordinate_source):
    meta, data = _explicit_graph()
    meta["granularity"] = granularity
    meta["coordinate_source"] = coordinate_source

    _assert_invalid(meta, data, "requires coordinate_source")


@pytest.mark.parametrize("granularity", ["mixed"])
def test_global_or_mixed_learned_coordinates_are_model_specific(granularity):
    meta, data = _learned_embedding()
    meta["granularity"] = granularity
    meta["coordinate_source"] = "ca"

    _assert_invalid(meta, data, "model_specific")


@pytest.mark.parametrize("field", ["implementation"])
def test_encoder_fields_are_required_and_non_empty(field):
    meta, data = _explicit_graph()
    del meta["encoder"][field]

    _assert_invalid(meta, data, "meta.encoder")

    meta, data = _explicit_graph()
    meta["encoder"][field] = ""
    _assert_invalid(meta, data, f"encoder.{field}")


def test_encoder_unknown_fields_are_rejected():
    meta, data = _explicit_graph()
    meta["encoder"]["representation_name"] = meta["representation_name"]

    _assert_invalid(meta, data, "unknown fields")


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
    "mutation,match",
    [
        (lambda model: model.pop("architecture"), "missing required"),
        (lambda model: model["checkpoint"].pop("identifier"), "missing required"),
        (lambda model: model["checkpoint"].update({"sha256": "ABC"}), "lowercase"),
    ],
)
def test_learned_model_and_checkpoint_are_strict(mutation, match):
    meta, data = _learned_embedding()
    mutation(meta["model"])

    _assert_invalid(meta, data, match)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value.update({"source_sha256": "A" * 64}), "lowercase"),
        (lambda value: value.update({"model_index": -1}), "integer >= 0"),
        (lambda value: value.update({"chain_ids": []}), "non-empty"),
        (lambda value: value.update({"chain_ids": ["A", "A"]}), "unique"),
        (lambda value: value.update({"assembly": "1"}), "unknown fields"),
    ],
)
def test_input_contract_rejects_invalid_values(mutation, match):
    meta, data = _explicit_graph()
    mutation(meta["input"])

    _assert_invalid(meta, data, match)


def test_input_optional_fields_and_chain_list_are_accepted():
    meta, data = _explicit_graph()
    meta["input"].update(
        {
            "chain_ids": ["A"],
            "original_filename": "tiny.pdb",
            "structure_id": "tiny",
            "source_database": "PDB",
        }
    )

    validate_schema_v2(meta, data, strict=True)


def test_legacy_scalar_chain_metadata_is_rejected():
    meta, data = _explicit_graph()
    del meta["input"]["chain_ids"]
    meta["input"]["chain_id"] = "A"

    _assert_invalid(meta, data, "missing required")


@pytest.mark.parametrize("field", ["config_fingerprint"])
def test_provenance_core_fields_are_required(field):
    meta, data = _explicit_graph()
    del meta["provenance"][field]

    _assert_invalid(meta, data, "missing required")


def test_provenance_optional_environment_created_at_and_invocation_are_valid():
    meta, data = _explicit_graph()
    meta["provenance"].update(
        {
            "environment": {
                "python_version": "3.12.13",
                "platform": "test",
                "dependencies": {"numpy": "2.0"},
            },
            "created_at": "2026-08-21T12:30:00Z",
            "invocation": {"command": "synthetic-test"},
        }
    )

    validate_schema_v2(meta, data, strict=True)


@pytest.mark.parametrize("value", [float("nan")])
def test_provenance_must_be_finite_json(value):
    meta, data = _explicit_graph()
    meta["provenance"]["environment"] = {"measurement": value}

    _assert_invalid(meta, data, "finite")


def test_parameters_preserve_nested_json_and_none():
    meta, data = _explicit_graph()
    meta["parameters"] = {"selection": {"chain": None}, "features": ["mass", "charge"]}
    meta["provenance"]["config_fingerprint"] = compute_config_fingerprint(
        meta["representation_name"], meta["parameters"], chain_ids=["A"]
    )

    normalized = validate_schema_v2(meta, data, strict=True)

    assert normalized["parameters"] == meta["parameters"]
    assert normalized["parameters"]["selection"]["chain"] is None


@pytest.mark.parametrize(
    "value,match",
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

    normalized = validate_schema_v2(meta, data, strict=True)

    assert normalized["outputs"] == meta["outputs"]


def test_duplicate_output_names_fail():
    meta, data = _explicit_graph()
    meta["outputs"][1]["name"] = "pos"

    _assert_invalid(meta, data, "must be unique")


def test_output_names_must_exactly_equal_data_keys():
    meta, data = _explicit_graph()
    del data["pos"]

    _assert_invalid(meta, data, "exactly match data keys")


@pytest.mark.parametrize("name", ["bad-name"])
def test_output_and_data_key_syntax_is_restricted(name):
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


def test_scalar_shape_is_empty_list():
    data = {"score": np.array(1.0, dtype=np.float32)}
    meta = _common_meta(data)
    meta["representation_type"] = "matrix"
    meta["granularity"] = "structure"
    meta["outputs"] = [{"name": "score", "role": "score"}]

    validate_schema_v2(meta, data, strict=True)
    assert meta["shapes"]["score"] == []


@pytest.mark.parametrize(
    "array",
    [
        np.array([1.0], dtype=np.float32),
        np.array([True], dtype=np.bool_),
        np.array(["A:1"], dtype=np.str_),
    ],
)
def test_safe_data_dtypes_are_accepted(array):
    data = {"value": array}
    meta = _common_meta(data)
    meta["outputs"] = [{"name": "value", "role": "custom"}]

    validate_schema_v2(meta, data, strict=True)


@pytest.mark.parametrize(
    "array",
    [
        np.array([object()], dtype=object),
        np.array([1 + 2j], dtype=np.complex64),
    ],
)
def test_pickle_dependent_or_unsupported_data_dtypes_are_rejected(array):
    data = {"value": array}
    meta = _common_meta(data)
    meta["outputs"] = [{"name": "value", "role": "custom"}]

    _assert_invalid(meta, data, "prohibited dtype")


def test_data_values_must_be_ndarrays():
    meta, data = _explicit_graph()
    data["pos"] = [[0.0, 0.0, 0.0]]

    _assert_invalid(meta, data, "numpy.ndarray")


def test_feature_keys_accept_only_core_known_ordered_lists():
    meta, data = _explicit_graph()
    normalized = validate_schema_v2(meta, data, strict=True)
    assert normalized["feature_keys"]["node_attr_keys"] == [
        "residue.mass",
        "residue.charge",
    ]

    meta["feature_keys"]["matrix_key"] = "matrix"
    _assert_invalid(meta, data, "legacy")


def test_empty_feature_keys_are_valid_and_not_inferred():
    meta, data = _learned_embedding()

    normalized = validate_schema_v2(meta, data, strict=True)

    assert normalized["feature_keys"] == {}


@pytest.mark.parametrize(
    "field",
    ["node_attr_keys", "grid_channels"],
)
def test_feature_key_cardinality_is_checked_when_applicable(field):
    meta, data = _explicit_graph()
    if field == "grid_channels":
        grid = np.zeros((2, 2, 2, 2), dtype=np.float32)
        data = {"grid": grid}
        meta = _common_meta(data)
        meta["representation_type"] = "grid"
        meta["outputs"] = [{"name": "grid", "role": "grid", "layout": "CXYZ"}]
        meta["feature_keys"] = {"grid_channels": ["occupancy"]}
    else:
        meta["feature_keys"][field] = []

    _assert_invalid(meta, data, field)


def test_counts_require_non_negative_plain_integers_and_allow_empty_mapping():
    meta, data = _explicit_graph()
    meta["counts"] = {}
    validate_schema_v2(meta, data, strict=True)

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
    validate_schema_v2(meta, data, strict=True)

    meta["extensions"] = {"": {}}
    _assert_invalid(meta, data, "namespaces")

    meta["extensions"] = {"org.example.analysis": []}
    _assert_invalid(meta, data, "must be a mapping")


def test_config_fingerprint_is_deterministic_and_key_order_independent():
    left = {"center": True, "nested": {"b": 2, "a": 1}}
    right = {"nested": {"a": 1, "b": 2}, "center": True}

    assert compute_config_fingerprint(
        "residue_knn_graph", left, chain_ids=["A"]
    ) == compute_config_fingerprint("residue_knn_graph", right, chain_ids=["A"])


def test_config_fingerprint_changes_only_for_accepted_identity_inputs():
    baseline = compute_config_fingerprint("residue_knn_graph", {"k": 8}, chain_ids=["A"])

    assert baseline != compute_config_fingerprint("residue_knn_graph", {"k": 9}, chain_ids=["A"])
    assert baseline != compute_config_fingerprint("atom_knn_graph", {"k": 8}, chain_ids=["A"])


def test_strict_validation_checks_config_fingerprint_against_name_and_parameters():
    meta, data = _explicit_graph()
    meta["provenance"]["config_fingerprint"] = "0" * 64

    _assert_invalid(meta, data, "does not match representation_name")


def test_normalization_converts_numpy_scalars_and_tuples_without_mutating_input():
    original = {"values": (np.int64(1), np.float32(2.5)), "enabled": np.bool_(True)}
    snapshot = deepcopy(original)

    normalized = normalize_metadata(original)

    assert normalized == {"values": [1, 2.5], "enabled": True}
    assert isinstance(original["values"], tuple)
    assert original["values"][0] == snapshot["values"][0]


@pytest.mark.parametrize(
    "value,match",
    [
        (Path("artifact"), "Path"),
        (float("nan"), "finite"),
    ],
)
def test_normalization_rejects_unsupported_metadata(value, match):
    with pytest.raises(ArtifactSchemaV2ValidationError, match=match):
        normalize_metadata(value)


def test_learned_node_ids_must_align_be_unique_and_use_safe_unicode():
    meta, data = _learned_embedding()
    data["node_ids"] = np.array(["A:1"], dtype=np.str_)
    meta["shapes"]["node_ids"] = [1]
    _assert_invalid(meta, data, "must align")

    meta, data = _learned_embedding()
    data["node_ids"] = np.array(["A:1", "A:1"], dtype=np.str_)
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


@pytest.mark.parametrize("missing", ["residue_ids", "resname"])
def test_residue_validation_reports_missing_key(missing):
    from eris.core.contracts.validations import validate_residue_data

    data = {"residue_ids": np.array(["A:1::GLY"]), "resname": np.array(["GLY"])}
    del data[missing]

    with pytest.raises(ValueError, match=f"missing {missing}"):
        validate_residue_data(data)
