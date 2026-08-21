from __future__ import annotations

import json
from collections import OrderedDict
from copy import deepcopy
from types import MappingProxyType

import numpy as np
import pytest

from eris.core.artifact_schema_v2 import (
    ARTIFACT_SCHEMA_V2_VERSION,
    ArtifactSchemaV2ValidationError,
    build_counts,
    build_dtypes,
    build_feature_keys,
    build_input_metadata,
    build_outputs,
    build_provenance,
    build_shapes,
    validate_schema_v2,
)


def _metadata(
    *,
    data: dict[str, np.ndarray],
    declarations: list[dict],
    feature_keys: dict[str, list[str]],
    counts: dict[str, int],
    representation_name: str = "synthetic_explicit",
    representation_family: str = "explicit",
    representation_type: str = "graph",
    granularity: str = "residue",
) -> dict:
    parameters = {"synthetic": True}
    metadata = {
        "schema_version": ARTIFACT_SCHEMA_V2_VERSION,
        "representation_name": representation_name,
        "representation_family": representation_family,
        "representation_type": representation_type,
        "granularity": granularity,
        "coordinate_source": "model_specific" if representation_family == "learned" else "ca",
        "encoder": {
            "implementation": "tests.synthetic.OutputProducer",
            "version": "1.0",
        },
        "input": build_input_metadata(
            source="memory://synthetic",
            source_sha256="a" * 64,
            input_format="pdb",
            model_index=0,
            chain_id=None,
        ),
        "parameters": parameters,
        "provenance": build_provenance(
            representation_name=representation_name,
            parameters=parameters,
        ),
        "outputs": build_outputs(data=data, declarations=declarations),
        "feature_keys": feature_keys,
        "shapes": build_shapes(data),
        "dtypes": build_dtypes(data),
        "counts": build_counts(counts),
    }
    if representation_family == "learned":
        metadata["model"] = {
            "architecture": "synthetic-structure-model",
            "checkpoint": {"identifier": "synthetic-checkpoint"},
        }
    return metadata


def _explicit_graph() -> tuple[dict, dict[str, np.ndarray]]:
    data = {
        "pos": np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32),
        "node_attr": np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32),
        "edge_index": np.array([[0, 1, 2], [1, 0, 2]], dtype=np.int64),
        "edge_dist": np.array([1.0, 1.0, 0.0], dtype=np.float32),
        "edge_attr": np.array([[1.0], [1.0], [0.0]], dtype=np.float32),
        "graph_attr": np.array([3.0, 3.0], dtype=np.float32),
        "residue_ids": np.array(["A:1", "A:2", "A:3"], dtype=np.str_),
    }
    declarations = [
        {"name": "pos", "role": "coordinates", "granularity": "residue", "layout": "N3"},
        {
            "name": "node_attr",
            "role": "node_features",
            "granularity": "residue",
            "layout": "NF",
        },
        {"name": "edge_index", "role": "edge_index", "layout": "2E"},
        {"name": "edge_dist", "role": "edge_distance", "layout": "E"},
        {"name": "edge_attr", "role": "edge_features", "layout": "EF"},
        {"name": "graph_attr", "role": "graph_features", "layout": "F"},
        {"name": "residue_ids", "role": "node_ids", "granularity": "residue"},
    ]
    metadata = _metadata(
        data=data,
        declarations=declarations,
        feature_keys=build_feature_keys(
            node_attr_keys=["mass", "charge"],
            edge_attr_keys=["distance"],
            graph_attr_keys=["n_nodes", "n_edges"],
        ),
        counts={"n_nodes": 3, "n_edges": 3, "n_self_loops": 1},
    )
    return metadata, data


def test_build_outputs_preserves_declaration_order_not_data_order_and_copies():
    data = OrderedDict(
        [
            ("centroid", np.zeros((1, 3), dtype=np.float32)),
            ("pos", np.zeros((2, 3), dtype=np.float32)),
        ]
    )
    declarations = [
        {
            "name": "pos",
            "role": "coordinates",
            "granularity": "residue",
            "layout": "N3",
            "description": "Normalized coordinates",
        },
        {"name": "centroid", "role": "centroid"},
    ]

    outputs = build_outputs(data=data, declarations=declarations)
    declarations[0]["role"] = "changed"

    assert [output["name"] for output in outputs] == ["pos", "centroid"]
    assert list(data) == ["centroid", "pos"]
    assert outputs[0]["role"] == "coordinates"
    assert outputs is not declarations
    json.dumps(outputs)


@pytest.mark.parametrize(
    ("declarations", "match"),
    [
        ([], "non-empty ordered list"),
        (
            [{"name": "value", "role": "score"}, {"name": "value", "role": "score"}],
            "names must be unique",
        ),
        ([{"name": "Bad-Name", "role": "score"}], "must match"),
        ([{"name": "value", "role": ""}], "role must be a non-empty string"),
        ([{"name": "value", "role": "score", "granularity": "chain"}], "granularity"),
        ([{"name": "value", "role": "score", "layout": ""}], "layout"),
        ([{"name": "value", "role": "score", "description": ""}], "description"),
        ([{"name": "value", "role": "score", "unknown": True}], "unknown fields"),
    ],
)
def test_build_outputs_rejects_invalid_declarations(declarations, match):
    with pytest.raises(ArtifactSchemaV2ValidationError, match=match):
        build_outputs(data={"value": np.zeros(1, dtype=np.float32)}, declarations=declarations)


def test_build_outputs_requires_exact_declaration_data_coverage():
    data = {"first": np.zeros(1), "second": np.zeros(1)}

    with pytest.raises(ArtifactSchemaV2ValidationError, match="missing declarations"):
        build_outputs(data=data, declarations=[{"name": "first", "role": "score"}])
    with pytest.raises(ArtifactSchemaV2ValidationError, match="nonexistent arrays"):
        build_outputs(
            data={"first": data["first"]},
            declarations=[
                {"name": "first", "role": "score"},
                {"name": "second", "role": "score"},
            ],
        )


def test_build_outputs_requires_a_sequence_of_mappings():
    data = {"value": np.zeros(1)}
    with pytest.raises(ArtifactSchemaV2ValidationError, match="sequence of mappings"):
        build_outputs(data=data, declarations="value")
    with pytest.raises(ArtifactSchemaV2ValidationError, match="must be a mapping"):
        build_outputs(data=data, declarations=["value"])


def test_shapes_cover_all_arrays_preserve_order_and_support_scalars():
    data = OrderedDict(
        [
            ("scalar", np.array(1.0, dtype=np.float32)),
            ("vector", np.zeros(3, dtype=np.int64)),
            ("matrix", np.zeros((2, 4), dtype=np.float16)),
            ("grid", np.zeros((3, 4, 5, 6), dtype=np.float32)),
        ]
    )
    original = {name: array.copy() for name, array in data.items()}

    shapes = build_shapes(data)

    assert shapes == {
        "scalar": [],
        "vector": [3],
        "matrix": [2, 4],
        "grid": [3, 4, 5, 6],
    }
    assert all(type(dimension) is int for shape in shapes.values() for dimension in shape)
    for name in data:
        np.testing.assert_array_equal(data[name], original[name])


@pytest.mark.parametrize("builder", [build_shapes, build_dtypes])
def test_structural_builders_reject_non_arrays_and_unsafe_dtypes(builder):
    with pytest.raises(ArtifactSchemaV2ValidationError, match="must be a numpy.ndarray"):
        builder({"value": [1, 2]})
    with pytest.raises(ArtifactSchemaV2ValidationError, match="prohibited dtype"):
        builder({"value": np.array([object()], dtype=object)})


def test_dtypes_use_exact_numpy_strings_and_preserve_unicode_width():
    data = OrderedDict(
        [
            ("float32_value", np.zeros(1, dtype=np.float32)),
            ("float16_value", np.zeros(1, dtype=np.float16)),
            ("int64_value", np.zeros(1, dtype=np.int64)),
            ("uint8_value", np.zeros(1, dtype=np.uint8)),
            ("bool_value", np.zeros(1, dtype=np.bool_)),
            ("unicode_value", np.array(["ALA"], dtype="<U3")),
        ]
    )

    assert build_dtypes(data) == {
        "float32_value": "float32",
        "float16_value": "float16",
        "int64_value": "int64",
        "uint8_value": "uint8",
        "bool_value": "bool",
        "unicode_value": "<U3",
    }


def test_feature_keys_are_explicit_ordered_and_defensively_copied():
    node = ["mass", "charge"]
    edge = ["distance", "rbf.0"]
    graph = ["n_nodes"]
    channels = ["occupancy", "confidence"]

    feature_keys = build_feature_keys(
        node_attr_keys=node,
        edge_attr_keys=edge,
        graph_attr_keys=graph,
        grid_channels=channels,
    )
    node.append("mutated")

    assert feature_keys == {
        "node_attr_keys": ["mass", "charge"],
        "edge_attr_keys": ["distance", "rbf.0"],
        "graph_attr_keys": ["n_nodes"],
        "grid_channels": ["occupancy", "confidence"],
    }
    assert build_feature_keys() == {}
    assert build_feature_keys(node_attr_keys=[]) == {"node_attr_keys": []}
    assert build_feature_keys(node_attr_keys=("second", "first"))["node_attr_keys"] == [
        "second",
        "first",
    ]


@pytest.mark.parametrize("value", ["mass", ["mass", ""], ["mass", 1]])
def test_feature_keys_reject_invalid_sequences_or_members(value):
    with pytest.raises(ArtifactSchemaV2ValidationError, match="sequence|non-empty"):
        build_feature_keys(node_attr_keys=value)


def test_feature_keys_api_cannot_emit_legacy_layout_fields():
    with pytest.raises(TypeError, match="unexpected keyword"):
        build_feature_keys(matrix_key="matrix")
    assert build_feature_keys() == {}  # matrices and learned embeddings


def test_counts_are_producer_declared_ordered_and_defensively_copied():
    source = OrderedDict([("n_nodes", 3), ("n_edges", 2), ("n_self_loops", 0)])

    counts = build_counts(source)
    source["n_nodes"] = 99

    assert list(counts) == ["n_nodes", "n_edges", "n_self_loops"]
    assert counts == {"n_nodes": 3, "n_edges": 2, "n_self_loops": 0}
    assert build_counts({}) == {}
    assert build_counts(MappingProxyType({"n_nodes": 2})) == {"n_nodes": 2}


@pytest.mark.parametrize(
    ("counts", "match"),
    [
        ({"n_nodes": True}, "integer"),
        ({"n_nodes": np.int64(2)}, "integer"),
        ({"n_nodes": -1}, "integer"),
        ({"": 1}, "non-empty strings"),
    ],
)
def test_counts_reject_invalid_names_and_values(counts, match):
    with pytest.raises(ArtifactSchemaV2ValidationError, match=match):
        build_counts(counts)


def test_explicit_graph_builders_produce_strict_valid_metadata():
    metadata, data = _explicit_graph()

    normalized = validate_schema_v2(metadata, data, strict=True)

    assert normalized["counts"] == {"n_nodes": 3, "n_edges": 3, "n_self_loops": 1}
    assert [output["name"] for output in normalized["outputs"]] == list(data)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("n_nodes", 2, "n_nodes"),
        ("n_edges", 2, "n_edges"),
        ("n_self_loops", 0, "n_self_loops"),
    ],
)
def test_explicit_graph_strict_validation_rejects_wrong_counts(field, value, match):
    metadata, data = _explicit_graph()
    metadata["counts"][field] = value

    with pytest.raises(ArtifactSchemaV2ValidationError, match=match):
        validate_schema_v2(metadata, data, strict=True)


def _matrix_metadata(*, binned: bool = False) -> tuple[dict, dict[str, np.ndarray]]:
    matrix_name = "dist_bins" if binned else "dist"
    matrix = np.array([[0, 1], [1, 0]], dtype=np.uint8 if binned else np.float32)
    data = {
        matrix_name: matrix,
        "pos": np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        "residue_ids": np.array(["A:1", "A:2"], dtype=np.str_),
    }
    declarations = [
        {"name": matrix_name, "role": "matrix", "granularity": "residue", "layout": "NN"},
        {"name": "pos", "role": "coordinates", "granularity": "residue", "layout": "N3"},
        {"name": "residue_ids", "role": "node_ids", "granularity": "residue", "layout": "N"},
    ]
    if binned:
        data["bin_edges"] = np.array([0.0, 4.0, 8.0], dtype=np.float32)
        declarations.append({"name": "bin_edges", "role": "bin_edges"})
    metadata = _metadata(
        data=data,
        declarations=declarations,
        feature_keys=build_feature_keys(),
        counts={"n_nodes": 2},
        representation_name="synthetic_matrix",
        representation_type="matrix",
    )
    return metadata, data


def test_matrix_and_binned_matrix_are_strict_valid_with_empty_feature_keys():
    for binned in (False, True):
        metadata, data = _matrix_metadata(binned=binned)

        validate_schema_v2(metadata, data, strict=True)

        assert metadata["feature_keys"] == {}
        if binned:
            assert {output["name"] for output in metadata["outputs"]} >= {"dist_bins", "bin_edges"}


def test_matrix_dimension_conflicting_with_n_nodes_fails():
    metadata, data = _matrix_metadata()
    metadata["counts"]["n_nodes"] = 3

    with pytest.raises(ArtifactSchemaV2ValidationError, match="n_nodes"):
        validate_schema_v2(metadata, data, strict=True)


def test_voxel_effective_arrays_are_outputs_with_channels_and_empty_counts():
    data = {
        "grid": np.zeros((2, 3, 3, 3), dtype=np.float32),
        "voxel_size": np.array([1.5], dtype=np.float32),
        "origin": np.array([-1.5, -1.5, -1.5], dtype=np.float32),
        "grid_size": np.array([3], dtype=np.int32),
        "residue_ids": np.array(["A:1", "A:2", "A:3"], dtype=np.str_),
    }
    declarations = [
        {"name": "grid", "role": "grid", "layout": "CXYZ"},
        {"name": "voxel_size", "role": "voxel_size"},
        {"name": "origin", "role": "origin"},
        {"name": "grid_size", "role": "grid_size"},
        {"name": "residue_ids", "role": "residue_ids", "granularity": "residue"},
    ]
    metadata = _metadata(
        data=data,
        declarations=declarations,
        feature_keys=build_feature_keys(grid_channels=["occupancy", "confidence"]),
        counts={},
        representation_name="synthetic_grid",
        representation_type="grid",
    )

    validate_schema_v2(metadata, data, strict=True)

    assert metadata["counts"] == {}
    assert set(metadata["shapes"]) == set(data)


def _learned_metadata(kind: str) -> tuple[dict, dict[str, np.ndarray]]:
    node_data = {
        "node_embedding": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "node_ids": np.array(["A:1", "A:2"], dtype=np.str_),
    }
    global_data = {"global_embedding": np.array([0.2, 0.3], dtype=np.float32)}
    if kind == "residue":
        data = node_data
        granularity = "residue"
        counts = {"n_nodes": 2}
    elif kind == "global":
        data = global_data
        granularity = "structure"
        counts = {}
    else:
        data = {
            **node_data,
            **global_data,
            "valid_mask": np.array([True, False], dtype=np.bool_),
        }
        granularity = "mixed"
        counts = {"n_nodes": 2}

    role_by_name = {
        "node_embedding": "node_embedding",
        "node_ids": "node_ids",
        "global_embedding": "global_embedding",
        "valid_mask": "valid_mask",
    }
    layout_by_name = {
        "node_embedding": "ND",
        "node_ids": "N",
        "global_embedding": "D",
        "valid_mask": "N",
    }
    declarations = [
        {"name": name, "role": role_by_name[name], "layout": layout_by_name[name]} for name in data
    ]
    metadata = _metadata(
        data=data,
        declarations=declarations,
        feature_keys=build_feature_keys(),
        counts=counts,
        representation_name=f"synthetic_{kind}_embedding",
        representation_family="learned",
        representation_type="embedding",
        granularity=granularity,
    )
    return metadata, data


@pytest.mark.parametrize("kind", ["residue", "global", "mixed"])
def test_learned_residue_global_and_mixed_are_strict_valid(kind):
    metadata, data = _learned_metadata(kind)

    validate_schema_v2(metadata, data, strict=True)

    assert metadata["feature_keys"] == {}
    assert metadata["counts"] == ({} if kind == "global" else {"n_nodes": 2})


def test_all_builder_results_are_json_safe():
    data = {"value": np.array([1, 2], dtype=np.int64)}
    metadata_parts = {
        "outputs": build_outputs(
            data=data,
            declarations=[{"name": "value", "role": "score", "description": "Values"}],
        ),
        "feature_keys": build_feature_keys(node_attr_keys=[]),
        "shapes": build_shapes(data),
        "dtypes": build_dtypes(data),
        "counts": build_counts({"n_nodes": 2}),
    }

    json.dumps(metadata_parts)


def test_builders_do_not_mutate_inputs():
    data = {"value": np.array([1, 2], dtype=np.int64)}
    declarations = [{"name": "value", "role": "score"}]
    counts = {"n_nodes": 2}
    expected_data = data["value"].copy()
    expected_declarations = deepcopy(declarations)
    expected_counts = counts.copy()

    build_outputs(data=data, declarations=declarations)
    build_shapes(data)
    build_dtypes(data)
    build_counts(counts)

    np.testing.assert_array_equal(data["value"], expected_data)
    assert declarations == expected_declarations
    assert counts == expected_counts
