from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TypedDict, Union

import numpy as np

from eris.exceptions import ErisError

ARTIFACT_SCHEMA_V2_VERSION = 2

RepresentationFamily = Literal["explicit", "learned"]
RepresentationType = Literal["cloud", "graph", "matrix", "grid", "embedding"]
Granularity = Literal["atom", "residue", "structure", "mixed"]
CoordinateSource = Literal["atom", "ca", "residue_mean", "residue_com", "model_specific"]
JSONValue = Union[  # noqa: UP007 - explicit Union keeps the recursive alias runtime-safe
    None, bool, int, float, str, list["JSONValue"], dict[str, "JSONValue"]
]


class EncoderMetadata(TypedDict):
    implementation: str
    version: str


class InputMetadataRequired(TypedDict):
    source: str
    source_sha256: str
    format: str
    model_index: int
    chain_ids: list[str]


class InputMetadata(InputMetadataRequired, total=False):
    original_filename: str
    structure_id: str
    source_database: str


class CheckpointMetadataRequired(TypedDict):
    identifier: str


class CheckpointMetadata(CheckpointMetadataRequired, total=False):
    revision: str
    source: str
    sha256: str


class ModelMetadata(TypedDict):
    architecture: str
    checkpoint: CheckpointMetadata


class ProvenanceMetadataRequired(TypedDict):
    eris_version: str
    config_fingerprint: str


class ProvenanceMetadata(ProvenanceMetadataRequired, total=False):
    environment: dict[str, JSONValue]
    created_at: str
    invocation: JSONValue


class OutputMetadataRequired(TypedDict):
    name: str
    role: str


class OutputMetadata(OutputMetadataRequired, total=False):
    granularity: Granularity
    layout: str
    description: str


class ArtifactSchemaV2MetadataRequired(TypedDict):
    schema_version: int
    representation_name: str
    representation_family: RepresentationFamily
    representation_type: RepresentationType
    granularity: Granularity
    coordinate_source: CoordinateSource
    encoder: EncoderMetadata
    input: InputMetadata
    parameters: dict[str, JSONValue]
    provenance: ProvenanceMetadata
    outputs: list[OutputMetadata]
    feature_keys: dict[str, list[str]]
    shapes: dict[str, list[int]]
    dtypes: dict[str, str]
    counts: dict[str, int]


class ArtifactSchemaV2Metadata(ArtifactSchemaV2MetadataRequired, total=False):
    model: ModelMetadata
    extensions: dict[str, dict[str, JSONValue]]


class ArtifactSchemaV2ValidationError(ErisError):
    """Raised when metadata or array data violates Artifact Schema v2."""


_REQUIRED_TOP_LEVEL = {
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
}
_OPTIONAL_TOP_LEVEL = {"model", "extensions"}
_EXPLICIT_TYPES = {"cloud", "graph", "matrix", "grid"}
_LEARNED_TYPES = {"embedding"}
_GRANULARITIES = {"atom", "residue", "structure", "mixed"}
_COORDINATE_SOURCES = {"atom", "ca", "residue_mean", "residue_com", "model_specific"}
_FEATURE_KEY_FIELDS = {
    "node_attr_keys",
    "edge_attr_keys",
    "graph_attr_keys",
    "grid_channels",
}
_DATA_KEY = re.compile(r"^[a-z][a-z0-9_]*$")
_IMPLEMENTATION_PATH = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_DTYPE_KINDS = {"i", "u", "f", "b", "U"}


def _fail(message: str) -> None:
    raise ArtifactSchemaV2ValidationError(message)


def normalize_metadata(value: Any, *, _path: str = "metadata") -> JSONValue:
    """Return a JSON-safe copy using the controlled Schema v2 normalization rules."""
    if isinstance(value, np.ndarray):
        _fail(f"{_path} must not contain numpy arrays")
    if isinstance(value, Path):
        _fail(f"{_path} must not contain pathlib.Path values")
    if isinstance(value, np.generic):
        return normalize_metadata(value.item(), _path=_path)
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            _fail(f"{_path} must contain only finite numbers")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(f"{_path} mapping keys must be strings")
            normalized[key] = normalize_metadata(item, _path=f"{_path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            normalize_metadata(item, _path=f"{_path}[{index}]") for index, item in enumerate(value)
        ]
    _fail(f"{_path} contains unsupported value type {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize metadata with the canonical JSON form used by Schema v2 fingerprints."""
    normalized = normalize_metadata(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _require_non_empty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{path} must be a non-empty string")
    return value


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{path} must be a mapping")
    return value


def _require_exact_keys(
    value: dict[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    path: str,
) -> None:
    optional = optional or set()
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - required - optional)
    if missing:
        _fail(f"{path} is missing required fields: {missing}")
    if unknown:
        _fail(f"{path} contains unknown fields: {unknown}")


def _require_sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _fail(f"{path} must be a lowercase 64-character SHA-256 hex string")
    return value


def _reject_numpy_in_parameters(value: Any, path: str = "meta.parameters") -> None:
    if isinstance(value, (np.ndarray, np.generic)):
        _fail(f"{path} must not contain NumPy objects")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_numpy_in_parameters(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_numpy_in_parameters(item, f"{path}[{index}]")


def _validated_chain_ids(chain_ids: Sequence[str], path: str) -> list[str]:
    if isinstance(chain_ids, (str, bytes)) or not isinstance(chain_ids, Sequence):
        _fail(f"{path} must be a non-empty sequence[str]")
    normalized = list(chain_ids)
    if not normalized or any(not isinstance(chain, str) for chain in normalized):
        _fail(f"{path} must be a non-empty sequence[str]")
    if len(normalized) != len(set(normalized)):
        _fail(f"{path} must contain unique chain IDs")
    return normalized


def compute_config_fingerprint(
    representation_name: str,
    parameters: Mapping[str, Any],
    *,
    chain_ids: Sequence[str],
) -> str:
    """Hash the representation configuration and canonical resolved chain selection."""
    _require_non_empty_string(representation_name, "representation_name")
    if not isinstance(parameters, Mapping):
        _fail("parameters must be a mapping")
    _reject_numpy_in_parameters(parameters, "parameters")
    payload = {
        "representation_name": representation_name,
        "parameters": parameters,
        "input_selection": {"chain_ids": _validated_chain_ids(chain_ids, "chain_ids")},
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _validate_encoder(encoder: Any) -> None:
    value = _require_mapping(encoder, "meta.encoder")
    _require_exact_keys(value, required={"implementation", "version"}, path="meta.encoder")
    implementation = _require_non_empty_string(
        value["implementation"], "meta.encoder.implementation"
    )
    if _IMPLEMENTATION_PATH.fullmatch(implementation) is None:
        _fail("meta.encoder.implementation must be an import-path-like dotted name")
    _require_non_empty_string(value["version"], "meta.encoder.version")


def _validate_input(input_meta: Any) -> None:
    value = _require_mapping(input_meta, "meta.input")
    required = {"source", "source_sha256", "format", "model_index", "chain_ids"}
    optional = {"original_filename", "structure_id", "source_database"}
    _require_exact_keys(value, required=required, optional=optional, path="meta.input")
    _require_non_empty_string(value["source"], "meta.input.source")
    _require_sha256(value["source_sha256"], "meta.input.source_sha256")
    _require_non_empty_string(value["format"], "meta.input.format")
    if type(value["model_index"]) is not int or value["model_index"] < 0:
        _fail("meta.input.model_index must be an integer >= 0")
    _validated_chain_ids(value["chain_ids"], "meta.input.chain_ids")
    for key in optional & value.keys():
        if not isinstance(value[key], str):
            _fail(f"meta.input.{key} must be a string")


def build_input_metadata(
    *,
    source: str | Path,
    source_sha256: str,
    input_format: str,
    model_index: int,
    chain_ids: Sequence[str],
) -> InputMetadata:
    """Build the canonical five-field Schema v2 input block."""
    if not isinstance(source, (str, Path)):
        _fail("source must be a string or pathlib.Path")
    input_meta: InputMetadata = {
        "source": str(source),
        "source_sha256": source_sha256,
        "format": input_format,
        "model_index": model_index,
        "chain_ids": list(chain_ids),
    }
    _validate_input(input_meta)
    return input_meta


def _validate_created_at(value: Any) -> None:
    _require_non_empty_string(value, "meta.provenance.created_at")
    if not value.endswith("Z"):
        _fail("meta.provenance.created_at must be a UTC ISO-8601 string ending in 'Z'")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        _fail("meta.provenance.created_at must be a valid UTC ISO-8601 string")


def _validate_provenance(provenance: Any) -> None:
    value = _require_mapping(provenance, "meta.provenance")
    _require_exact_keys(
        value,
        required={"eris_version", "config_fingerprint"},
        optional={"environment", "created_at", "invocation"},
        path="meta.provenance",
    )
    _require_non_empty_string(value["eris_version"], "meta.provenance.eris_version")
    _require_sha256(value["config_fingerprint"], "meta.provenance.config_fingerprint")
    if "environment" in value and not isinstance(value["environment"], dict):
        _fail("meta.provenance.environment must be a mapping")
    if "created_at" in value:
        _validate_created_at(value["created_at"])
    if "invocation" in value:
        invocation = value["invocation"]
        if not isinstance(invocation, (dict, str)):
            _fail("meta.provenance.invocation must be a mapping or string")


def build_provenance(
    *,
    representation_name: str,
    parameters: Mapping[str, Any],
    chain_ids: Sequence[str],
) -> ProvenanceMetadata:
    """Build deterministic required provenance without optional environment state."""
    from eris import __version__

    provenance: ProvenanceMetadata = {
        "eris_version": __version__,
        "config_fingerprint": compute_config_fingerprint(
            representation_name,
            parameters,
            chain_ids=chain_ids,
        ),
    }
    _validate_provenance(provenance)
    return provenance


def _validate_model(family: str, model: Any, *, present: bool) -> None:
    if family == "explicit":
        if present:
            _fail("meta.model must be absent for explicit representations")
        return
    if not present:
        _fail("meta.model is required for learned representations")
    value = _require_mapping(model, "meta.model")
    _require_exact_keys(value, required={"architecture", "checkpoint"}, path="meta.model")
    _require_non_empty_string(value["architecture"], "meta.model.architecture")
    checkpoint = _require_mapping(value["checkpoint"], "meta.model.checkpoint")
    _require_exact_keys(
        checkpoint,
        required={"identifier"},
        optional={"revision", "source", "sha256"},
        path="meta.model.checkpoint",
    )
    _require_non_empty_string(checkpoint["identifier"], "meta.model.checkpoint.identifier")
    for key in {"revision", "source"} & checkpoint.keys():
        if not isinstance(checkpoint[key], str):
            _fail(f"meta.model.checkpoint.{key} must be a string")
    if "sha256" in checkpoint:
        _require_sha256(checkpoint["sha256"], "meta.model.checkpoint.sha256")


def _validate_outputs(outputs: Any) -> list[dict[str, Any]]:
    if not isinstance(outputs, list) or not outputs:
        _fail("meta.outputs must be a non-empty ordered list")
    names: list[str] = []
    for index, output in enumerate(outputs):
        path = f"meta.outputs[{index}]"
        value = _require_mapping(output, path)
        _require_exact_keys(
            value,
            required={"name", "role"},
            optional={"granularity", "layout", "description"},
            path=path,
        )
        name = _require_non_empty_string(value["name"], f"{path}.name")
        if _DATA_KEY.fullmatch(name) is None:
            _fail(f"{path}.name must match {_DATA_KEY.pattern}")
        _require_non_empty_string(value["role"], f"{path}.role")
        if "granularity" in value and value["granularity"] not in _GRANULARITIES:
            _fail(f"{path}.granularity must be one of {sorted(_GRANULARITIES)}")
        if "layout" in value:
            _require_non_empty_string(value["layout"], f"{path}.layout")
        if "description" in value:
            _require_non_empty_string(value["description"], f"{path}.description")
        names.append(name)
    if len(names) != len(set(names)):
        _fail("meta.outputs names must be unique")
    return outputs


def _validate_feature_keys(feature_keys: Any) -> None:
    value = _require_mapping(feature_keys, "meta.feature_keys")
    unknown = sorted(value.keys() - _FEATURE_KEY_FIELDS)
    if unknown:
        _fail(f"meta.feature_keys contains unknown or legacy fields: {unknown}")
    for field, keys in value.items():
        if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
            _fail(f"meta.feature_keys.{field} must be an ordered list[str]")


def _validate_shapes(shapes: Any, output_names: set[str]) -> None:
    value = _require_mapping(shapes, "meta.shapes")
    if set(value) != output_names:
        _fail("meta.shapes keys must exactly match output/data names")
    for name, shape in value.items():
        if not isinstance(shape, list):
            _fail(f"meta.shapes.{name} must be a list[int]")
        if any(type(dimension) is not int or dimension < 0 for dimension in shape):
            _fail(f"meta.shapes.{name} dimensions must be integers >= 0")


def _validate_dtypes(dtypes: Any, output_names: set[str]) -> None:
    value = _require_mapping(dtypes, "meta.dtypes")
    if set(value) != output_names:
        _fail("meta.dtypes keys must exactly match output/data names")
    for name, dtype in value.items():
        _require_non_empty_string(dtype, f"meta.dtypes.{name}")


def _validate_counts(counts: Any) -> None:
    value = _require_mapping(counts, "meta.counts")
    for name, count in value.items():
        if not isinstance(name, str) or not name:
            _fail("meta.counts keys must be non-empty strings")
        if type(count) is not int or count < 0:
            _fail(f"meta.counts.{name} must be an integer >= 0")


def _validate_extensions(extensions: Any) -> None:
    value = _require_mapping(extensions, "meta.extensions")
    for namespace, content in value.items():
        if not isinstance(namespace, str) or not namespace.strip():
            _fail("meta.extensions namespaces must be non-empty strings")
        if not isinstance(content, dict):
            _fail(f"meta.extensions.{namespace} must be a mapping")


def _validate_data(data: Any) -> dict[str, np.ndarray]:
    if not isinstance(data, Mapping):
        _fail("data must be a mapping[str, numpy.ndarray]")
    validated: dict[str, np.ndarray] = {}
    for name, array in data.items():
        if not isinstance(name, str) or _DATA_KEY.fullmatch(name) is None:
            _fail(f"data key {name!r} must match {_DATA_KEY.pattern}")
        if not isinstance(array, np.ndarray):
            _fail(f"data.{name} must be a numpy.ndarray")
        if array.dtype.kind not in _ALLOWED_DTYPE_KINDS:
            _fail(f"data.{name} has prohibited dtype {array.dtype}")
        validated[name] = array
    return validated


def build_outputs(
    *,
    data: Mapping[str, np.ndarray],
    declarations: Sequence[Mapping[str, Any]],
) -> list[OutputMetadata]:
    """Copy explicit ordered output declarations and verify complete payload coverage."""
    arrays = _validate_data(data)
    if not isinstance(declarations, Sequence) or isinstance(declarations, (str, bytes)):
        _fail("output declarations must be a non-empty sequence of mappings")

    outputs: list[dict[str, Any]] = []
    for index, declaration in enumerate(declarations):
        if not isinstance(declaration, Mapping):
            _fail(f"output declarations[{index}] must be a mapping")
        copied = normalize_metadata(dict(declaration), _path=f"output declarations[{index}]")
        if not isinstance(copied, dict):  # pragma: no cover - guaranteed by the mapping check
            _fail(f"output declarations[{index}] must normalize to a mapping")
        outputs.append(copied)

    _validate_outputs(outputs)
    output_names = {output["name"] for output in outputs}
    data_names = set(arrays)
    if output_names != data_names:
        missing = sorted(data_names - output_names)
        extra = sorted(output_names - data_names)
        _fail(
            "output declaration names must exactly match data keys "
            f"(missing declarations: {missing}; nonexistent arrays: {extra})"
        )
    return outputs  # type: ignore[return-value]


def build_shapes(data: Mapping[str, np.ndarray]) -> dict[str, list[int]]:
    """Derive JSON-safe shapes for every validated payload array."""
    arrays = _validate_data(data)
    return {name: [int(dimension) for dimension in array.shape] for name, array in arrays.items()}


def build_dtypes(data: Mapping[str, np.ndarray]) -> dict[str, str]:
    """Derive exact NumPy dtype strings for every validated payload array."""
    arrays = _validate_data(data)
    return {name: str(array.dtype) for name, array in arrays.items()}


def build_feature_keys(
    *,
    node_attr_keys: Sequence[str] | None = None,
    edge_attr_keys: Sequence[str] | None = None,
    graph_attr_keys: Sequence[str] | None = None,
    grid_channels: Sequence[str] | None = None,
) -> dict[str, list[str]]:
    """Copy producer-owned ordered feature semantics without inference or sorting."""
    feature_keys: dict[str, list[str]] = {}
    supplied = {
        "node_attr_keys": node_attr_keys,
        "edge_attr_keys": edge_attr_keys,
        "graph_attr_keys": graph_attr_keys,
        "grid_channels": grid_channels,
    }
    for field, keys in supplied.items():
        if keys is None:
            continue
        if not isinstance(keys, Sequence) or isinstance(keys, (str, bytes)):
            _fail(f"feature_keys.{field} must be an ordered sequence[str]")
        copied = list(keys)
        if any(not isinstance(key, str) or not key for key in copied):
            _fail(f"feature_keys.{field} entries must be non-empty strings")
        feature_keys[field] = copied

    _validate_feature_keys(feature_keys)
    return feature_keys


def build_counts(counts: Mapping[str, int]) -> dict[str, int]:
    """Copy producer-declared semantic counts without deriving values from arrays."""
    if not isinstance(counts, Mapping):
        _fail("counts must be a mapping")
    copied = dict(counts)
    _validate_counts(copied)
    return copied


def build_artifact_metadata(
    *,
    data: Mapping[str, np.ndarray],
    representation_name: str,
    representation_family: str,
    representation_type: str,
    granularity: str,
    coordinate_source: str,
    encoder: Mapping[str, Any],
    input_metadata: Mapping[str, Any],
    parameters: Mapping[str, Any],
    provenance: Mapping[str, Any],
    output_declarations: Sequence[Mapping[str, Any]],
    feature_keys: Mapping[str, Sequence[str]],
    counts: Mapping[str, int],
    model: Mapping[str, Any] | None = None,
    extensions: Mapping[str, Mapping[str, Any]] | None = None,
) -> ArtifactSchemaV2Metadata:
    """Build and strictly validate complete Schema v2 metadata for any producer family."""
    metadata: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_V2_VERSION,
        "representation_name": representation_name,
        "representation_family": representation_family,
        "representation_type": representation_type,
        "granularity": granularity,
        "coordinate_source": coordinate_source,
        "encoder": dict(encoder),
        "input": dict(input_metadata),
        "parameters": dict(parameters),
        "provenance": dict(provenance),
        "outputs": build_outputs(data=data, declarations=output_declarations),
        "feature_keys": {name: list(keys) for name, keys in feature_keys.items()},
        "shapes": build_shapes(data),
        "dtypes": build_dtypes(data),
        "counts": build_counts(counts),
    }
    if model is not None:
        metadata["model"] = dict(model)
    if extensions is not None:
        metadata["extensions"] = {
            namespace: dict(content) for namespace, content in extensions.items()
        }

    return validate_schema_v2(metadata, data, strict=True)


def _outputs_by_role(outputs: list[dict[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for output in outputs:
        result.setdefault(output["role"], []).append(output["name"])
    return result


def _single_role_array(
    role: str,
    roles: dict[str, list[str]],
    data: dict[str, np.ndarray],
) -> np.ndarray | None:
    names = roles.get(role, [])
    if not names:
        return None
    if len(names) > 1:
        _fail(f"strict validation requires at most one output with role={role!r}")
    return data[names[0]]


def _require_shape(array: np.ndarray, expected: tuple[Any, ...], role: str) -> None:
    if array.ndim != len(expected):
        _fail(f"role={role!r} requires {len(expected)} dimensions, got shape {array.shape}")
    for actual, wanted in zip(array.shape, expected, strict=True):
        if wanted is not None and actual != wanted:
            _fail(f"role={role!r} requires shape {expected}, got {array.shape}")


def _validate_strict_roles(
    meta: dict[str, Any],
    data: dict[str, np.ndarray],
    outputs: list[dict[str, Any]],
) -> None:
    roles = _outputs_by_role(outputs)
    arrays = {
        role: _single_role_array(role, roles, data)
        for role in (
            "coordinates",
            "edge_index",
            "edge_distance",
            "node_features",
            "edge_features",
            "graph_features",
            "matrix",
            "grid",
            "node_embedding",
            "global_embedding",
            "node_ids",
            "chain_id",
            "valid_mask",
        )
    }

    coordinates = arrays["coordinates"]
    if coordinates is not None and coordinates.ndim == 0:
        _fail("role='coordinates' requires at least one dimension")

    edge_index = arrays["edge_index"]
    if edge_index is not None:
        _require_shape(edge_index, (2, None), "edge_index")
        if edge_index.dtype.kind not in {"i", "u"}:
            _fail("role='edge_index' requires an integer dtype")
    edge_count = edge_index.shape[1] if edge_index is not None else None

    edge_distance = arrays["edge_distance"]
    if edge_distance is not None:
        _require_shape(edge_distance, (edge_count,), "edge_distance")
    node_features = arrays["node_features"]
    if node_features is not None:
        _require_shape(node_features, (None, None), "node_features")
    edge_features = arrays["edge_features"]
    if edge_features is not None:
        _require_shape(edge_features, (edge_count, None), "edge_features")
    graph_features = arrays["graph_features"]
    if graph_features is not None:
        _require_shape(graph_features, (None,), "graph_features")
    matrix = arrays["matrix"]
    if matrix is not None:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            _fail(f"role='matrix' requires a square 2-D array, got shape {matrix.shape}")
    grid = arrays["grid"]
    if grid is not None:
        _require_shape(grid, (None, None, None, None), "grid")

    node_embedding = arrays["node_embedding"]
    if node_embedding is not None:
        _require_shape(node_embedding, (None, None), "node_embedding")
        if node_embedding.dtype.kind != "f":
            _fail("role='node_embedding' requires a floating dtype")
        if 0 in node_embedding.shape:
            _fail("role='node_embedding' requires N,D > 0")
        if not np.isfinite(node_embedding).all():
            _fail("role='node_embedding' requires finite values")

    global_embedding = arrays["global_embedding"]
    if global_embedding is not None:
        _require_shape(global_embedding, (None,), "global_embedding")
        if global_embedding.dtype.kind != "f":
            _fail("role='global_embedding' requires a floating dtype")
        if global_embedding.shape[0] == 0:
            _fail("role='global_embedding' requires D > 0")
        if not np.isfinite(global_embedding).all():
            _fail("role='global_embedding' requires finite values")

    node_ids = arrays["node_ids"]
    if node_ids is not None:
        _require_shape(node_ids, (None,), "node_ids")
        if node_ids.dtype.kind != "U":
            _fail("role='node_ids' requires a Unicode dtype")
        identifiers = node_ids.tolist()
        if any(not identifier for identifier in identifiers):
            _fail("role='node_ids' requires non-empty identifiers")
        if meta["representation_family"] == "learned" and len(identifiers) != len(set(identifiers)):
            _fail("role='node_ids' requires unique identifiers")

    chain_id = arrays["chain_id"]
    requires_chain_id = meta["granularity"] == "residue" or (
        meta["granularity"] == "mixed" and node_embedding is not None
    )
    if requires_chain_id and chain_id is None:
        _fail("residue-level artifacts require an aligned role='chain_id' output")
    if chain_id is not None:
        _require_shape(chain_id, (None,), "chain_id")
        if chain_id.dtype.kind != "U":
            _fail("role='chain_id' requires a Unicode dtype")
        selected_chain_ids = meta["input"]["chain_ids"]
        aligned_chain_ids = chain_id.tolist()
        if any(identifier not in selected_chain_ids for identifier in aligned_chain_ids):
            _fail("role='chain_id' contains a chain outside meta.input.chain_ids")
        chain_order = {identifier: index for index, identifier in enumerate(selected_chain_ids)}
        aligned_order = [chain_order[identifier] for identifier in aligned_chain_ids]
        if aligned_order != sorted(aligned_order):
            _fail("role='chain_id' rows must preserve meta.input.chain_ids order")

    valid_mask = arrays["valid_mask"]
    if valid_mask is not None:
        _require_shape(valid_mask, (None,), "valid_mask")
        if valid_mask.dtype.kind != "b":
            _fail("role='valid_mask' requires a bool dtype")

    if meta["representation_family"] == "learned" and node_embedding is not None:
        if node_ids is None:
            _fail("learned node_embedding requires a node_ids output")
        if node_ids.shape[0] != node_embedding.shape[0]:
            _fail("learned node_ids must align with node_embedding rows")
    if valid_mask is not None:
        if node_embedding is None:
            _fail("valid_mask requires a node_embedding output")
        if valid_mask.shape[0] != node_embedding.shape[0]:
            _fail("valid_mask must align with node_embedding rows")

    feature_roles = {
        "node_attr_keys": "node_features",
        "edge_attr_keys": "edge_features",
        "graph_attr_keys": "graph_features",
        "grid_channels": "grid",
    }
    for field, role in feature_roles.items():
        if field not in meta["feature_keys"]:
            continue
        array = arrays[role]
        if array is None:
            _fail(f"meta.feature_keys.{field} requires an output with role={role!r}")
        expected = array.shape[0] if role == "graph_features" else array.shape[-1]
        if role == "grid":
            expected = array.shape[0]
        if len(meta["feature_keys"][field]) != expected:
            _fail(
                f"meta.feature_keys.{field} has {len(meta['feature_keys'][field])} entries "
                f"for role={role!r} with {expected} features"
            )

    counts = meta["counts"]
    if edge_count is not None and "n_edges" in counts and counts["n_edges"] != edge_count:
        _fail("meta.counts.n_edges does not match edge_index")
    if edge_index is not None and "n_self_loops" in counts:
        self_loops = int(np.count_nonzero(edge_index[0] == edge_index[1]))
        if counts["n_self_loops"] != self_loops:
            _fail("meta.counts.n_self_loops does not match edge_index")
    node_count_candidates = [
        array.shape[0]
        for array in (
            coordinates,
            node_features,
            node_embedding,
            node_ids,
            chain_id,
            valid_mask,
            matrix,
        )
        if array is not None
    ]
    if node_count_candidates and len(set(node_count_candidates)) != 1:
        _fail("node-oriented outputs do not agree on their first dimension")
    if node_count_candidates and "n_nodes" in counts:
        if counts["n_nodes"] != node_count_candidates[0]:
            _fail("meta.counts.n_nodes does not match node-oriented outputs")


def validate_schema_v2(
    meta: Mapping[str, Any],
    data: Mapping[str, np.ndarray],
    *,
    strict: bool = False,
) -> ArtifactSchemaV2Metadata:
    """Validate a synthetic/runtime Schema v2 mapping and return normalized metadata."""
    if not isinstance(meta, Mapping):
        _fail("meta must be a mapping")
    if "parameters" in meta:
        _reject_numpy_in_parameters(meta["parameters"])
    normalized = normalize_metadata(meta)
    if not isinstance(normalized, dict):
        _fail("meta must normalize to a mapping")

    _require_exact_keys(
        normalized,
        required=_REQUIRED_TOP_LEVEL,
        optional=_OPTIONAL_TOP_LEVEL,
        path="meta",
    )
    if type(normalized["schema_version"]) is not int or normalized["schema_version"] != 2:
        _fail("meta.schema_version must equal exactly 2")
    _require_non_empty_string(normalized["representation_name"], "meta.representation_name")

    family = normalized["representation_family"]
    if family not in {"explicit", "learned"}:
        _fail("meta.representation_family must be 'explicit' or 'learned'")
    representation_type = normalized["representation_type"]
    allowed_types = _EXPLICIT_TYPES if family == "explicit" else _LEARNED_TYPES
    if representation_type not in allowed_types:
        _fail(
            f"meta.representation_type={representation_type!r} is incompatible with "
            f"family={family!r}; expected one of {sorted(allowed_types)}"
        )
    if normalized["granularity"] not in _GRANULARITIES:
        _fail(f"meta.granularity must be one of {sorted(_GRANULARITIES)}")
    if normalized["coordinate_source"] not in _COORDINATE_SOURCES:
        _fail(f"meta.coordinate_source must be one of {sorted(_COORDINATE_SOURCES)}")
    if family == "explicit" and normalized["granularity"] == "atom":
        if normalized["coordinate_source"] != "atom":
            _fail("explicit atom granularity requires coordinate_source='atom'")
    if family == "explicit" and normalized["granularity"] == "residue":
        if normalized["coordinate_source"] not in {"ca", "residue_mean", "residue_com"}:
            _fail(
                "explicit residue granularity requires coordinate_source to be 'ca', "
                "'residue_mean', or 'residue_com'"
            )
    if family == "learned" and normalized["granularity"] in {"structure", "mixed"}:
        if normalized["coordinate_source"] != "model_specific":
            _fail("learned structure/mixed granularity requires coordinate_source='model_specific'")

    _validate_encoder(normalized["encoder"])
    _validate_input(normalized["input"])
    parameters = _require_mapping(normalized["parameters"], "meta.parameters")
    canonical_json(parameters)
    _validate_provenance(normalized["provenance"])
    _validate_model(family, normalized.get("model"), present="model" in normalized)
    outputs = _validate_outputs(normalized["outputs"])
    _validate_feature_keys(normalized["feature_keys"])
    arrays = _validate_data(data)

    output_names = {output["name"] for output in outputs}
    if output_names != set(arrays):
        _fail("meta.outputs names must exactly match data keys")
    _validate_shapes(normalized["shapes"], output_names)
    _validate_dtypes(normalized["dtypes"], output_names)
    _validate_counts(normalized["counts"])
    if "extensions" in normalized:
        _validate_extensions(normalized["extensions"])

    if strict:
        for name, array in arrays.items():
            if normalized["shapes"][name] != list(array.shape):
                _fail(f"meta.shapes.{name} does not match data.{name}.shape")
            if normalized["dtypes"][name] != str(array.dtype):
                _fail(f"meta.dtypes.{name} does not match data.{name}.dtype")
        expected_fingerprint = compute_config_fingerprint(
            normalized["representation_name"],
            normalized["parameters"],
            chain_ids=normalized["input"]["chain_ids"],
        )
        if normalized["provenance"]["config_fingerprint"] != expected_fingerprint:
            _fail(
                "meta.provenance.config_fingerprint does not match representation_name "
                "and canonical parameters"
            )
        _validate_strict_roles(normalized, arrays, outputs)

    return normalized  # type: ignore[return-value]


__all__ = [
    "ARTIFACT_SCHEMA_V2_VERSION",
    "ArtifactSchemaV2Metadata",
    "ArtifactSchemaV2ValidationError",
    "build_artifact_metadata",
    "build_counts",
    "build_dtypes",
    "build_feature_keys",
    "build_input_metadata",
    "build_outputs",
    "build_provenance",
    "build_shapes",
    "canonical_json",
    "compute_config_fingerprint",
    "normalize_metadata",
    "validate_schema_v2",
]
