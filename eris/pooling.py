from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from eris.core.artifact_schema_v2 import (
    build_artifact_metadata,
    build_feature_keys,
    build_provenance,
    canonical_json,
)
from eris.core.artifacts import StructureArtifact
from eris.exceptions import ConfigError
from eris.features.utils import normalize_choice

_POOLING_MODES = {"mean", "sum", "max", "mean_std"}
_NAN_POLICIES = {"omit", "propagate", "error"}


def _canonical_array_sha256(array: np.ndarray) -> str:
    dtype = array.dtype
    canonical_dtype = dtype if dtype.byteorder == "|" else dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(canonical_dtype, copy=False))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _source_artifact_fingerprint(
    artifact: StructureArtifact,
    *,
    source: str,
    feature_keys: list[str],
) -> str:
    array = artifact.data[source]
    meta = artifact.meta
    input_meta = meta["input"]
    payload = {
        "schema_version": meta["schema_version"],
        "representation_identity": {
            "name": meta["representation_name"],
            "family": meta["representation_family"],
            "type": meta["representation_type"],
            "granularity": meta["granularity"],
            "coordinate_source": meta["coordinate_source"],
        },
        "encoder_identity": meta["encoder"],
        "input_identity": {
            "source_sha256": input_meta["source_sha256"],
            "model_index": input_meta["model_index"],
            "chain_ids": input_meta["chain_ids"],
        },
        "source_config_fingerprint": meta["provenance"]["config_fingerprint"],
        "selected_array": {
            "name": source,
            "feature_keys": feature_keys,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "canonical_byte_sha256": _canonical_array_sha256(array),
        },
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _source_values_and_keys(
    artifact: StructureArtifact,
    source: str,
) -> tuple[np.ndarray, list[str]]:
    if source not in artifact.data:
        raise ConfigError(f"pooling source array {source!r} is not present in the artifact")
    if artifact.meta["granularity"] != "residue":
        raise ConfigError("pooling requires an artifact with residue granularity")

    declaration = next(
        (output for output in artifact.meta["outputs"] if output["name"] == source),
        None,
    )
    if declaration is None or declaration["role"] != "node_features":
        raise ConfigError(f"pooling source {source!r} must be declared with role='node_features'")
    if declaration.get("granularity", artifact.meta["granularity"]) != "residue":
        raise ConfigError(f"pooling source {source!r} must be residue-aligned")

    values = np.asarray(artifact.data[source])
    if values.ndim != 2:
        raise ConfigError(f"pooling source {source!r} must have shape (N, D)")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ConfigError(f"pooling source {source!r} must have N,D > 0")
    if values.dtype.kind not in {"i", "u", "f"}:
        raise ConfigError(f"pooling source {source!r} must have a numerical dtype")
    if np.isinf(values).any():
        raise ConfigError(f"pooling source {source!r} must not contain infinities")

    keys = artifact.meta["feature_keys"].get("node_attr_keys")
    if not isinstance(keys, list) or len(keys) != values.shape[1]:
        raise ConfigError(
            f"pooling source {source!r} requires exactly one ordered key per feature column"
        )
    return values, list(keys)


def _reduce_omit(values: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray | None]:
    valid = ~np.isnan(values)
    counts = valid.sum(axis=0)
    has_value = counts > 0
    sums = np.where(valid, values, 0.0).sum(axis=0, dtype=np.float64)

    if mode in {"mean", "mean_std"}:
        primary = np.full(values.shape[1], np.nan, dtype=np.float64)
        primary[has_value] = sums[has_value] / counts[has_value]
    elif mode == "sum":
        primary = sums
        primary[~has_value] = np.nan
    else:
        primary = np.full(values.shape[1], np.nan, dtype=np.float64)
        for column in np.flatnonzero(has_value):
            primary[column] = np.max(values[valid[:, column], column])

    secondary = None
    if mode == "mean_std":
        centered = np.where(valid, values - primary, 0.0)
        variance = np.full(values.shape[1], np.nan, dtype=np.float64)
        variance[has_value] = np.square(centered[:, has_value]).sum(axis=0) / counts[has_value]
        secondary = np.sqrt(variance)
    return primary, secondary


def _reduce(values: np.ndarray, mode: str, nan_policy: str) -> np.ndarray:
    numeric = values.astype(np.float64, copy=True)
    has_nan = np.isnan(numeric).any()
    if nan_policy == "error" and has_nan:
        raise ConfigError("pooling source contains NaN while nan_policy='error'")

    if nan_policy == "omit":
        primary, secondary = _reduce_omit(numeric, mode)
    else:
        reducers = {
            "mean": np.mean,
            "sum": np.sum,
            "max": np.max,
            "mean_std": np.mean,
        }
        primary = np.asarray(reducers[mode](numeric, axis=0), dtype=np.float64)
        secondary = (
            np.asarray(np.std(numeric, axis=0, ddof=0), dtype=np.float64)
            if mode == "mean_std"
            else None
        )
    if secondary is not None:
        return np.concatenate([primary, secondary]).astype(np.float64)
    return primary.astype(np.float64)


def _pooled_keys(mode: str, source_keys: list[str]) -> list[str]:
    if mode == "mean_std":
        return [
            *(f"pool.mean.{key}" for key in source_keys),
            *(f"pool.std.{key}" for key in source_keys),
        ]
    return [f"pool.{mode}.{key}" for key in source_keys]


def pool(
    artifact: StructureArtifact | str | Path,
    *,
    source: str = "node_attr",
    mode: str = "mean",
    nan_policy: str = "omit",
    output: str | Path | None = None,
) -> StructureArtifact:
    """Pool one residue-aligned feature matrix into a new structure-level artifact."""
    if isinstance(artifact, (str, Path)):
        source_artifact = StructureArtifact.load(Path(artifact), strict=True)
    elif isinstance(artifact, StructureArtifact):
        source_artifact = artifact
        source_artifact.validate(strict=True)
    else:
        raise TypeError("artifact must be a StructureArtifact or artifact-directory path")
    if not isinstance(source, str) or not source.strip():
        raise TypeError("source must be a non-empty string")
    if output is not None and not isinstance(output, (str, Path)):
        raise TypeError("output must be a string, pathlib.Path, or None")

    mode = normalize_choice("mode", mode, sorted(_POOLING_MODES), error=ConfigError)
    nan_policy = normalize_choice(
        "nan_policy", nan_policy, sorted(_NAN_POLICIES), error=ConfigError
    )
    values, source_keys = _source_values_and_keys(source_artifact, source)
    pooled_values = _reduce(values, mode, nan_policy)
    pooled_keys = _pooled_keys(mode, source_keys)
    source_fingerprint = _source_artifact_fingerprint(
        source_artifact,
        source=source,
        feature_keys=source_keys,
    )
    parameters: dict[str, Any] = {
        "mode": mode,
        "nan_policy": nan_policy,
        "source": source,
        "source_artifact_fingerprint": source_fingerprint,
        "source_feature_keys": source_keys,
    }
    if mode == "mean_std":
        parameters["std_ddof"] = 0

    data = {"pooled_attr": pooled_values}
    chain_ids = source_artifact.meta["input"]["chain_ids"]
    meta = build_artifact_metadata(
        data=data,
        representation_name="residue_pool",
        representation_family="explicit",
        representation_type="vector",
        granularity="structure",
        coordinate_source=source_artifact.meta["coordinate_source"],
        encoder={"implementation": "eris.pooling.pool", "version": "0.1.0"},
        input_metadata=deepcopy(source_artifact.meta["input"]),
        parameters=parameters,
        provenance=build_provenance(
            representation_name="residue_pool",
            parameters=parameters,
            chain_ids=chain_ids,
        ),
        output_declarations=[
            {
                "name": "pooled_attr",
                "role": "pooled_features",
                "granularity": "structure",
                "layout": "F",
            }
        ],
        feature_keys=build_feature_keys(pooled_attr_keys=pooled_keys),
        counts={"n_features": int(pooled_values.shape[0])},
    )
    pooled_artifact = StructureArtifact(data=data, meta=meta)
    if output is not None:
        pooled_artifact.save(Path(output))
    return pooled_artifact


__all__ = ["pool"]
