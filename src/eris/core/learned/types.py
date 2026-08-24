from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eris.core.learned.config import OutputMode
from eris.exceptions import ExecutionError


@dataclass(frozen=True, kw_only=True)
class StructuralEmbeddingResult:
    """Framework-independent outputs returned by a structural embedder backend."""

    node_embedding: np.ndarray | None = None
    global_embedding: np.ndarray | None = None
    node_ids: np.ndarray | None = None
    valid_mask: np.ndarray | None = None


def _require_array(value: object, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ExecutionError(f"{name} must be a NumPy array")
    return value


def _validate_node_embedding(value: object) -> np.ndarray:
    array = _require_array(value, "node_embedding")
    if array.dtype.kind != "f":
        raise ExecutionError("node_embedding must have a floating dtype")
    if array.ndim != 2:
        raise ExecutionError("node_embedding must be two-dimensional")
    if 0 in array.shape:
        raise ExecutionError("node_embedding dimensions must be non-zero")
    if not np.isfinite(array).all():
        raise ExecutionError("node_embedding must contain only finite values")
    return array


def _validate_global_embedding(value: object) -> np.ndarray:
    array = _require_array(value, "global_embedding")
    if array.dtype.kind != "f":
        raise ExecutionError("global_embedding must have a floating dtype")
    if array.ndim != 1:
        raise ExecutionError("global_embedding must be one-dimensional")
    if array.shape[0] == 0:
        raise ExecutionError("global_embedding dimension must be non-zero")
    if not np.isfinite(array).all():
        raise ExecutionError("global_embedding must contain only finite values")
    return array


def _validate_node_ids(value: object) -> np.ndarray:
    array = _require_array(value, "node_ids")
    if array.dtype.kind != "U":
        raise ExecutionError("node_ids must have a NumPy Unicode dtype")
    if array.ndim != 1:
        raise ExecutionError("node_ids must be one-dimensional")
    identifiers = array.tolist()
    if any(not identifier.strip() for identifier in identifiers):
        raise ExecutionError("node_ids must contain only non-empty values")
    if len(identifiers) != len(set(identifiers)):
        raise ExecutionError("node_ids must contain unique values")
    return array


def _validate_valid_mask(value: object) -> np.ndarray:
    array = _require_array(value, "valid_mask")
    if array.dtype.kind != "b":
        raise ExecutionError("valid_mask must have a bool dtype")
    if array.ndim != 1:
        raise ExecutionError("valid_mask must be one-dimensional")
    return array


def validate_embedding_result(
    result: StructuralEmbeddingResult,
    *,
    output_mode: OutputMode,
) -> None:
    """Validate backend output shape, dtype, alignment, and requested output mode."""

    if not isinstance(result, StructuralEmbeddingResult):
        raise ExecutionError("result must be a StructuralEmbeddingResult")
    if not isinstance(output_mode, str) or output_mode not in {"node", "global", "both"}:
        raise ExecutionError("output_mode must be one of: both, global, node")

    node_embedding = (
        _validate_node_embedding(result.node_embedding)
        if result.node_embedding is not None
        else None
    )
    global_embedding = (
        _validate_global_embedding(result.global_embedding)
        if result.global_embedding is not None
        else None
    )
    node_ids = _validate_node_ids(result.node_ids) if result.node_ids is not None else None
    valid_mask = _validate_valid_mask(result.valid_mask) if result.valid_mask is not None else None

    if node_embedding is not None:
        if node_ids is None:
            raise ExecutionError("node_embedding requires node_ids")
        if node_ids.shape[0] != node_embedding.shape[0]:
            raise ExecutionError("node_ids must align with node_embedding rows")
    elif node_ids is not None:
        raise ExecutionError("node_ids requires node_embedding")

    if valid_mask is not None:
        if node_embedding is None:
            raise ExecutionError("valid_mask requires node_embedding")
        if valid_mask.shape[0] != node_embedding.shape[0]:
            raise ExecutionError("valid_mask must align with node_embedding rows")

    if output_mode == "node":
        if node_embedding is None:
            raise ExecutionError("output_mode='node' requires node_embedding and node_ids")
        if global_embedding is not None:
            raise ExecutionError("output_mode='node' forbids global_embedding")
    elif output_mode == "global":
        if global_embedding is None:
            raise ExecutionError("output_mode='global' requires global_embedding")
        if node_embedding is not None:
            raise ExecutionError("output_mode='global' forbids node_embedding")
    else:
        if node_embedding is None or global_embedding is None:
            raise ExecutionError(
                "output_mode='both' requires node_embedding, node_ids, and global_embedding"
            )
