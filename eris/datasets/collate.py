from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch

_IDENTITY_KEYS = (
    "representation_name",
    "representation_family",
    "representation_type",
    "granularity",
)
_TRACE_KEYS = ("id", "artifact_dir", "meta_path")
_COMMON_METADATA_KEYS = ("parameters", "feature_keys", "outputs")


def _batch_list(batch: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    samples = list(batch)
    if not samples:
        raise ValueError("batch must be non-empty")
    for index, sample in enumerate(samples):
        if sample is None:
            raise ValueError(f"batch[{index}] must not be None")
    return samples


def _require_key(sample: Mapping[str, Any], index: int, key: str) -> Any:
    if key not in sample:
        raise ValueError(f"batch[{index}] missing required key {key!r}")
    value = sample[key]
    if value is None:
        raise ValueError(f"batch[{index}] key {key!r} must not be None")
    return value


def _normalize_metadata(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _normalize_metadata(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_normalize_metadata(item) for item in value]
    if isinstance(value, list):
        return [_normalize_metadata(item) for item in value]
    return value


def _validate_metadata(
    samples: Sequence[Mapping[str, Any]],
    *,
    family: str,
    representation_type: str,
) -> dict[str, Any]:
    first = samples[0]
    metadata: dict[str, Any] = {}

    for key in _IDENTITY_KEYS:
        _require_key(first, 0, key)

    for key in _IDENTITY_KEYS:
        expected = first[key]
        metadata[key] = expected
        for index, sample in enumerate(samples):
            actual = _require_key(sample, index, key)
            if actual != expected:
                raise ValueError(
                    f"mixed representation metadata for {key!r}: "
                    f"batch[0]={expected!r}, batch[{index}]={actual!r}"
                )

    if metadata["representation_family"] != family:
        raise ValueError(
            f"collate expected representation_family={family!r}, got {metadata['representation_family']!r}"
        )
    if metadata["representation_type"] != representation_type:
        raise ValueError(
            f"collate expected representation_type={representation_type!r}, "
            f"got {metadata['representation_type']!r}"
        )

    for key in _TRACE_KEYS:
        present = [key in sample and sample[key] is not None for sample in samples]
        if any(present) and not all(present):
            raise ValueError(f"partial optional metadata presence for key {key!r}")
        if all(present):
            metadata[key] = [_require_key(sample, index, key) for index, sample in enumerate(samples)]

    for key in _COMMON_METADATA_KEYS:
        present = [key in sample and sample[key] is not None for sample in samples]
        if any(present) and not all(present):
            raise ValueError(f"partial optional metadata presence for key {key!r}")
        if not all(present):
            continue
        expected = _normalize_metadata(first[key])
        for index, sample in enumerate(samples[1:], start=1):
            actual = _normalize_metadata(sample[key])
            if actual != expected:
                raise ValueError(f"incompatible metadata for {key!r}: batch[0] does not match batch[{index}]")
        metadata[key] = first[key]

    return metadata


def _tensor(value: Any, *, key: str, index: int) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value)
    except Exception as error:
        raise ValueError(f"batch[{index}] key {key!r} cannot be converted to a tensor") from error
    return tensor


def _require_tensor(
    sample: Mapping[str, Any],
    index: int,
    key: str,
    *,
    ndim: int | None = None,
) -> torch.Tensor:
    tensor = _tensor(_require_key(sample, index, key), key=key, index=index)
    if ndim is not None and tensor.ndim != ndim:
        raise ValueError(f"batch[{index}] key {key!r} must be {ndim}D, got shape {tuple(tensor.shape)}")
    return tensor


def _optional_tensors(
    samples: Sequence[Mapping[str, Any]],
    key: str,
    *,
    ndim: int | None = None,
) -> list[torch.Tensor] | None:
    present = [key in sample and sample[key] is not None for sample in samples]
    if any(present) and not all(present):
        raise ValueError(f"partial optional-field presence for key {key!r}")
    if not all(present):
        return None
    return [_require_tensor(sample, index, key, ndim=ndim) for index, sample in enumerate(samples)]


def _validate_same_dtype_device(tensors: Sequence[torch.Tensor], key: str) -> None:
    first = tensors[0]
    for index, tensor in enumerate(tensors[1:], start=1):
        if tensor.dtype != first.dtype:
            raise ValueError(
                f"dtype mismatch for {key!r}: batch[0]={first.dtype}, batch[{index}]={tensor.dtype}"
            )
        if tensor.device != first.device:
            raise ValueError(
                f"device mismatch for {key!r}: batch[0]={first.device}, batch[{index}]={tensor.device}"
            )


def _validate_same_shape(tensors: Sequence[torch.Tensor], key: str) -> None:
    first_shape = tuple(tensors[0].shape)
    for index, tensor in enumerate(tensors[1:], start=1):
        shape = tuple(tensor.shape)
        if shape != first_shape:
            raise ValueError(f"shape mismatch for {key!r}: batch[0]={first_shape}, batch[{index}]={shape}")


def _validate_width(tensors: Sequence[torch.Tensor], key: str) -> int:
    width = int(tensors[0].shape[1])
    for index, tensor in enumerate(tensors[1:], start=1):
        actual = int(tensor.shape[1])
        if actual != width:
            raise ValueError(f"feature width mismatch for {key!r}: batch[0]={width}, batch[{index}]={actual}")
    return width


def _validate_optional_rows(
    tensors: Sequence[torch.Tensor] | None,
    reference_lengths: Sequence[int],
    key: str,
) -> None:
    if tensors is None:
        return
    for index, (tensor, expected) in enumerate(zip(tensors, reference_lengths, strict=True)):
        if int(tensor.shape[0]) != expected:
            raise ValueError(
                f"batch[{index}] key {key!r} row count {int(tensor.shape[0])} "
                f"does not match expected {expected}"
            )


def _stack_optional(
    tensors: Sequence[torch.Tensor] | None,
    key: str,
) -> torch.Tensor | None:
    if tensors is None:
        return None
    _validate_same_dtype_device(tensors, key)
    _validate_same_shape(tensors, key)
    return torch.stack(list(tensors), dim=0)


def _metadata_output(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return dict(metadata)


def _ptr(lengths: Iterable[int], *, device: torch.device | None = None) -> torch.Tensor:
    values = [0]
    total = 0
    for length in lengths:
        total += int(length)
        values.append(total)
    return torch.tensor(values, dtype=torch.long, device=device)


def _validate_edge_index(edge_index: torch.Tensor, n_nodes: int, index: int) -> None:
    if edge_index.ndim != 2 or tuple(edge_index.shape[:1]) != (2,):
        raise ValueError(
            f"batch[{index}] key 'edge_index' must have shape (2, E), got {tuple(edge_index.shape)}"
        )
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if edge_index.dtype not in integer_dtypes:
        raise ValueError(f"batch[{index}] key 'edge_index' must use an integer dtype")
    if edge_index.numel() == 0:
        return
    min_index = int(edge_index.min().item())
    max_index = int(edge_index.max().item())
    if min_index < 0 or max_index >= n_nodes:
        raise ValueError(f"batch[{index}] key 'edge_index' contains node index outside [0, {n_nodes})")


def collate_graph(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Canonical strict graph collate without PyG.

    Compatible graph samples are concatenated in input order. Feature-width,
    dtype, optional-field, and representation incompatibilities raise
    ``ValueError``; no padding, truncation, or topology invention is performed.
    """
    samples = _batch_list(batch)
    metadata = _validate_metadata(samples, family="explicit", representation_type="graph")

    xs = [_require_tensor(sample, index, "x", ndim=2) for index, sample in enumerate(samples)]
    edge_indices = [_require_tensor(sample, index, "edge_index") for index, sample in enumerate(samples)]
    node_counts = [int(x.shape[0]) for x in xs]
    edge_counts = [int(edge_index.shape[1]) for edge_index in edge_indices]

    _validate_same_dtype_device(xs, "x")
    _validate_width(xs, "x")
    _validate_same_dtype_device(edge_indices, "edge_index")
    for index, (edge_index, n_nodes) in enumerate(zip(edge_indices, node_counts, strict=True)):
        _validate_edge_index(edge_index, n_nodes, index)

    edge_attrs = _optional_tensors(samples, "edge_attr", ndim=2)
    if edge_attrs is not None:
        _validate_same_dtype_device(edge_attrs, "edge_attr")
        _validate_width(edge_attrs, "edge_attr")
        _validate_optional_rows(edge_attrs, edge_counts, "edge_attr")

    edge_dists = _optional_tensors(samples, "edge_dist")
    if edge_dists is not None:
        _validate_same_dtype_device(edge_dists, "edge_dist")
        for index, (edge_dist, expected) in enumerate(zip(edge_dists, edge_counts, strict=True)):
            if edge_dist.ndim not in {1, 2}:
                raise ValueError(
                    f"batch[{index}] key 'edge_dist' must be 1D or 2D, got shape {tuple(edge_dist.shape)}"
                )
            if int(edge_dist.shape[0]) != expected:
                raise ValueError(
                    f"batch[{index}] key 'edge_dist' length {int(edge_dist.shape[0])} "
                    f"does not match edge count {expected}"
                )

    positions = _optional_tensors(samples, "pos", ndim=2)
    if positions is not None:
        _validate_same_dtype_device(positions, "pos")
        for index, (pos, expected) in enumerate(zip(positions, node_counts, strict=True)):
            if tuple(pos.shape) != (expected, 3):
                raise ValueError(
                    f"batch[{index}] key 'pos' must have shape ({expected}, 3), got {tuple(pos.shape)}"
                )

    graph_attrs = _optional_tensors(samples, "graph_attr")
    if graph_attrs is not None:
        graph_attrs = [graph_attr.reshape(-1) for graph_attr in graph_attrs]
        _validate_same_dtype_device(graph_attrs, "graph_attr")
        _validate_same_shape(graph_attrs, "graph_attr")

    offsets = _ptr(node_counts, device=xs[0].device)
    shifted_edges = [edge_index + offsets[index] for index, edge_index in enumerate(edge_indices)]

    out = _metadata_output(metadata)
    out.update(
        {
            "x": torch.cat(xs, dim=0),
            "edge_index": torch.cat(shifted_edges, dim=1),
            "batch": torch.cat(
                [
                    torch.full((count,), index, dtype=torch.long, device=xs[0].device)
                    for index, count in enumerate(node_counts)
                ],
                dim=0,
            ),
            "ptr": offsets,
        }
    )
    if edge_attrs is not None:
        out["edge_attr"] = torch.cat(edge_attrs, dim=0)
    if edge_dists is not None:
        out["edge_dist"] = torch.cat(edge_dists, dim=0)
    if positions is not None:
        out["pos"] = torch.cat(positions, dim=0)
    if graph_attrs is not None:
        out["graph_attr"] = torch.stack(graph_attrs, dim=0)
    return out


def collate_cloud(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Canonical point-cloud collate using concatenation plus sample boundaries."""
    samples = _batch_list(batch)
    metadata = _validate_metadata(samples, family="explicit", representation_type="cloud")

    positions = [_require_tensor(sample, index, "pos", ndim=2) for index, sample in enumerate(samples)]
    node_counts = [int(pos.shape[0]) for pos in positions]
    _validate_same_dtype_device(positions, "pos")
    for index, pos in enumerate(positions):
        if int(pos.shape[1]) != 3:
            raise ValueError(f"batch[{index}] key 'pos' must have shape (N, 3), got {tuple(pos.shape)}")

    xs = _optional_tensors(samples, "x", ndim=2)
    if xs is not None:
        _validate_same_dtype_device(xs, "x")
        _validate_width(xs, "x")
        _validate_optional_rows(xs, node_counts, "x")

    centroids = _optional_tensors(samples, "centroid")
    stacked_centroid = _stack_optional(centroids, "centroid")

    boundaries = _ptr(node_counts, device=positions[0].device)
    out = _metadata_output(metadata)
    out.update(
        {
            "pos": torch.cat(positions, dim=0),
            "batch": torch.cat(
                [
                    torch.full((count,), index, dtype=torch.long, device=positions[0].device)
                    for index, count in enumerate(node_counts)
                ],
                dim=0,
            ),
            "ptr": boundaries,
        }
    )
    if xs is not None:
        out["x"] = torch.cat(xs, dim=0)
    if stacked_centroid is not None:
        out["centroid"] = stacked_centroid
    return out


def collate_matrix(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Canonical matrix collate: stack only identical shapes, never pad."""
    samples = _batch_list(batch)
    metadata = _validate_metadata(samples, family="explicit", representation_type="matrix")

    matrices = [_require_tensor(sample, index, "x") for index, sample in enumerate(samples)]
    for index, matrix in enumerate(matrices):
        if matrix.ndim not in {2, 3}:
            raise ValueError(f"batch[{index}] key 'x' must be 2D or 3D, got shape {tuple(matrix.shape)}")
    _validate_same_dtype_device(matrices, "x")
    _validate_same_shape(matrices, "x")

    matrix_key = _require_key(samples[0], 0, "matrix_key")
    for index, sample in enumerate(samples[1:], start=1):
        actual = _require_key(sample, index, "matrix_key")
        if actual != matrix_key:
            raise ValueError(f"matrix_key mismatch: batch[0]={matrix_key!r}, batch[{index}]={actual!r}")

    ns = [int(_require_key(sample, index, "n")) for index, sample in enumerate(samples)]

    bin_edges = _optional_tensors(samples, "bin_edges")
    if bin_edges is not None:
        _validate_same_dtype_device(bin_edges, "bin_edges")
        _validate_same_shape(bin_edges, "bin_edges")
        reference = bin_edges[0]
        for index, current in enumerate(bin_edges[1:], start=1):
            if not torch.equal(current, reference):
                raise ValueError(f"bin_edges mismatch: batch[0] does not match batch[{index}]")

    positions = _optional_tensors(samples, "pos", ndim=2)
    if positions is not None:
        _validate_same_dtype_device(positions, "pos")
        _validate_same_shape(positions, "pos")

    centroids = _optional_tensors(samples, "centroid")
    stacked_centroid = _stack_optional(centroids, "centroid")

    out = _metadata_output(metadata)
    out.update({"x": torch.stack(matrices, dim=0), "matrix_key": matrix_key, "n": ns})
    if bin_edges is not None:
        out["bin_edges"] = bin_edges[0]
    if positions is not None:
        out["pos"] = torch.stack(positions, dim=0)
    if stacked_centroid is not None:
        out["centroid"] = stacked_centroid
    return out


def collate_grid(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Canonical voxel-grid collate: stack exact-compatible grids only."""
    samples = _batch_list(batch)
    metadata = _validate_metadata(samples, family="explicit", representation_type="grid")

    grids = [_require_tensor(sample, index, "x", ndim=4) for index, sample in enumerate(samples)]
    _validate_same_dtype_device(grids, "x")
    _validate_same_shape(grids, "x")

    grid_sizes = _optional_tensors(samples, "grid_size")
    if grid_sizes is not None:
        _validate_same_dtype_device(grid_sizes, "grid_size")
        _validate_same_shape(grid_sizes, "grid_size")
        expected_size = int(grids[0].shape[-1])
        for index, grid_size in enumerate(grid_sizes):
            values = grid_size.reshape(-1)
            if values.numel() != 1 or int(values[0].item()) != expected_size:
                raise ValueError(
                    f"batch[{index}] key 'grid_size' is inconsistent with grid shape "
                    f"{tuple(grids[index].shape)}"
                )

    stacked_optional: dict[str, torch.Tensor] = {}
    for key in ("voxel_size", "origin", "centroid"):
        stacked = _stack_optional(_optional_tensors(samples, key), key)
        if stacked is not None:
            stacked_optional[key] = stacked

    out = _metadata_output(metadata)
    out["x"] = torch.stack(grids, dim=0)
    if grid_sizes is not None:
        out["grid_size"] = torch.stack(grid_sizes, dim=0)
    out.update(stacked_optional)
    return out


def _learned_output_names(sample: Mapping[str, Any]) -> set[str]:
    outputs = sample.get("outputs")
    if isinstance(outputs, Sequence) and not isinstance(outputs, (str, bytes)):
        names = {
            item.get("name")
            for item in outputs
            if isinstance(item, Mapping) and isinstance(item.get("name"), str)
        }
        if names:
            return names
    return {
        name
        for name in ("node_embedding", "global_embedding", "node_ids", "valid_mask")
        if name in sample and sample[name] is not None
    }


def _learned_layout(sample: Mapping[str, Any], index: int) -> str:
    names = _learned_output_names(sample)
    has_node = "node_embedding" in names
    has_node_ids = "node_ids" in names
    has_global = "global_embedding" in names
    has_valid_mask = "valid_mask" in names

    if has_node != has_node_ids:
        raise ValueError(f"batch[{index}] learned node output requires both 'node_embedding' and 'node_ids'")
    if has_valid_mask and not has_node:
        raise ValueError(f"batch[{index}] key 'valid_mask' requires node embeddings")
    if has_node and has_global:
        return "both"
    if has_node:
        return "node"
    if has_global:
        return "global"
    raise ValueError(f"batch[{index}] learned sample has no recognized embedding outputs")


def _node_ids(value: Any, *, expected: int, index: int) -> list[Any]:
    if isinstance(value, np.ndarray):
        ids = value.reshape(-1).tolist()
    elif isinstance(value, (str, bytes)):
        ids = [value]
    else:
        ids = list(value)
    if len(ids) != expected:
        raise ValueError(
            f"batch[{index}] key 'node_ids' length {len(ids)} does not match node count {expected}"
        )
    return ids


def collate_learned(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Canonical learned embedding collate for node, global, and both outputs."""
    samples = _batch_list(batch)
    metadata = _validate_metadata(samples, family="learned", representation_type="embedding")

    layout = _learned_layout(samples[0], 0)
    for index, sample in enumerate(samples[1:], start=1):
        actual = _learned_layout(sample, index)
        if actual != layout:
            raise ValueError(
                f"learned output layout mismatch: batch[0]={layout!r}, batch[{index}]={actual!r}"
            )

    out = _metadata_output(metadata)

    if layout in {"node", "both"}:
        embeddings = [
            _require_tensor(sample, index, "node_embedding", ndim=2) for index, sample in enumerate(samples)
        ]
        node_counts = [int(embedding.shape[0]) for embedding in embeddings]
        _validate_same_dtype_device(embeddings, "node_embedding")
        _validate_width(embeddings, "node_embedding")
        node_ids = [
            node_id
            for index, sample in enumerate(samples)
            for node_id in _node_ids(
                _require_key(sample, index, "node_ids"),
                expected=node_counts[index],
                index=index,
            )
        ]
        chain_ids = [
            chain_id
            for index, sample in enumerate(samples)
            for chain_id in _node_ids(
                _require_key(sample, index, "chain_id"),
                expected=node_counts[index],
                index=index,
            )
        ]
        valid_masks = _optional_tensors(samples, "valid_mask", ndim=1)
        if valid_masks is not None:
            _validate_same_dtype_device(valid_masks, "valid_mask")
            for index, mask in enumerate(valid_masks):
                if mask.dtype != torch.bool:
                    raise ValueError(f"batch[{index}] key 'valid_mask' must use bool dtype, got {mask.dtype}")
            _validate_optional_rows(valid_masks, node_counts, "valid_mask")

        out.update(
            {
                "node_embedding": torch.cat(embeddings, dim=0),
                "node_ids": node_ids,
                "chain_id": chain_ids,
                "batch": torch.cat(
                    [
                        torch.full(
                            (count,),
                            index,
                            dtype=torch.long,
                            device=embeddings[0].device,
                        )
                        for index, count in enumerate(node_counts)
                    ],
                    dim=0,
                ),
                "ptr": _ptr(node_counts, device=embeddings[0].device),
            }
        )
        if valid_masks is not None:
            out["valid_mask"] = torch.cat(valid_masks, dim=0)

    if layout in {"global", "both"}:
        global_embeddings = [
            _require_tensor(sample, index, "global_embedding", ndim=1) for index, sample in enumerate(samples)
        ]
        _validate_same_dtype_device(global_embeddings, "global_embedding")
        _validate_same_shape(global_embeddings, "global_embedding")
        out["global_embedding"] = torch.stack(global_embeddings, dim=0)

    return out


__all__ = [
    "collate_cloud",
    "collate_graph",
    "collate_grid",
    "collate_learned",
    "collate_matrix",
]
