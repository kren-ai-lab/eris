"""Construction of the frozen GearNet-Edge graph inputs for native inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from eris.core.learned.backends.gearnet.metadata import (
    GEARNET_EDGE_EDGE_INPUT_DIM,
    GEARNET_EDGE_KNN_K,
    GEARNET_EDGE_KNN_MIN_DISTANCE,
    GEARNET_EDGE_NODE_INPUT_DIM,
    GEARNET_EDGE_NUM_ANGLE_BIN,
    GEARNET_EDGE_NUM_RELATION,
    GEARNET_EDGE_SEQUENTIAL_MAX_DISTANCE,
    GEARNET_EDGE_SPATIAL_MAX_NEIGHBORS,
    GEARNET_EDGE_SPATIAL_MIN_DISTANCE,
    GEARNET_EDGE_SPATIAL_RADIUS,
)
from eris.core.learned.backends.gearnet.preprocessing import (
    TransportNodeRecord,
    _validate_transport_records,
)
from eris.exceptions import ExecutionError

_RESIDUE_TO_ID: Final = {
    "GLY": 0,
    "ALA": 1,
    "SER": 2,
    "PRO": 3,
    "VAL": 4,
    "THR": 5,
    "CYS": 6,
    "ILE": 7,
    "LEU": 8,
    "ASN": 9,
    "ASP": 10,
    "GLN": 11,
    "LYS": 12,
    "GLU": 13,
    "MET": 14,
    "HIS": 15,
    "PHE": 16,
    "ARG": 17,
    "TYR": 18,
    "TRP": 19,
}
_KDTREE_LEAF_SIZE: Final = 10


class NativeGearNetGraphError(ExecutionError):
    """Raised when native graph construction cannot preserve frozen semantics."""


@dataclass(frozen=True, slots=True)
class NativeGearNetGraph:
    """The numerical graph arrays required by the native GearNet runtime."""

    node_feature: np.ndarray
    node_position: np.ndarray
    edge_source: np.ndarray
    edge_destination: np.ndarray
    edge_relation: np.ndarray
    edge_feature: np.ndarray
    line_edge_source: np.ndarray
    line_edge_destination: np.ndarray
    line_angle_relation: np.ndarray


@dataclass(slots=True)
class _KdNode:
    indexes: np.ndarray | None = None
    axis: int | None = None
    cut: np.float32 | None = None
    left: _KdNode | None = None
    right: _KdNode | None = None


def _children_near_first(node: _KdNode, query: np.ndarray) -> tuple[_KdNode, _KdNode]:
    assert node.axis is not None and node.cut is not None  # noqa: S101, PT018
    assert node.left is not None and node.right is not None  # noqa: S101, PT018
    if np.float32(2.0) * (query[node.axis] - node.cut) < 0:
        return node.left, node.right
    return node.right, node.left


class _ReferenceKdTree:
    """Minimal CPU radius / kNN traversal matching frozen reference search order."""

    def __init__(self, points: np.ndarray) -> None:
        self.points = np.ascontiguousarray(points, dtype=np.float32)
        if self.points.ndim != 2 or self.points.shape[0] == 0:
            raise NativeGearNetGraphError("reference KD-tree points must be a non-empty matrix")
        indexes = np.arange(self.points.shape[0], dtype=np.int64)
        low = self.points.min(axis=0).astype(np.float32, copy=False)
        high = self.points.max(axis=0).astype(np.float32, copy=False)
        self.root = self._build(indexes, low, high)

    def _build(self, indexes: np.ndarray, low: np.ndarray, high: np.ndarray) -> _KdNode:
        if indexes.shape[0] <= _KDTREE_LEAF_SIZE:
            return _KdNode(indexes=indexes.copy())

        spans = high - low
        max_span = spans.max()
        candidates = np.flatnonzero(spans > np.float32(0.99999) * max_span)
        spreads = [self.points[indexes, axis].max() - self.points[indexes, axis].min() for axis in candidates]
        axis = int(candidates[int(np.argmax(np.asarray(spreads, dtype=np.float32)))])
        minimum = self.points[indexes, axis].min()
        maximum = self.points[indexes, axis].max()
        split = np.float32((low[axis] + high[axis]) / np.float32(2.0))
        cut = minimum if split < minimum else min(split, maximum)
        ordered, lim1, lim2 = self._plane_split(indexes.copy(), axis, cut)
        half = indexes.shape[0] // 2
        split_index = lim1 if lim1 > half else min(half, lim2)

        left_high = high.copy()
        left_high[axis] = cut
        right_low = low.copy()
        right_low[axis] = cut
        return _KdNode(
            axis=axis,
            cut=cut,
            left=self._build(ordered[:split_index], low.copy(), left_high),
            right=self._build(ordered[split_index:], right_low, high.copy()),
        )

    def _plane_split(self, indexes: np.ndarray, axis: int, cut: np.float32) -> tuple[np.ndarray, int, int]:
        left = 0
        right = indexes.shape[0] - 1
        while True:
            while left <= right and self.points[indexes[left], axis] < cut:
                left += 1
            while right and left <= right and self.points[indexes[right], axis] >= cut:
                right -= 1
            if left > right or not right:
                break
            indexes[left], indexes[right] = indexes[right], indexes[left]
            left += 1
            right -= 1
        lim1 = left
        right = indexes.shape[0] - 1
        while True:
            while left <= right and self.points[indexes[left], axis] <= cut:
                left += 1
            while right and left <= right and self.points[indexes[right], axis] > cut:
                right -= 1
            if left > right or not right:
                break
            indexes[left], indexes[right] = indexes[right], indexes[left]
            left += 1
            right -= 1
        return indexes, lim1, left

    def radius(self, query: np.ndarray, radius: float) -> list[int]:
        result: list[int] = []
        radius_squared = np.float32(radius * radius)

        def visit(node: _KdNode) -> None:
            if node.indexes is not None:
                for index in node.indexes:
                    delta = query - self.points[index]
                    distance = np.sum(delta * delta, dtype=np.float32)
                    if distance < radius_squared:
                        result.append(int(index))
                return
            first, second = _children_near_first(node, query)
            visit(first)
            visit(second)

        visit(self.root)
        return result

    def knn(self, query: np.ndarray, count: int) -> list[int]:
        # KNNResultSet maintains ascending distance while retaining traversal order for ties.
        found: list[tuple[np.float32, int]] = []

        def visit(node: _KdNode) -> None:
            if node.indexes is not None:
                for index in node.indexes:
                    delta = query - self.points[index]
                    distance = np.sum(delta * delta, dtype=np.float32)
                    insert = len(found)
                    while insert > 0 and found[insert - 1][0] > distance:
                        insert -= 1
                    if insert < count:
                        found.insert(insert, (distance, int(index)))
                        if len(found) > count:
                            found.pop()
                return
            first, second = _children_near_first(node, query)
            visit(first)
            visit(second)

        visit(self.root)
        return [index for _, index in found]


def _positions(records: tuple[TransportNodeRecord, ...]) -> np.ndarray:
    # Frozen reference coordinates use PDB-style %.3f quantization before graph construction.
    return np.asarray(
        [[float(f"{value:.3f}") for value in record.coordinate] for record in records],
        dtype=np.float32,
    )


def _node_feature(records: tuple[TransportNodeRecord, ...]) -> np.ndarray:
    feature = np.zeros((len(records), GEARNET_EDGE_NODE_INPUT_DIM), dtype=np.int64)
    for index, record in enumerate(records):
        try:
            residue_id = _RESIDUE_TO_ID[record.residue_name.strip().upper()]
        except KeyError as error:
            raise NativeGearNetGraphError(
                f"frozen GearNet graph has no residue ID for {record.residue_name!r}"
            ) from error
        feature[index, residue_id] = 1
    return feature


def _sequential_edges(nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sources: list[int] = []
    destinations: list[int] = []
    relations: list[int] = []
    for delta in range(-GEARNET_EDGE_SEQUENTIAL_MAX_DISTANCE, GEARNET_EDGE_SEQUENTIAL_MAX_DISTANCE + 1):
        for source in range(nodes):
            destination = source + delta
            if 0 <= destination < nodes:
                sources.append(source)
                destinations.append(destination)
                relations.append(delta + GEARNET_EDGE_SEQUENTIAL_MAX_DISTANCE)
    return (
        np.asarray(sources, dtype=np.int64),
        np.asarray(destinations, dtype=np.int64),
        np.asarray(relations, dtype=np.int64),
    )


def _spatial_edges(tree: _ReferenceKdTree, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sources: list[int] = []
    destinations: list[int] = []
    for destination, query in enumerate(positions):
        candidates = [
            source for source in tree.radius(query, GEARNET_EDGE_SPATIAL_RADIUS) if source != destination
        ]
        for source in candidates[:GEARNET_EDGE_SPATIAL_MAX_NEIGHBORS]:
            if abs(source - destination) < GEARNET_EDGE_SPATIAL_MIN_DISTANCE:
                continue
            if np.sqrt(np.sum((positions[source] - query) ** 2, dtype=np.float32)) < np.float32(1e-10):
                continue
            sources.append(source)
            destinations.append(destination)
    return np.asarray(sources, dtype=np.int64), np.asarray(destinations, dtype=np.int64)


def _knn_edges(tree: _ReferenceKdTree, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sources: list[int] = []
    destinations: list[int] = []
    for destination, query in enumerate(positions):
        for source in tree.knn(query, GEARNET_EDGE_KNN_K + 1):
            if abs(source - destination) < GEARNET_EDGE_KNN_MIN_DISTANCE:
                continue
            if np.sqrt(np.sum((positions[source] - query) ** 2, dtype=np.float32)) < np.float32(1e-10):
                continue
            sources.append(source)
            destinations.append(destination)
    return np.asarray(sources, dtype=np.int64), np.asarray(destinations, dtype=np.int64)


def _edge_feature(
    node_feature: np.ndarray,
    positions: np.ndarray,
    source: np.ndarray,
    destination: np.ndarray,
    relation: np.ndarray,
) -> np.ndarray:
    rows = source.shape[0]
    feature = np.zeros((rows, GEARNET_EDGE_EDGE_INPUT_DIM), dtype=np.float32)
    feature[:, :20] = node_feature[source, :20]
    feature[:, 20:40] = node_feature[destination, :20]
    feature[np.arange(rows), 40 + relation] = 1.0
    sequential_distance = np.minimum(np.abs(source - destination), 10)
    feature[np.arange(rows), 47 + sequential_distance] = 1.0
    delta = positions[source] - positions[destination]
    feature[:, 58] = np.sqrt(np.sum(delta * delta, axis=1, dtype=np.float32))
    return feature


def _line_graph(
    positions: np.ndarray, source: np.ndarray, destination: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Frozen reference line-graph ordering and fp32 angle arithmetic use Torch.
    # Torch remains optional and is imported only when native learned construction runs.
    try:
        import torch  # noqa: PLC0415
    except ModuleNotFoundError as error:
        raise NativeGearNetGraphError(
            "native GearNet graph construction requires the optional `torch` dependency"
        ) from error

    edge_source = torch.from_numpy(source)
    edge_destination = torch.from_numpy(destination)
    edge_index = torch.arange(source.shape[0], dtype=torch.long)
    # Stable endpoint ties are part of Eris reproducibility; do not depend on
    # platform-specific torch.argsort tie ordering.
    incoming = edge_index[torch.argsort(edge_destination, stable=True)]
    outgoing = edge_index[torch.argsort(edge_source, stable=True)]
    in_degree = torch.bincount(edge_source, minlength=positions.shape[0])
    out_degree = torch.bincount(edge_destination, minlength=positions.shape[0])
    size = out_degree * in_degree
    starts = (size.cumsum(0) - size).repeat_interleave(size)
    local_index = torch.arange(int(size.sum()), dtype=torch.long) - starts
    inner_size = in_degree.repeat_interleave(size)
    incoming_offset = (out_degree.cumsum(0) - out_degree).repeat_interleave(size)
    outgoing_offset = (in_degree.cumsum(0) - in_degree).repeat_interleave(size)
    line_source = incoming[torch.div(local_index, inner_size, rounding_mode="floor") + incoming_offset]
    line_destination = outgoing[local_index % inner_size + outgoing_offset]

    node_position = torch.from_numpy(positions)
    vector1 = node_position[edge_destination[line_destination]] - node_position[edge_source[line_destination]]
    vector2 = node_position[edge_source[line_source]] - node_position[edge_source[line_destination]]
    angle = torch.atan2(torch.cross(vector1, vector2, dim=-1).norm(dim=-1), (vector1 * vector2).sum(dim=-1))
    bins = (angle / torch.pi * GEARNET_EDGE_NUM_ANGLE_BIN).long().clamp(max=GEARNET_EDGE_NUM_ANGLE_BIN - 1)
    return (
        line_source.numpy().astype(np.int64, copy=False),
        line_destination.numpy().astype(np.int64, copy=False),
        bins.numpy().astype(np.int64, copy=False),
    )


def construct_native_gearnet_graph(records: tuple[TransportNodeRecord, ...]) -> NativeGearNetGraph:
    """Construct frozen GearNet graph arrays for native inference."""
    normalized = _validate_transport_records(records)
    positions = _positions(normalized)
    node_feature = _node_feature(normalized)
    tree = _ReferenceKdTree(positions)
    sequential_source, sequential_destination, sequential_relation = _sequential_edges(len(normalized))
    spatial_source, spatial_destination = _spatial_edges(tree, positions)
    knn_source, knn_destination = _knn_edges(tree, positions)
    source = np.concatenate((sequential_source, spatial_source, knn_source))
    destination = np.concatenate((sequential_destination, spatial_destination, knn_destination))
    relation = np.concatenate(
        (
            sequential_relation,
            np.full(spatial_source.shape[0], 5, dtype=np.int64),
            np.full(knn_source.shape[0], 6, dtype=np.int64),
        )
    )
    if relation.size and int(relation.max()) >= GEARNET_EDGE_NUM_RELATION:
        raise NativeGearNetGraphError("native GearNet graph relation ID is out of range")
    line_source, line_destination, line_relation = _line_graph(positions, source, destination)
    return NativeGearNetGraph(
        node_feature=node_feature,
        node_position=positions,
        edge_source=source,
        edge_destination=destination,
        edge_relation=relation,
        edge_feature=_edge_feature(node_feature, positions, source, destination, relation),
        line_edge_source=line_source,
        line_edge_destination=line_destination,
        line_angle_relation=line_relation,
    )


__all__ = ["NativeGearNetGraph", "NativeGearNetGraphError", "construct_native_gearnet_graph"]
