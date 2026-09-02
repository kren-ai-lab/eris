"""Native PyTorch implementation of the frozen GearNet-Edge numerical model.

This internal numerical implementation is the frozen production runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from eris.core.learned.backends.gearnet.metadata import (
    GEARNET_EDGE_EDGE_INPUT_DIM,
    GEARNET_EDGE_HIDDEN_DIMS,
    GEARNET_EDGE_NODE_INPUT_DIM,
    GEARNET_EDGE_NUM_ANGLE_BIN,
    GEARNET_EDGE_NUM_RELATION,
    GEARNET_EDGE_OUTPUT_DIM,
)
from eris.core.learned.backends.gearnet.model import (
    _deserialize_state_dict,
    _load_state_dict_strict,
    _validate_checkpoint_identity,
)
from eris.core.learned.backends.gearnet.native_graph import NativeGearNetGraph
from eris.core.learned.checkpoints import VerifiedCheckpoint
from eris.exceptions import ExecutionError


class NativeGearNetModelError(ExecutionError):
    """Raised when native GearNet-Edge cannot preserve the frozen model contract."""


def _scatter_add(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    result = values.new_zeros((size, values.shape[1]))
    return result.index_add_(0, index, values)


class _GeometricRelationalGraphConv(nn.Module):
    """Parameter-compatible native form of TorchDrug's geometric relation layer."""

    def __init__(self, input_dim: int, output_dim: int, num_relation: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.self_loop = nn.Linear(input_dim, output_dim)
        self.linear = nn.Linear(num_relation * input_dim, output_dim)

    def forward(
        self,
        source: torch.Tensor,
        destination: torch.Tensor,
        relation: torch.Tensor,
        input: torch.Tensor,
    ) -> torch.Tensor:
        if input.ndim != 2 or input.shape[1] != self.input_dim:
            raise NativeGearNetModelError("native GearNet layer input has an invalid shape")
        target = destination * self.num_relation + relation
        update = _scatter_add(input[source], target, input.shape[0] * self.num_relation)
        update = update.reshape(input.shape[0], self.num_relation * self.input_dim)
        return functional.relu(self.batch_norm(self.linear(update) + self.self_loop(input)))


class NativeGearNetEdgeModel(nn.Module):
    """Six-layer CPU/fp32 GearNet-Edge with direct legacy-checkpoint compatibility."""

    def __init__(self) -> None:
        super().__init__()
        self.input_dim = GEARNET_EDGE_NODE_INPUT_DIM
        self.output_dim = GEARNET_EDGE_OUTPUT_DIM
        self.dims = [self.input_dim, *GEARNET_EDGE_HIDDEN_DIMS]
        self.edge_dims = [GEARNET_EDGE_EDGE_INPUT_DIM, *self.dims[:-1]]
        self.num_relation = GEARNET_EDGE_NUM_RELATION
        self.num_angle_bin = GEARNET_EDGE_NUM_ANGLE_BIN
        self.short_cut = True
        self.concat_hidden = True
        self.batch_norm = True
        self.layers = nn.ModuleList(
            _GeometricRelationalGraphConv(input_dim, output_dim, self.num_relation)
            for input_dim, output_dim in zip(self.dims, self.dims[1:])
        )
        self.edge_layers = nn.ModuleList(
            _GeometricRelationalGraphConv(input_dim, output_dim, self.num_angle_bin)
            for input_dim, output_dim in zip(self.edge_dims, self.edge_dims[1:])
        )
        self.batch_norms = nn.ModuleList(nn.BatchNorm1d(dim) for dim in self.dims[1:])

    @staticmethod
    def _tensor(array: np.ndarray, *, dtype: torch.dtype) -> torch.Tensor:
        tensor = torch.from_numpy(array)
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        return tensor

    def _validate_graph(self, graph: NativeGearNetGraph) -> None:
        if not isinstance(graph, NativeGearNetGraph):
            raise TypeError("graph must be a NativeGearNetGraph")
        node_count = graph.node_feature.shape[0]
        if graph.node_feature.shape != (node_count, GEARNET_EDGE_NODE_INPUT_DIM):
            raise NativeGearNetModelError("native GearNet node_feature width must be 21")
        if graph.edge_feature.shape != (graph.edge_source.shape[0], GEARNET_EDGE_EDGE_INPUT_DIM):
            raise NativeGearNetModelError("native GearNet edge_feature width must be 59")
        if graph.edge_relation.shape != graph.edge_source.shape:
            raise NativeGearNetModelError("native GearNet relation array must align with edges")
        if graph.line_edge_source.shape != graph.line_edge_destination.shape:
            raise NativeGearNetModelError("native GearNet line edges must align")
        if graph.line_angle_relation.shape != graph.line_edge_source.shape:
            raise NativeGearNetModelError("native GearNet line relations must align")

    def forward(self, graph: NativeGearNetGraph) -> dict[str, torch.Tensor]:
        """Return canonical node and single-graph sum embeddings for one native graph."""

        self._validate_graph(graph)
        source = self._tensor(graph.edge_source, dtype=torch.int64)
        destination = self._tensor(graph.edge_destination, dtype=torch.int64)
        relation = self._tensor(graph.edge_relation, dtype=torch.int64)
        line_source = self._tensor(graph.line_edge_source, dtype=torch.int64)
        line_destination = self._tensor(graph.line_edge_destination, dtype=torch.int64)
        line_relation = self._tensor(graph.line_angle_relation, dtype=torch.int64)
        layer_input = self._tensor(graph.node_feature, dtype=torch.float32)
        edge_input = self._tensor(graph.edge_feature, dtype=torch.float32)

        hiddens: list[torch.Tensor] = []
        for index, layer in enumerate(self.layers):
            hidden = layer(source, destination, relation, layer_input)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input

            edge_hidden = self.edge_layers[index](
                line_source,
                line_destination,
                line_relation,
                edge_input,
            )
            target = destination * self.num_relation + relation
            update = _scatter_add(
                edge_hidden,
                target,
                layer_input.shape[0] * self.num_relation,
            ).reshape(layer_input.shape[0], self.num_relation * edge_hidden.shape[1])
            hidden = hidden + functional.relu(layer.linear(update))
            hidden = self.batch_norms[index](hidden)
            hiddens.append(hidden)
            layer_input = hidden
            edge_input = edge_hidden

        node_feature = torch.cat(hiddens, dim=-1)
        return {
            "node_feature": node_feature,
            "graph_feature": node_feature.sum(dim=0, keepdim=True),
        }


def construct_native_gearnet_edge_model() -> NativeGearNetEdgeModel:
    """Construct the frozen native GearNet-Edge architecture on CPU/fp32."""

    return NativeGearNetEdgeModel()


def load_native_gearnet_edge_model(checkpoint: VerifiedCheckpoint) -> NativeGearNetEdgeModel:
    """Load the original verified GearNet checkpoint through strict key and shape matching."""

    _validate_checkpoint_identity(checkpoint)
    model = construct_native_gearnet_edge_model()
    state_dict = _deserialize_state_dict(checkpoint, torch)
    if not isinstance(state_dict, Mapping):
        raise NativeGearNetModelError("native GearNet checkpoint must be a state_dict mapping")
    _load_state_dict_strict(model, state_dict)
    model.eval()
    return model


__all__ = [
    "NativeGearNetEdgeModel",
    "NativeGearNetModelError",
    "construct_native_gearnet_edge_model",
    "load_native_gearnet_edge_model",
]
