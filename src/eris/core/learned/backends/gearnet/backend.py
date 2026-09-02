"""Internal deterministic CPU/fp32 GearNet-Edge backend."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from eris.core.learned.backends.gearnet.dependencies import (
    import_native_gearnet_torch,
    probe_gearnet_availability,
)
from eris.core.learned.backends.gearnet.metadata import (
    GEARNET_EDGE_BACKEND_ID,
    GEARNET_EDGE_EDGE_INPUT_DIM,
    GEARNET_EDGE_NODE_INPUT_DIM,
    GEARNET_EDGE_NUM_RELATION,
)
from eris.core.learned.backends.gearnet.model import (
    GEARNET_EDGE_OUTPUT_DIM,
    _validate_cpu_fp32,
    _validate_execution,
)
from eris.core.learned.backends.gearnet.native_graph import NativeGearNetGraph
from eris.core.learned.backends.gearnet.preprocessing import (
    GearNetPreparedInput,
    construct_native_gearnet_input,
    load_selected_biological_structure,
    select_gearnet_transport_records,
)
from eris.core.learned.checkpoints import VerifiedCheckpoint
from eris.core.learned.config import StructuralEmbedderConfig
from eris.core.learned.execution import ExecutionCapabilities, ResolvedExecution
from eris.core.learned.types import (
    StructuralEmbeddingResult,
    StructuralPreprocessingResult,
    validate_embedding_result,
)
from eris.core.types import StructureInput
from eris.exceptions import ExecutionError


def _import_torch() -> Any:
    try:
        return import_native_gearnet_torch()
    except ImportError as error:
        raise ExecutionError(str(error)) from error


def _validate_prepared(prepared: GearNetPreparedInput) -> int:
    if not isinstance(prepared, GearNetPreparedInput):
        raise TypeError("prepared must be a GearNetPreparedInput")
    graph = prepared.graph
    if not isinstance(graph, NativeGearNetGraph):
        raise ExecutionError("GearNet prepared graph must be a NativeGearNetGraph")

    num_node = graph.node_feature.shape[0]
    if num_node <= 0:
        raise ExecutionError("GearNet prepared graph must contain at least one node")

    node_feature = getattr(graph, "node_feature", None)
    if not isinstance(node_feature, np.ndarray) or node_feature.ndim != 2:
        raise ExecutionError("GearNet prepared node_feature must be two-dimensional")
    if tuple(node_feature.shape) != (num_node, GEARNET_EDGE_NODE_INPUT_DIM):
        raise ExecutionError(
            "GearNet prepared node_feature must have shape "
            f"({num_node}, {GEARNET_EDGE_NODE_INPUT_DIM})"
        )

    edge_feature = getattr(graph, "edge_feature", None)
    if not isinstance(edge_feature, np.ndarray) or edge_feature.ndim != 2:
        raise ExecutionError("GearNet prepared edge_feature must be two-dimensional")
    if edge_feature.shape[1] != GEARNET_EDGE_EDGE_INPUT_DIM:
        raise ExecutionError(
            f"GearNet prepared edge_feature width must be {GEARNET_EDGE_EDGE_INPUT_DIM}"
        )

    if graph.edge_relation.size and (
        graph.edge_relation.min() < 0 or graph.edge_relation.max() >= GEARNET_EDGE_NUM_RELATION
    ):
        raise ExecutionError(
            f"GearNet prepared graph must declare {GEARNET_EDGE_NUM_RELATION} relations"
        )
    if (
        graph.edge_source.shape != graph.edge_destination.shape
        or graph.edge_source.shape != graph.edge_relation.shape
    ):
        raise ExecutionError("GearNet prepared edge arrays must align")

    carrier = prepared.carrier
    if not isinstance(carrier, np.ndarray) or carrier.ndim != 1:
        raise ExecutionError("GearNet carrier must be a one-dimensional NumPy array")
    if carrier.dtype.kind not in "iu":
        raise ExecutionError("GearNet carrier must have an integer dtype")
    if carrier.shape[0] != num_node:
        raise ExecutionError("GearNet carrier must align with prepared graph nodes")

    node_ids = prepared.node_ids
    if not isinstance(node_ids, np.ndarray) or node_ids.ndim != 1:
        raise ExecutionError("GearNet node_ids must be a one-dimensional NumPy array")
    if node_ids.dtype.kind != "U":
        raise ExecutionError("GearNet node_ids must have a NumPy Unicode dtype")
    if node_ids.shape[0] != num_node:
        raise ExecutionError("GearNet node_ids must align with prepared graph nodes")
    if len(node_ids) != len(set(node_ids.tolist())):
        raise ExecutionError("GearNet node_ids must be unique")

    expected_node_ids = prepared.biological_table.node_ids_for(carrier.tolist())
    if node_ids.tolist() != list(expected_node_ids):
        raise ExecutionError(
            "GearNet node_ids must match the biological table through the authoritative carrier"
        )
    return num_node


def _validate_native_graph(prepared: GearNetPreparedInput) -> None:
    graph = prepared.graph
    node_feature = graph.node_feature
    edge_feature = graph.edge_feature
    if node_feature.dtype != np.int64:
        raise ExecutionError("GearNet prepared node_feature must be int64")
    if edge_feature.dtype != np.float32:
        raise ExecutionError("GearNet prepared edge_feature must be float32")
    if not np.isfinite(node_feature).all():
        raise ExecutionError("GearNet prepared node_feature must contain only finite values")
    if not np.isfinite(edge_feature).all():
        raise ExecutionError("GearNet prepared edge_feature must contain only finite values")


def _validate_output_tensor(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    torch: Any,
) -> Any:
    if not isinstance(value, torch.Tensor):
        raise ExecutionError(f"GearNet output {name!r} must be a torch.Tensor")
    if tuple(value.shape) != shape:
        raise ExecutionError(f"GearNet output {name!r} must have shape {shape}")
    if value.dtype != torch.float32:
        raise ExecutionError(f"GearNet output {name!r} must be torch.float32")
    if value.device.type != "cpu":
        raise ExecutionError(f"GearNet output {name!r} must be on CPU")
    if not bool(torch.isfinite(value).all().item()):
        raise ExecutionError(f"GearNet output {name!r} must contain only finite values")
    return value if value.is_contiguous() else value.contiguous()


def _validate_numpy_output(
    value: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...],
) -> None:
    if value.shape != shape:
        raise ExecutionError(f"GearNet NumPy {name} must have shape {shape}")
    if value.dtype != np.float32:
        raise ExecutionError(f"GearNet NumPy {name} must have dtype float32")
    if not value.flags.c_contiguous:
        raise ExecutionError(f"GearNet NumPy {name} must be C-contiguous")
    if not np.isfinite(value).all():
        raise ExecutionError(f"GearNet NumPy {name} must contain only finite values")


def _forward_error(error: Exception) -> ExecutionError:
    return ExecutionError(f"GearNet-Edge native CPU/fp32 inference failed: {error}")


class GearNetEdgeBackend:
    """Production GearNet-Edge adapter for the frozen v0.1 stack."""

    @property
    def backend_id(self) -> str:
        return GEARNET_EDGE_BACKEND_ID

    def is_available(self) -> bool:
        return probe_gearnet_availability().available

    def execution_capabilities(self) -> ExecutionCapabilities:
        return ExecutionCapabilities(
            devices=("cpu",),
            precisions=("fp32",),
            preferred_device="cpu",
        )

    def preprocess(
        self,
        structure: StructureInput,
        *,
        config: StructuralEmbedderConfig,
    ) -> StructuralPreprocessingResult[GearNetPreparedInput]:
        if not isinstance(structure, StructureInput):
            raise TypeError("structure must be a StructureInput")
        if not isinstance(config, StructuralEmbedderConfig):
            raise TypeError("config must be a StructuralEmbedderConfig")
        selected = load_selected_biological_structure(structure)
        records = select_gearnet_transport_records(selected)
        prepared = construct_native_gearnet_input(records, selected.biological_table)
        return StructuralPreprocessingResult(
            prepared=prepared,
            input_format=selected.input_format,
        )

    def load(
        self,
        checkpoint: VerifiedCheckpoint,
        *,
        execution: ResolvedExecution,
    ) -> Any:
        _validate_execution(execution)
        _import_torch()
        from eris.core.learned.backends.gearnet.native_model import load_native_gearnet_edge_model

        return load_native_gearnet_edge_model(checkpoint)

    def infer(
        self,
        model: Any,
        prepared: GearNetPreparedInput,
        *,
        config: StructuralEmbedderConfig,
        execution: ResolvedExecution,
    ) -> StructuralEmbeddingResult:
        if not isinstance(config, StructuralEmbedderConfig):
            raise TypeError("config must be a StructuralEmbedderConfig")
        _validate_execution(execution)
        if getattr(model, "training", None) is not False:
            raise ExecutionError("GearNet-Edge model must be in eval mode before inference")
        num_node = _validate_prepared(prepared)

        _validate_native_graph(prepared)
        torch = _import_torch()
        _validate_cpu_fp32(model, torch)

        try:
            with torch.inference_mode():
                output = model(prepared.graph)
        except (ImportError, OSError, RuntimeError) as error:
            raise _forward_error(error) from error

        if not isinstance(output, Mapping):
            raise ExecutionError("GearNet-Edge forward output must be a mapping")
        missing = {"node_feature", "graph_feature"}.difference(output)
        if missing:
            missing_names = ", ".join(sorted(missing))
            raise ExecutionError(
                f"GearNet-Edge forward output is missing required keys: {missing_names}"
            )

        node_tensor = _validate_output_tensor(
            output["node_feature"],
            name="node_feature",
            shape=(num_node, GEARNET_EDGE_OUTPUT_DIM),
            torch=torch,
        )
        graph_tensor = _validate_output_tensor(
            output["graph_feature"],
            name="graph_feature",
            shape=(1, GEARNET_EDGE_OUTPUT_DIM),
            torch=torch,
        )
        node_numpy = node_tensor.cpu().numpy()
        global_numpy = graph_tensor[0].cpu().numpy()
        _validate_numpy_output(
            node_numpy,
            name="node_embedding",
            shape=(num_node, GEARNET_EDGE_OUTPUT_DIM),
        )
        _validate_numpy_output(
            global_numpy,
            name="global_embedding",
            shape=(GEARNET_EDGE_OUTPUT_DIM,),
        )

        if config.output_mode == "node":
            result = StructuralEmbeddingResult(
                node_embedding=node_numpy,
                node_ids=prepared.node_ids,
                global_embedding=None,
                valid_mask=None,
            )
        elif config.output_mode == "global":
            result = StructuralEmbeddingResult(
                node_embedding=None,
                node_ids=None,
                global_embedding=global_numpy,
                valid_mask=None,
            )
        else:
            result = StructuralEmbeddingResult(
                node_embedding=node_numpy,
                node_ids=prepared.node_ids,
                global_embedding=global_numpy,
                valid_mask=None,
            )
        validate_embedding_result(result, output_mode=config.output_mode)
        return result


__all__ = ["GEARNET_EDGE_BACKEND_ID", "GearNetEdgeBackend"]
