"""Internal deterministic CPU/fp32 GearNet-Edge backend."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from eris.core.learned.backends.gearnet.dependencies import import_native_gearnet_torch
from eris.core.learned.backends.gearnet.loading import (
    _validate_cpu_fp32,
    _validate_execution,
)
from eris.core.learned.backends.gearnet.metadata import (
    GEARNET_EDGE_BACKEND_ID,
)
from eris.core.learned.backends.gearnet.preprocessing import (
    GearNetPreparedInput,
    construct_native_gearnet_input,
    load_selected_biological_structure,
    select_gearnet_transport_records,
)
from eris.core.learned.checkpoints import VerifiedCheckpoint
from eris.core.learned.config import StructuralEmbedderConfig
from eris.core.learned.execution import ResolvedExecution
from eris.core.learned.types import (
    StructuralEmbeddingResult,
    StructuralPreprocessingResult,
)
from eris.core.types import StructureInput
from eris.exceptions import ExecutionError


def _import_torch() -> Any:
    try:
        return import_native_gearnet_torch()
    except ImportError as error:
        raise ExecutionError(str(error)) from error


def _forward_error(error: Exception) -> ExecutionError:
    return ExecutionError(f"GearNet-Edge native CPU/fp32 inference failed: {error}")


class GearNetEdgeBackend:
    """Production GearNet-Edge adapter for the frozen v0.1 stack."""

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

        node_numpy = output["node_feature"].cpu().numpy()
        global_numpy = output["graph_feature"][0].cpu().numpy()

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
        return result


__all__ = ["GEARNET_EDGE_BACKEND_ID", "GearNetEdgeBackend"]
