"""Exact construction and safe checkpoint loading for frozen GearNet-Edge."""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from typing import Any

from eris.core.learned.backends.gearnet.metadata import (
    GEARNET_EDGE_ARCHITECTURE,
    GEARNET_EDGE_CHECKPOINT_IDENTIFIER,
    GEARNET_EDGE_CHECKPOINT_SHA256,
    GEARNET_EDGE_EDGE_INPUT_DIM,
    GEARNET_EDGE_HIDDEN_DIMS,
    GEARNET_EDGE_NODE_INPUT_DIM,
    GEARNET_EDGE_NUM_ANGLE_BIN,
    GEARNET_EDGE_NUM_RELATION,
    GEARNET_EDGE_OUTPUT_DIM,
)
from eris.core.learned.checkpoints import VerifiedCheckpoint
from eris.core.learned.execution import ResolvedExecution
from eris.exceptions import ExecutionError


def _import_frameworks() -> tuple[Any, Any]:
    try:
        import torch
        from torchdrug import models
    except (ImportError, OSError) as error:
        raise ExecutionError(f"Unable to import the frozen GearNet runtime: {error}") from error
    return torch, models


def _validate_execution(execution: ResolvedExecution) -> None:
    if not isinstance(execution, ResolvedExecution):
        raise TypeError("execution must be a ResolvedExecution")
    if execution.device != "cpu" or execution.precision != "fp32":
        raise ExecutionError(
            "GearNet-Edge model loading supports only device='cpu' and precision='fp32'"
        )


def _validate_checkpoint_identity(checkpoint: VerifiedCheckpoint) -> None:
    if not isinstance(checkpoint, VerifiedCheckpoint):
        raise TypeError("checkpoint must be a VerifiedCheckpoint")
    if checkpoint.sha256 != checkpoint.descriptor.sha256:
        raise ExecutionError(
            f"Verified checkpoint '{checkpoint.descriptor.identifier}' has inconsistent "
            "SHA-256 metadata"
        )
    if (
        checkpoint.descriptor.identifier != GEARNET_EDGE_CHECKPOINT_IDENTIFIER
        or checkpoint.descriptor.sha256 != GEARNET_EDGE_CHECKPOINT_SHA256
    ):
        raise ExecutionError(
            "GearNet-Edge requires the frozen mc_gearnet_edge.pth checkpoint identity"
        )


def _validate_model_identity(model: Any, models: Any) -> None:
    model_type = type(model)
    expected_type = models.GearNet
    if model_type is not expected_type:
        actual = f"{model_type.__module__}.{model_type.__qualname__}"
        raise ExecutionError(f"Unexpected GearNet model class: {actual}")
    if model.output_dim != GEARNET_EDGE_OUTPUT_DIM:
        raise ExecutionError(
            f"Unexpected GearNet-Edge output_dim: {model.output_dim!r}; "
            f"expected {GEARNET_EDGE_OUTPUT_DIM}"
        )


def _construct_model(models: Any) -> Any:
    model = models.GearNet(
        input_dim=GEARNET_EDGE_NODE_INPUT_DIM,
        hidden_dims=list(GEARNET_EDGE_HIDDEN_DIMS),
        num_relation=GEARNET_EDGE_NUM_RELATION,
        edge_input_dim=GEARNET_EDGE_EDGE_INPUT_DIM,
        num_angle_bin=GEARNET_EDGE_NUM_ANGLE_BIN,
        short_cut=True,
        batch_norm=True,
        concat_hidden=True,
        readout="sum",
    )
    _validate_model_identity(model, models)
    return model


def construct_gearnet_edge_model() -> Any:
    """Construct the exact TorchDrug model required by the frozen checkpoint."""

    _, models = _import_frameworks()
    return _construct_model(models)


def _deserialize_state_dict(checkpoint: VerifiedCheckpoint, torch: Any) -> Mapping[str, Any]:
    try:
        state_dict = torch.load(
            checkpoint.path,
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, EOFError, pickle.UnpicklingError, RuntimeError) as error:
        raise ExecutionError(
            f"Unable to safely load GearNet-Edge checkpoint "
            f"'{checkpoint.descriptor.identifier}' at {checkpoint.path}: {error}"
        ) from error

    if not isinstance(state_dict, Mapping):
        raise ExecutionError("GearNet-Edge checkpoint root must be a state_dict mapping")
    if any(not isinstance(key, str) for key in state_dict):
        raise ExecutionError("GearNet-Edge state_dict keys must all be strings")
    if any(not isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise ExecutionError("GearNet-Edge state_dict values must all be tensors")
    return state_dict


def _load_state_dict_strict(model: Any, state_dict: Mapping[str, Any]) -> None:
    try:
        result = model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ExecutionError(
            "Official GearNet-Edge checkpoint is incompatible with the frozen architecture: "
            f"{error}"
        ) from error
    if result.missing_keys or result.unexpected_keys:
        raise ExecutionError(
            "Strict GearNet-Edge state loading returned unresolved missing or unexpected keys"
        )


def _validate_cpu_fp32(model: Any, torch: Any) -> None:
    tensors = [
        *(("parameter", name, value) for name, value in model.named_parameters()),
        *(("buffer", name, value) for name, value in model.named_buffers()),
    ]
    for kind, name, tensor in tensors:
        if tensor.device.type != "cpu":
            raise ExecutionError(f"GearNet-Edge {kind} '{name}' is not on CPU")
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise ExecutionError(f"GearNet-Edge {kind} '{name}' is not torch.float32")


def _set_and_validate_eval(model: Any) -> None:
    model.eval()
    if model.training or any(module.training for module in model.modules()):
        raise ExecutionError("GearNet-Edge model did not enter eval mode completely")


def load_verified_gearnet_edge_model(
    checkpoint: VerifiedCheckpoint,
    *,
    execution: ResolvedExecution,
) -> Any:
    """Safely load the frozen verified checkpoint into the exact CPU/fp32 model."""

    _validate_execution(execution)
    _validate_checkpoint_identity(checkpoint)
    torch, models = _import_frameworks()
    model = _construct_model(models)
    state_dict = _deserialize_state_dict(checkpoint, torch)
    _load_state_dict_strict(model, state_dict)
    _validate_cpu_fp32(model, torch)
    _set_and_validate_eval(model)
    return model


__all__ = [
    "GEARNET_EDGE_ARCHITECTURE",
    "GEARNET_EDGE_CHECKPOINT_IDENTIFIER",
    "GEARNET_EDGE_CHECKPOINT_SHA256",
    "GEARNET_EDGE_OUTPUT_DIM",
    "construct_gearnet_edge_model",
    "load_verified_gearnet_edge_model",
]
