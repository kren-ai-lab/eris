from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from eris.exceptions import ConfigError, ExecutionError

Precision = Literal["fp32", "fp16", "bf16"]

_PRECISIONS = {"fp32", "fp16", "bf16"}
_DEVICE_REQUEST = re.compile(r"^(?:auto|cpu|cuda|cuda:(?:0|[1-9][0-9]*))$")
_CONCRETE_DEVICE = re.compile(r"^(?:cpu|cuda:(?:0|[1-9][0-9]*))$")


def _validate_device_request(device: object) -> None:
    if not isinstance(device, str) or _DEVICE_REQUEST.fullmatch(device) is None:
        raise ConfigError("device must be one of: auto, cpu, cuda, cuda:<nonnegative index>")


def _validate_concrete_device(device: object, field: str) -> None:
    if not isinstance(device, str) or _CONCRETE_DEVICE.fullmatch(device) is None:
        raise ConfigError(f"{field} must be a concrete device: cpu or cuda:<nonnegative index>")


def _validate_precision(precision: object, field: str = "precision") -> None:
    if not isinstance(precision, str) or precision not in _PRECISIONS:
        raise ConfigError(f"{field} must be one of: bf16, fp16, fp32")


@dataclass(frozen=True, kw_only=True)
class StructuralExecutionConfig:
    """Requested operational controls, separate from scientific representation config."""

    device: str = "auto"
    precision: Precision = "fp32"

    def __post_init__(self) -> None:
        _validate_device_request(self.device)
        _validate_precision(self.precision)


@dataclass(frozen=True, kw_only=True)
class ExecutionCapabilities:
    """Concrete devices and precisions advertised by one available backend."""

    devices: tuple[str, ...]
    precisions: tuple[Precision, ...]
    preferred_device: str

    def __post_init__(self) -> None:
        if not isinstance(self.devices, tuple) or not self.devices:
            raise ConfigError("devices must be a non-empty tuple")
        for device in self.devices:
            _validate_concrete_device(device, "devices entries")
        if len(self.devices) != len(set(self.devices)):
            raise ConfigError("devices must not contain duplicates")

        if not isinstance(self.precisions, tuple) or not self.precisions:
            raise ConfigError("precisions must be a non-empty tuple")
        for precision in self.precisions:
            _validate_precision(precision, "precisions entries")
        if len(self.precisions) != len(set(self.precisions)):
            raise ConfigError("precisions must not contain duplicates")

        _validate_concrete_device(self.preferred_device, "preferred_device")
        if self.preferred_device not in self.devices:
            raise ConfigError("preferred_device must appear in devices")


@dataclass(frozen=True, kw_only=True)
class ResolvedExecution:
    """Concrete operational controls selected for one backend execution."""

    device: str
    precision: Precision

    def __post_init__(self) -> None:
        _validate_concrete_device(self.device, "device")
        _validate_precision(self.precision)


def resolve_execution(
    request: StructuralExecutionConfig,
    capabilities: ExecutionCapabilities,
) -> ResolvedExecution:
    """Resolve a validated request against declared capabilities without probing hardware."""

    if not isinstance(request, StructuralExecutionConfig):
        raise TypeError("request must be a StructuralExecutionConfig")
    if not isinstance(capabilities, ExecutionCapabilities):
        raise TypeError("capabilities must be ExecutionCapabilities")

    if request.device == "auto":
        device = capabilities.preferred_device
    elif request.device == "cuda":
        cuda_devices = [
            candidate for candidate in capabilities.devices if candidate.startswith("cuda:")
        ]
        if not cuda_devices:
            available = ", ".join(capabilities.devices)
            raise ExecutionError(
                f"Requested device 'cuda' is unavailable; available devices: {available}"
            )
        device = min(cuda_devices, key=lambda candidate: int(candidate.partition(":")[2]))
    elif request.device in capabilities.devices:
        device = request.device
    else:
        available = ", ".join(capabilities.devices)
        raise ExecutionError(
            f"Requested device {request.device!r} is unavailable; available devices: {available}"
        )

    if request.precision not in capabilities.precisions:
        supported = ", ".join(capabilities.precisions)
        raise ExecutionError(
            f"Requested precision {request.precision!r} is unsupported; "
            f"supported precisions: {supported}"
        )

    return ResolvedExecution(device=device, precision=request.precision)


def build_execution_provenance(
    request: StructuralExecutionConfig,
    resolved: ResolvedExecution,
) -> dict[str, object]:
    """Build deterministic JSON-safe requested/resolved execution provenance."""

    if not isinstance(request, StructuralExecutionConfig):
        raise TypeError("request must be a StructuralExecutionConfig")
    if not isinstance(resolved, ResolvedExecution):
        raise TypeError("resolved must be a ResolvedExecution")
    return {
        "requested": {
            "device": request.device,
            "precision": request.precision,
        },
        "resolved": {
            "device": resolved.device,
            "precision": resolved.precision,
        },
    }


def execution_cache_identity(resolved: ResolvedExecution) -> dict[str, str]:
    """Project resolved execution onto precision-sensitive artifact cache identity."""

    if not isinstance(resolved, ResolvedExecution):
        raise TypeError("resolved must be a ResolvedExecution")
    return {"precision": resolved.precision}


__all__ = [
    "ExecutionCapabilities",
    "Precision",
    "ResolvedExecution",
    "StructuralExecutionConfig",
    "build_execution_provenance",
    "execution_cache_identity",
    "resolve_execution",
]
