from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from eris.exceptions import ConfigError

Precision = Literal["fp32"]


@dataclass(frozen=True, kw_only=True)
class StructuralExecutionConfig:
    """Requested operational controls, separate from scientific representation config."""

    device: str = "auto"
    precision: Precision = "fp32"

    def __post_init__(self) -> None:
        if self.device not in ("auto", "cpu"):
            raise ConfigError(
                f"device must be one of: auto, cpu (CPU-only backend); got {self.device!r}"
            )
        if self.precision != "fp32":
            raise ConfigError(f"precision must be fp32 (fp32-only backend); got {self.precision!r}")


@dataclass(frozen=True, kw_only=True)
class ResolvedExecution:
    """Concrete operational controls selected for one backend execution."""

    device: str
    precision: Precision

    def __post_init__(self) -> None:
        if self.device != "cpu":
            raise ConfigError(f"resolved device must be cpu; got {self.device!r}")
        if self.precision != "fp32":
            raise ConfigError(f"resolved precision must be fp32; got {self.precision!r}")


def resolve_execution(request: StructuralExecutionConfig) -> ResolvedExecution:
    """Resolve a validated request; ``auto`` means CPU."""

    if not isinstance(request, StructuralExecutionConfig):
        raise TypeError("request must be a StructuralExecutionConfig")
    return ResolvedExecution(device="cpu", precision=request.precision)


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


def execution_request_cache_identity(
    request: StructuralExecutionConfig,
) -> dict[str, str]:
    """Project a request onto precision-sensitive artifact cache identity."""

    if not isinstance(request, StructuralExecutionConfig):
        raise TypeError("request must be a StructuralExecutionConfig")
    return {"precision": request.precision}


__all__ = [
    "Precision",
    "ResolvedExecution",
    "StructuralExecutionConfig",
    "build_execution_provenance",
    "execution_request_cache_identity",
    "resolve_execution",
]
