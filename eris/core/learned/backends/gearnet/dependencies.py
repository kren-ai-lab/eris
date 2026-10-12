"""Frozen optional dependency boundary for the GearNet-Edge backend."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

from eris.exceptions import OptionalDependencyError


@dataclass(frozen=True)
class BackendAvailability:
    """Dependency-level availability reported without instantiating a backend."""

    available: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        """Validate availability fields."""
        if type(self.available) is not bool:
            raise TypeError("available must be a bool")
        if self.reason is not None:
            if not isinstance(self.reason, str):
                raise TypeError("reason must be a string or None")
            if not self.reason.strip():
                raise ValueError("reason must not be empty when provided")
        if not self.available and self.reason is None:
            raise ValueError("reason is required when the backend is unavailable")


def probe_gearnet_availability() -> BackendAvailability:
    """Check that torch is importable without importing it or running GearNet."""
    if importlib.util.find_spec("torch") is None:
        return BackendAvailability(
            available=False,
            reason=(
                "GearNet backend dependencies are not installed: torch. "
                "Install the 'gearnet' extra: pip install 'erislib[gearnet]'."
            ),
        )
    return BackendAvailability(available=True)


def import_native_gearnet_torch() -> ModuleType:
    """Import the PyTorch runtime required for native GearNet execution."""
    try:
        import torch  # noqa: PLC0415
    except (ImportError, OSError) as error:
        raise OptionalDependencyError(
            "Native GearNet-Edge requires optional dependency torch; "
            "install the 'gearnet' extra: pip install 'erislib[gearnet]'"
        ) from error
    return torch


__all__ = [
    "BackendAvailability",
    "import_native_gearnet_torch",
    "probe_gearnet_availability",
]
