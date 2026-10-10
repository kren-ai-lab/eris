"""Frozen optional dependency boundary for the GearNet-Edge backend."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from importlib import metadata

from eris.core.learned.registry import BackendAvailability


@dataclass(frozen=True)
class GearNetDependencyRequirement:
    """One distribution/import requirement in the backend stack."""

    distribution: str
    import_name: str


GEARNET_DEPENDENCY_REQUIREMENTS = (GearNetDependencyRequirement("torch", "torch"),)


def _distribution_version(distribution: str) -> str:
    return metadata.version(distribution)


def _module_exists(import_name: str) -> bool:
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def probe_gearnet_availability() -> BackendAvailability:
    """Check the dependency stack without instantiating or running GearNet."""

    missing_distributions: list[str] = []
    missing_modules: list[str] = []

    for dependency in GEARNET_DEPENDENCY_REQUIREMENTS:
        try:
            _distribution_version(dependency.distribution)
        except metadata.PackageNotFoundError:
            missing_distributions.append(dependency.distribution)
            continue

        if not _module_exists(dependency.import_name):
            missing_modules.append(dependency.import_name)

    if missing_distributions:
        missing = ", ".join(missing_distributions)
        return BackendAvailability(
            available=False,
            reason=(
                f"GearNet backend dependencies are not installed: {missing}. "
                "Install the 'gearnet' extra: pip install 'erislib[gearnet]'."
            ),
        )
    if missing_modules:
        missing = ", ".join(missing_modules)
        return BackendAvailability(
            available=False,
            reason=(
                f"GearNet distributions are installed but import modules are missing: {missing}. "
                "Reinstall the 'gearnet' extra: pip install 'erislib[gearnet]'."
            ),
        )
    return BackendAvailability(available=True)


def import_native_gearnet_torch():
    """Import the PyTorch runtime required for native GearNet execution."""

    try:
        import torch
    except (ImportError, OSError) as error:
        raise ImportError(
            "Native GearNet-Edge requires optional dependency torch; "
            "install the 'gearnet' extra: pip install 'erislib[gearnet]'"
        ) from error
    return torch


__all__ = [
    "GEARNET_DEPENDENCY_REQUIREMENTS",
    "GearNetDependencyRequirement",
    "import_native_gearnet_torch",
    "probe_gearnet_availability",
]
