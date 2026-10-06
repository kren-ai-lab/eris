"""Frozen optional dependency boundary for the GearNet-Edge backend."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from importlib import metadata

from eris.core.learned.registry import BackendAvailability

GEARNET_PYTHON_REQUIREMENT = ">=3.11,<3.13"


@dataclass(frozen=True)
class GearNetDependencyRequirement:
    """One distribution/import/version requirement in the frozen backend stack."""

    distribution: str
    import_name: str
    requirement: str


GEARNET_DEPENDENCY_REQUIREMENTS = (GearNetDependencyRequirement("torch", "torch", "==2.11.0"),)


def _current_python_version() -> tuple[int, int]:
    return sys.version_info[:2]


def _distribution_version(distribution: str) -> str:
    return metadata.version(distribution)


def _module_exists(import_name: str) -> bool:
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _public_version(version: str) -> str:
    return version.partition("+")[0]


def _numeric_release(version: str) -> tuple[int, ...] | None:
    parts = version.split(".")
    if not parts or any(not part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def _matches_requirement(installed: str, requirement: str) -> bool:
    public = _public_version(installed)
    if requirement.startswith("=="):
        return public == requirement[2:]

    release = _numeric_release(public)
    if release is None:
        return False
    raise AssertionError(f"Unsupported frozen GearNet requirement: {requirement}")


def probe_gearnet_availability() -> BackendAvailability:
    """Check the frozen dependency stack without instantiating or running GearNet."""

    python_version = _current_python_version()
    if not (python_version >= (3, 11) and python_version < (3, 13)):
        actual = ".".join(str(part) for part in python_version)
        return BackendAvailability(
            available=False,
            reason=(
                f"GearNet backend requires Python {GEARNET_PYTHON_REQUIREMENT}; "
                f"running Python {actual}."
            ),
        )

    missing_distributions: list[str] = []
    missing_modules: list[str] = []
    incompatible: list[str] = []

    for dependency in GEARNET_DEPENDENCY_REQUIREMENTS:
        try:
            installed = _distribution_version(dependency.distribution)
        except metadata.PackageNotFoundError:
            missing_distributions.append(dependency.distribution)
            continue

        if not _module_exists(dependency.import_name):
            missing_modules.append(dependency.import_name)
        if not _matches_requirement(installed, dependency.requirement):
            incompatible.append(
                f"{dependency.distribution} {installed} (expected {dependency.requirement})"
            )

    if missing_distributions:
        missing = ", ".join(missing_distributions)
        return BackendAvailability(
            available=False,
            reason=(
                f"GearNet backend dependencies are not installed: {missing}. "
                "Install torch==2.11.0 for the Eris native GearNet CPU runtime."
            ),
        )
    if missing_modules:
        missing = ", ".join(missing_modules)
        return BackendAvailability(
            available=False,
            reason=(
                f"GearNet distributions are installed but import modules are missing: {missing}. "
                "Reinstall torch==2.11.0 for the Eris native GearNet CPU runtime."
            ),
        )
    if incompatible:
        details = "; ".join(incompatible)
        return BackendAvailability(
            available=False,
            reason=(
                f"GearNet backend dependency versions are incompatible: {details}. "
                "Install torch==2.11.0 for the frozen Eris native GearNet numerical runtime."
            ),
        )

    return BackendAvailability(available=True)


def import_native_gearnet_torch():
    """Import exactly the frozen PyTorch runtime required for native GearNet execution."""

    try:
        installed = _distribution_version("torch")
    except metadata.PackageNotFoundError as error:
        raise ImportError(
            "Native GearNet-Edge requires optional dependency torch==2.11.0; "
            "install the Eris GearNet CPU runtime"
        ) from error
    if not _matches_requirement(installed, "==2.11.0"):
        raise ImportError(
            "Native GearNet-Edge requires torch==2.11.0 for frozen numerical parity; "
            f"found torch {installed}"
        )
    try:
        import torch
    except (ImportError, OSError) as error:
        raise ImportError(
            "Native GearNet-Edge could not import the required torch==2.11.0 CPU runtime"
        ) from error
    return torch


__all__ = [
    "GEARNET_DEPENDENCY_REQUIREMENTS",
    "GEARNET_PYTHON_REQUIREMENT",
    "GearNetDependencyRequirement",
    "import_native_gearnet_torch",
    "probe_gearnet_availability",
]
