"""Frozen optional dependency boundary for the GearNet-Edge backend."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata

from eris.core.learned.registry import BackendAvailability

GEARNET_PYTHON_REQUIREMENT = ">=3.10,<3.11"
_IMPORT_SMOKE_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class GearNetDependencyRequirement:
    """One distribution/import/version requirement in the frozen backend stack."""

    distribution: str
    import_name: str
    requirement: str


GEARNET_DEPENDENCY_REQUIREMENTS = (
    GearNetDependencyRequirement("numpy", "numpy", ">=1.26,<2"),
    GearNetDependencyRequirement("torch", "torch", "==2.11.0"),
    GearNetDependencyRequirement("torchdrug", "torchdrug", "==0.2.1"),
    GearNetDependencyRequirement("torch-scatter", "torch_scatter", "==2.1.2"),
    GearNetDependencyRequirement("torch-cluster", "torch_cluster", "==1.6.3"),
    GearNetDependencyRequirement("rdkit-pypi", "rdkit", "==2022.9.5"),
    GearNetDependencyRequirement("ninja", "ninja", ">=1.11,<2"),
)

_IMPORT_SMOKE = """
import torch
import torchdrug
import torch_scatter
import torch_cluster
import rdkit
from torchdrug import models, layers, data
"""


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


def _compare_releases(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    width = max(len(left), len(right))
    normalized_left = left + (0,) * (width - len(left))
    normalized_right = right + (0,) * (width - len(right))
    return (normalized_left > normalized_right) - (normalized_left < normalized_right)


def _matches_requirement(installed: str, requirement: str) -> bool:
    public = _public_version(installed)
    if requirement.startswith("=="):
        return public == requirement[2:]

    release = _numeric_release(public)
    if release is None:
        return False
    if requirement == ">=1.26,<2":
        return _compare_releases(release, (1, 26)) >= 0 and _compare_releases(release, (2,)) < 0
    if requirement == ">=1.11,<2":
        return _compare_releases(release, (1, 11)) >= 0 and _compare_releases(release, (2,)) < 0
    raise AssertionError(f"Unsupported frozen GearNet requirement: {requirement}")


def _last_error_line(stderr: str, stdout: str) -> str:
    lines = [line.strip() for line in (stderr or stdout).splitlines() if line.strip()]
    for line in reversed(lines):
        if line != "Traceback (most recent call last):" and not line.startswith("File "):
            return line[:500]
    return "subprocess exited without an error message"


def _run_import_smoke() -> str | None:
    try:
        result = subprocess.run(
            [sys.executable, "-c", _IMPORT_SMOKE],
            check=False,
            capture_output=True,
            text=True,
            timeout=_IMPORT_SMOKE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return (
            "GearNet dependency import smoke timed out after "
            f"{_IMPORT_SMOKE_TIMEOUT_SECONDS} seconds"
        )
    except OSError as exc:
        return f"GearNet dependency import smoke could not start: {exc}"

    if result.returncode != 0:
        detail = _last_error_line(result.stderr, result.stdout)
        return f"GearNet dependency import smoke failed: {detail}"
    return None


def probe_gearnet_availability() -> BackendAvailability:
    """Check the frozen dependency stack without instantiating or running GearNet."""

    python_version = _current_python_version()
    if not (python_version >= (3, 10) and python_version < (3, 11)):
        actual = ".".join(str(part) for part in python_version)
        return BackendAvailability(
            available=False,
            reason=(
                f"GearNet backend requires Python {GEARNET_PYTHON_REQUIREMENT}; "
                f"running Python {actual}. Eris core remains supported on Python 3.10-3.12."
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
                "Install the documented Eris GearNet CPU environment."
            ),
        )
    if missing_modules:
        missing = ", ".join(missing_modules)
        return BackendAvailability(
            available=False,
            reason=(
                f"GearNet distributions are installed but import modules are missing: {missing}. "
                "Reinstall the documented Eris GearNet CPU environment."
            ),
        )
    if incompatible:
        details = "; ".join(incompatible)
        return BackendAvailability(
            available=False,
            reason=(
                f"GearNet backend dependency versions are incompatible: {details}. "
                "Install the frozen Eris v0.1 GearNet dependency versions."
            ),
        )

    smoke_error = _run_import_smoke()
    if smoke_error is not None:
        return BackendAvailability(available=False, reason=smoke_error)
    return BackendAvailability(available=True)


__all__ = [
    "GEARNET_DEPENDENCY_REQUIREMENTS",
    "GEARNET_PYTHON_REQUIREMENT",
    "GearNetDependencyRequirement",
    "probe_gearnet_availability",
]
