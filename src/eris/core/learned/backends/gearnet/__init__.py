"""Lightweight dependency contract for the future GearNet-Edge backend."""

from eris.core.learned.backends.gearnet.dependencies import (
    GEARNET_DEPENDENCY_REQUIREMENTS,
    GEARNET_PYTHON_REQUIREMENT,
    GearNetDependencyRequirement,
    probe_gearnet_availability,
)

__all__ = [
    "GEARNET_DEPENDENCY_REQUIREMENTS",
    "GEARNET_PYTHON_REQUIREMENT",
    "GearNetDependencyRequirement",
    "probe_gearnet_availability",
]
