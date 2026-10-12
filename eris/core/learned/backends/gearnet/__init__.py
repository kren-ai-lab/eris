"""Lightweight contracts and explicit registration for GearNet-Edge."""

from eris.core.learned.backends.gearnet.config import GearNetEdgeConfig
from eris.core.learned.backends.gearnet.dependencies import (
    GEARNET_DEPENDENCY_REQUIREMENTS,
    GearNetDependencyRequirement,
    probe_gearnet_availability,
)
from eris.core.learned.backends.gearnet.metadata import (
    GEARNET_EDGE_BACKEND_ID,
    GEARNET_EDGE_CHECKPOINT_DESCRIPTOR,
    GEARNET_EDGE_ENCODER_VERSION,
    GEARNET_EDGE_MODEL_DESCRIPTOR,
    GEARNET_EDGE_REPRESENTATION_NAME,
)
from eris.core.learned.backends.gearnet.registration import register_gearnet_edge

__all__ = [
    "GEARNET_DEPENDENCY_REQUIREMENTS",
    "GEARNET_EDGE_BACKEND_ID",
    "GEARNET_EDGE_CHECKPOINT_DESCRIPTOR",
    "GEARNET_EDGE_ENCODER_VERSION",
    "GEARNET_EDGE_MODEL_DESCRIPTOR",
    "GEARNET_EDGE_REPRESENTATION_NAME",
    "GearNetDependencyRequirement",
    "GearNetEdgeConfig",
    "probe_gearnet_availability",
    "register_gearnet_edge",
]
