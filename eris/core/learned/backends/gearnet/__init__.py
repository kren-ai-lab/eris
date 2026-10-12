"""Lightweight contracts for GearNet-Edge."""

from eris.core.learned.backends.gearnet.config import GearNetEdgeConfig
from eris.core.learned.backends.gearnet.dependencies import probe_gearnet_availability
from eris.core.learned.backends.gearnet.metadata import (
    GEARNET_EDGE_BACKEND_ID,
    GEARNET_EDGE_CHECKPOINT_DESCRIPTOR,
    GEARNET_EDGE_ENCODER_VERSION,
    GEARNET_EDGE_MODEL_DESCRIPTOR,
    GEARNET_EDGE_REPRESENTATION_NAME,
)

__all__ = [
    "GEARNET_EDGE_BACKEND_ID",
    "GEARNET_EDGE_CHECKPOINT_DESCRIPTOR",
    "GEARNET_EDGE_ENCODER_VERSION",
    "GEARNET_EDGE_MODEL_DESCRIPTOR",
    "GEARNET_EDGE_REPRESENTATION_NAME",
    "GearNetEdgeConfig",
    "probe_gearnet_availability",
]
