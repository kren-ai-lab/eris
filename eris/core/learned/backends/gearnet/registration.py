"""Side-effect-free learned Registry integration for GearNet-Edge."""

from __future__ import annotations

from eris.core.learned.backends.gearnet.config import GearNetEdgeConfig
from eris.core.learned.backends.gearnet.metadata import (
    GEARNET_EDGE_BACKEND_ID,
    GEARNET_EDGE_ENCODER_VERSION,
    GEARNET_EDGE_IMPLEMENTATION,
    GEARNET_EDGE_MODEL_DESCRIPTOR,
    GEARNET_EDGE_REPRESENTATION_NAME,
)
from eris.core.learned.registry import BackendAvailability, EmbedderInfo, EmbedderRegistry


def _backend_factory():
    from eris.core.learned.backends.gearnet.backend import GearNetEdgeBackend

    return GearNetEdgeBackend()


def _availability_probe() -> BackendAvailability:
    from eris.core.learned.backends.gearnet.dependencies import probe_gearnet_availability

    return probe_gearnet_availability()


def register_gearnet_edge(registry: EmbedderRegistry) -> EmbedderInfo:
    """Register exactly one frozen GearNet-Edge adapter in an explicit Registry."""

    if not isinstance(registry, EmbedderRegistry):
        raise TypeError("registry must be an EmbedderRegistry")
    return registry.register(
        representation_name=GEARNET_EDGE_REPRESENTATION_NAME,
        backend_id=GEARNET_EDGE_BACKEND_ID,
        implementation=GEARNET_EDGE_IMPLEMENTATION,
        encoder_version=GEARNET_EDGE_ENCODER_VERSION,
        config_type=GearNetEdgeConfig,
        model_descriptor=GEARNET_EDGE_MODEL_DESCRIPTOR,
        backend_factory=_backend_factory,
        availability_probe=_availability_probe,
    )


__all__ = ["register_gearnet_edge"]
