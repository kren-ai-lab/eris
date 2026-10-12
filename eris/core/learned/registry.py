"""The single learned embedder (GearNet-Edge), described without importing torch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from eris.core.learned.backends.gearnet.config import GearNetEdgeConfig
from eris.core.learned.backends.gearnet.metadata import (
    GEARNET_EDGE_BACKEND_ID,
    GEARNET_EDGE_ENCODER_VERSION,
    GEARNET_EDGE_IMPLEMENTATION,
    GEARNET_EDGE_MODEL_DESCRIPTOR,
    GEARNET_EDGE_REPRESENTATION_NAME,
)

if TYPE_CHECKING:
    from eris.core.learned.checkpoints import StructuralModelDescriptor


@dataclass(frozen=True)
class EmbedderInfo:
    """Identity of one learned representation.

    ``encoder_version`` versions the Eris adapter contract, independently of model weights.
    """

    representation_name: str
    backend_id: str
    implementation: str
    encoder_version: str
    config_type: type[GearNetEdgeConfig]
    model_descriptor: StructuralModelDescriptor


GEARNET_EDGE_INFO = EmbedderInfo(
    representation_name=GEARNET_EDGE_REPRESENTATION_NAME,
    backend_id=GEARNET_EDGE_BACKEND_ID,
    implementation=GEARNET_EDGE_IMPLEMENTATION,
    encoder_version=GEARNET_EDGE_ENCODER_VERSION,
    config_type=GearNetEdgeConfig,
    model_descriptor=GEARNET_EDGE_MODEL_DESCRIPTOR,
)


class _LearnedEmbedders:
    """Read-only name lookup over the single embedder."""

    def names(self) -> list[str]:
        return [GEARNET_EDGE_INFO.representation_name]

    def list_info(self) -> list[EmbedderInfo]:
        return [GEARNET_EDGE_INFO]

    def get_info(self, name: str) -> EmbedderInfo:
        if name != GEARNET_EDGE_INFO.representation_name:
            raise KeyError(f"Unknown embedder: {name}")
        return GEARNET_EDGE_INFO


learned_registry = _LearnedEmbedders()

__all__ = ["GEARNET_EDGE_INFO", "EmbedderInfo", "learned_registry"]
