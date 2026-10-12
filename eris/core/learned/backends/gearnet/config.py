"""Canonical scientific configuration for GearNet-Edge artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eris.core.learned.backends.gearnet.metadata import (
    GEARNET_EDGE_REPRESENTATION_NAME,
    gearnet_edge_preprocessing_parameters,
)
from eris.core.learned.config import StructuralEmbedderConfig
from eris.exceptions import ConfigError


@dataclass(frozen=True, kw_only=True)
class GearNetEdgeConfig(StructuralEmbedderConfig):
    """Frozen GearNet-Edge v0.1 scientific parameters plus output selection."""

    def __post_init__(self) -> None:
        """Validate the GearNet-Edge representation name."""
        super().__post_init__()
        if self.representation_name != GEARNET_EDGE_REPRESENTATION_NAME:
            raise ConfigError(
                f"GearNetEdgeConfig representation_name must be {GEARNET_EDGE_REPRESENTATION_NAME!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return canonical parameters used for artifact provenance and cache identity."""
        return {
            "representation_name": self.representation_name,
            "output_mode": self.output_mode,
            "preprocessing": gearnet_edge_preprocessing_parameters(),
        }


__all__ = ["GearNetEdgeConfig"]
