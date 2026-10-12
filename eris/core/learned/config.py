"""Base configuration for structural embedders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from eris.exceptions import ConfigError

OutputMode = Literal["node", "global", "both"]
_OUTPUT_MODES = {"node", "global", "both"}


@dataclass(frozen=True, kw_only=True)
class StructuralEmbedderConfig:
    """Canonical scientific configuration shared by structural embedders."""

    representation_name: str
    output_mode: OutputMode = "both"

    def __post_init__(self) -> None:
        """Validate and normalize the config fields."""
        if not isinstance(self.representation_name, str):
            raise ConfigError("representation_name must be a string")
        representation_name = self.representation_name.strip()
        if not representation_name:
            raise ConfigError("representation_name must not be empty")
        if not isinstance(self.output_mode, str) or self.output_mode not in _OUTPUT_MODES:
            allowed = ", ".join(sorted(_OUTPUT_MODES))
            raise ConfigError(f"output_mode must be one of: {allowed}")

        object.__setattr__(self, "representation_name", representation_name)

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible canonical parameters."""
        return {
            "representation_name": self.representation_name,
            "output_mode": self.output_mode,
        }
