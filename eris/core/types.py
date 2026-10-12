"""Public input types."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class StructureInput:
    """Structure source with chain and model selection."""

    source: str
    chain_id: str | Sequence[str] | None = None
    model_index: int = 0

    def __post_init__(self) -> None:
        """Validate the input and normalize the chain selection to a tuple."""
        source = os.fspath(self.source)
        if not isinstance(source, str) or not source:
            raise TypeError("source must be a non-empty path string")
        object.__setattr__(self, "source", source)
        if type(self.model_index) is not int or self.model_index < 0:
            raise ValueError("model_index must be a plain integer >= 0")

        selection = self.chain_id
        if selection is None:
            return
        if isinstance(selection, str):
            normalized = (selection,)
        elif isinstance(selection, Sequence) and not isinstance(selection, (bytes, bytearray)):
            normalized = tuple(selection)
        else:
            raise TypeError("chain_id must be None, a string, or a sequence of strings")
        if not normalized:
            raise ValueError("chain_id sequence must not be empty")
        if any(not isinstance(chain, str) for chain in normalized):
            raise TypeError("chain_id sequence entries must be strings")
        if len(normalized) != len(set(normalized)):
            raise ValueError("chain_id sequence must not contain duplicates")
        object.__setattr__(self, "chain_id", normalized)
