from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol


@dataclass(frozen=True)
class StructureInput:
    """Definition of a structure input (path, id, selection, etc.)."""
    source: str
    chain_id: Optional[str] = None
    model_index: int = 0


@dataclass
class StructureArtifact:
    """Serialized representation + metadata."""
    encoder_name: str
    data: Any
    meta: Dict[str, Any]


class StructuralEncoder(Protocol):
    """Common interface for all encoders."""
    name: str

    def fingerprint(self) -> str: ...
    def encode(self, inp: StructureInput) -> StructureArtifact: ...
