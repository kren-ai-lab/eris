from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from morpheo.core.artifacts import StructureArtifact


@dataclass(frozen=True)
class StructureInput:
    source: str
    chain_id: Optional[str] = None
    model_index: int = 0

    def to_string(self) -> str:
        return f"source={self.source}|chain={self.chain_id}|model={self.model_index}"


class StructuralEncoder(Protocol):
    name: str

    def fingerprint(self) -> str: ...
    def encode(self, inp: StructureInput) -> StructureArtifact: ...
