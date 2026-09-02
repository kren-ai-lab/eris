from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from eris.core.artifacts import StructureArtifact


@dataclass(frozen=True)
class StructureInput:
    source: str
    chain_id: Optional[str] = None
    model_index: int = 0
