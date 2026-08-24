from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from eris.core.learned.checkpoints import VerifiedCheckpoint
from eris.core.learned.config import StructuralEmbedderConfig
from eris.core.learned.types import StructuralEmbeddingResult

StructureT = TypeVar("StructureT")
PreparedT = TypeVar("PreparedT")
ModelT = TypeVar("ModelT")


@runtime_checkable
class StructuralEmbedderBackend(Protocol[StructureT, PreparedT, ModelT]):
    """Structural contract implemented by framework-specific learned backends."""

    @property
    def backend_id(self) -> str: ...

    def is_available(self) -> bool: ...

    def preprocess(
        self,
        structure: StructureT,
        *,
        config: StructuralEmbedderConfig,
    ) -> PreparedT: ...

    def load(self, checkpoint: VerifiedCheckpoint) -> ModelT: ...

    def infer(
        self,
        model: ModelT,
        prepared: PreparedT,
        *,
        config: StructuralEmbedderConfig,
    ) -> StructuralEmbeddingResult: ...
