from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from eris.core.learned.checkpoints import VerifiedCheckpoint
from eris.core.learned.config import StructuralEmbedderConfig
from eris.core.learned.execution import ExecutionCapabilities, ResolvedExecution
from eris.core.learned.types import StructuralEmbeddingResult, StructuralPreprocessingResult

StructureT = TypeVar("StructureT")
PreparedT = TypeVar("PreparedT")
ModelT = TypeVar("ModelT")


@runtime_checkable
class StructuralEmbedderBackend(Protocol[StructureT, PreparedT, ModelT]):
    """Structural contract implemented by framework-specific learned backends."""

    @property
    def backend_id(self) -> str: ...

    def is_available(self) -> bool: ...

    def execution_capabilities(self) -> ExecutionCapabilities: ...

    def preprocess(
        self,
        structure: StructureT,
        *,
        config: StructuralEmbedderConfig,
    ) -> StructuralPreprocessingResult[PreparedT]: ...

    def load(
        self,
        checkpoint: VerifiedCheckpoint,
        *,
        execution: ResolvedExecution,
    ) -> ModelT: ...

    def infer(
        self,
        model: ModelT,
        prepared: PreparedT,
        *,
        config: StructuralEmbedderConfig,
        execution: ResolvedExecution,
    ) -> StructuralEmbeddingResult: ...
