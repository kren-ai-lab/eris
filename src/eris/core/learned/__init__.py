"""Framework-independent contracts for learned structural representations."""

from eris.core.learned.artifacts import build_learned_artifact
from eris.core.learned.checkpoint_cache import CheckpointCache
from eris.core.learned.checkpoints import (
    CheckpointDescriptor,
    StructuralModelDescriptor,
    VerifiedCheckpoint,
    verify_checkpoint,
)
from eris.core.learned.config import OutputMode, StructuralEmbedderConfig
from eris.core.learned.execution import (
    ExecutionCapabilities,
    Precision,
    ResolvedExecution,
    StructuralExecutionConfig,
    build_execution_provenance,
    execution_cache_identity,
    resolve_execution,
)
from eris.core.learned.protocol import StructuralEmbedderBackend
from eris.core.learned.registry import (
    BackendAvailability,
    EmbedderInfo,
    EmbedderRegistry,
    learned_registry,
)
from eris.core.learned.types import StructuralEmbeddingResult, validate_embedding_result

__all__ = [
    "BackendAvailability",
    "CheckpointDescriptor",
    "CheckpointCache",
    "EmbedderInfo",
    "EmbedderRegistry",
    "ExecutionCapabilities",
    "OutputMode",
    "Precision",
    "ResolvedExecution",
    "StructuralEmbedderBackend",
    "StructuralEmbedderConfig",
    "StructuralEmbeddingResult",
    "StructuralExecutionConfig",
    "StructuralModelDescriptor",
    "VerifiedCheckpoint",
    "build_learned_artifact",
    "build_execution_provenance",
    "execution_cache_identity",
    "learned_registry",
    "resolve_execution",
    "validate_embedding_result",
    "verify_checkpoint",
]
