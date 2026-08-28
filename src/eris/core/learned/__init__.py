"""Framework-independent contracts for learned structural representations."""

from eris.core.learned.artifacts import build_learned_artifact
from eris.core.learned.backends.gearnet.registration import register_gearnet_edge
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
    execution_request_cache_identity,
    resolve_execution,
)
from eris.core.learned.protocol import StructuralEmbedderBackend
from eris.core.learned.registry import (
    BackendAvailability,
    EmbedderInfo,
    EmbedderRegistry,
    learned_registry,
)
from eris.core.learned.runner import StructuralEmbedderRunner
from eris.core.learned.types import (
    StructuralEmbeddingResult,
    StructuralPreprocessingResult,
    validate_embedding_result,
)

register_gearnet_edge(learned_registry)

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
    "StructuralEmbedderRunner",
    "StructuralEmbeddingResult",
    "StructuralPreprocessingResult",
    "StructuralExecutionConfig",
    "StructuralModelDescriptor",
    "VerifiedCheckpoint",
    "build_learned_artifact",
    "build_execution_provenance",
    "execution_cache_identity",
    "execution_request_cache_identity",
    "learned_registry",
    "resolve_execution",
    "validate_embedding_result",
    "verify_checkpoint",
]
