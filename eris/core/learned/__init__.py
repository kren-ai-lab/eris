"""Framework-independent contracts for learned structural representations."""

from eris.core.learned.artifacts import build_learned_artifact
from eris.core.learned.backends.gearnet.dependencies import BackendAvailability
from eris.core.learned.checkpoint_cache import CheckpointCache
from eris.core.learned.checkpoints import (
    CheckpointDescriptor,
    StructuralModelDescriptor,
    VerifiedCheckpoint,
    verify_checkpoint,
)
from eris.core.learned.config import OutputMode, StructuralEmbedderConfig
from eris.core.learned.execution import (
    Precision,
    ResolvedExecution,
    StructuralExecutionConfig,
    build_execution_provenance,
    execution_request_cache_identity,
    resolve_execution,
)
from eris.core.learned.registry import GEARNET_EDGE_INFO, EmbedderInfo, learned_registry
from eris.core.learned.runner import StructuralEmbedderRunner
from eris.core.learned.types import (
    StructuralEmbeddingResult,
    validate_embedding_result,
)

__all__ = [
    "BackendAvailability",
    "CheckpointDescriptor",
    "CheckpointCache",
    "EmbedderInfo",
    "GEARNET_EDGE_INFO",
    "OutputMode",
    "Precision",
    "ResolvedExecution",
    "StructuralEmbedderConfig",
    "StructuralEmbedderRunner",
    "StructuralEmbeddingResult",
    "StructuralExecutionConfig",
    "StructuralModelDescriptor",
    "VerifiedCheckpoint",
    "build_learned_artifact",
    "build_execution_provenance",
    "execution_request_cache_identity",
    "learned_registry",
    "resolve_execution",
    "validate_embedding_result",
    "verify_checkpoint",
]
