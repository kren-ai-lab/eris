"""Framework-independent contracts for learned structural representations."""

from eris.core.learned.checkpoints import (
    CheckpointDescriptor,
    StructuralModelDescriptor,
    VerifiedCheckpoint,
    verify_checkpoint,
)
from eris.core.learned.config import OutputMode, StructuralEmbedderConfig
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
    "EmbedderInfo",
    "EmbedderRegistry",
    "OutputMode",
    "StructuralEmbedderBackend",
    "StructuralEmbedderConfig",
    "StructuralEmbeddingResult",
    "StructuralModelDescriptor",
    "VerifiedCheckpoint",
    "learned_registry",
    "validate_embedding_result",
    "verify_checkpoint",
]
