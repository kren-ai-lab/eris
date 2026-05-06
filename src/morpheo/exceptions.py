class MorpheoError(Exception):
    """Base exception for Morpheo."""

class ExecutionError(MorpheoError):
    """Execution error for Morpheo."""

class EncoderNotFoundError(MorpheoError):
    """Raised when an encoder is not registered."""


class ConfigError(MorpheoError):
    """Raised when a configuration is invalid."""


class CacheError(MorpheoError):
    """Raised when cache operations fail."""
