"""Eris exception hierarchy."""


class ErisError(Exception):
    """Base exception for Eris."""


class ExecutionError(ErisError):
    """Execution error for Eris."""


class RepresentationNotFoundError(ErisError):
    """Raised when a public representation or embedder name is unknown."""


class ConfigError(ErisError):
    """Raised when a configuration is invalid."""


class CacheError(ErisError):
    """Raised when cache operations fail."""
