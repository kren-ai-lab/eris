from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from eris.core.hashing import sha256_file
from eris.exceptions import ConfigError, ExecutionError

_LOWERCASE_HEX = frozenset("0123456789abcdef")


def _require_canonical_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{field_name} must be a string")
    if not value.strip():
        raise ConfigError(f"{field_name} must not be empty")
    if value != value.strip():
        raise ConfigError(f"{field_name} must not contain surrounding whitespace")
    return value


def _require_sha256(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _LOWERCASE_HEX for character in value)
    ):
        raise ConfigError("sha256 must be a lowercase 64-character SHA-256 hex string")
    return value


def _require_https_url(value: object, field_name: str) -> str:
    url = _require_canonical_string(value, field_name)
    parsed = urlsplit(url)
    try:
        hostname = parsed.hostname
        parsed.port
    except ValueError as error:
        raise ConfigError(f"{field_name} must be a valid HTTPS URL") from error
    if (
        parsed.scheme != "https"
        or hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ConfigError(f"{field_name} must be an HTTPS URL without credentials or a fragment")
    return url


@dataclass(frozen=True, kw_only=True)
class CheckpointDescriptor:
    """Path-independent identity and acquisition metadata for one checkpoint."""

    identifier: str
    sha256: str
    revision: str | None = None
    source: str | None = None
    download_url: str | None = None

    def __post_init__(self) -> None:
        _require_canonical_string(self.identifier, "identifier")
        _require_sha256(self.sha256)
        if self.revision is not None:
            _require_canonical_string(self.revision, "revision")
        if self.source is not None:
            _require_canonical_string(self.source, "source")
        if self.download_url is not None:
            _require_https_url(self.download_url, "download_url")

    def to_dict(self) -> dict[str, str]:
        """Return deterministic artifact schema checkpoint metadata."""

        metadata = {
            "identifier": self.identifier,
            "sha256": self.sha256,
        }
        if self.revision is not None:
            metadata["revision"] = self.revision
        if self.source is not None:
            metadata["source"] = self.source
        return metadata


@dataclass(frozen=True, kw_only=True)
class StructuralModelDescriptor:
    """Generic structural model identity independent of its backend implementation."""

    architecture: str
    checkpoint: CheckpointDescriptor

    def __post_init__(self) -> None:
        _require_canonical_string(self.architecture, "architecture")
        if not isinstance(self.checkpoint, CheckpointDescriptor):
            raise ConfigError("checkpoint must be a CheckpointDescriptor")

    def to_dict(self) -> dict[str, object]:
        """Return complete artifact schema model metadata."""

        return {
            "architecture": self.architecture,
            "checkpoint": self.checkpoint.to_dict(),
        }

    def to_cache_identity(self) -> dict[str, object]:
        """Return scientific model identity without acquisition location."""

        checkpoint = {
            "identifier": self.checkpoint.identifier,
        }
        if self.checkpoint.revision is not None:
            checkpoint["revision"] = self.checkpoint.revision
        checkpoint["sha256"] = self.checkpoint.sha256
        return {
            "architecture": self.architecture,
            "checkpoint": checkpoint,
        }


@dataclass(frozen=True, kw_only=True)
class VerifiedCheckpoint:
    """Runtime result proving that a local file matched its expected byte digest."""

    path: Path
    descriptor: CheckpointDescriptor
    sha256: str


def verify_checkpoint(
    path: str | Path,
    descriptor: CheckpointDescriptor,
) -> VerifiedCheckpoint:
    """Verify local checkpoint bytes against a trusted descriptor digest.

    Digest equality establishes byte identity only; it does not establish authenticity,
    licensing, safe deserialization, or scientific compatibility.
    """

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise ExecutionError(
            f"Checkpoint '{descriptor.identifier}' does not exist: {checkpoint_path}"
        )
    if not checkpoint_path.is_file():
        raise ExecutionError(
            f"Checkpoint '{descriptor.identifier}' is not a regular file: {checkpoint_path}"
        )

    try:
        actual_sha256 = sha256_file(checkpoint_path)
    except OSError as error:
        raise ExecutionError(
            f"Unable to read checkpoint '{descriptor.identifier}' at {checkpoint_path}: {error}"
        ) from error

    if actual_sha256 != descriptor.sha256:
        raise ExecutionError(
            f"Checkpoint '{descriptor.identifier}' SHA-256 mismatch at {checkpoint_path}: "
            f"expected {descriptor.sha256}, actual {actual_sha256}"
        )

    return VerifiedCheckpoint(
        path=checkpoint_path.resolve(),
        descriptor=descriptor,
        sha256=actual_sha256,
    )


__all__ = [
    "CheckpointDescriptor",
    "StructuralModelDescriptor",
    "VerifiedCheckpoint",
    "verify_checkpoint",
]
