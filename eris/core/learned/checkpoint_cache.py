"""On-disk cache for verified checkpoints."""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from pathlib import Path

from platformdirs import user_cache_dir

from eris.core.learned.checkpoints import (
    CheckpointDescriptor,
    VerifiedCheckpoint,
    verify_checkpoint,
)
from eris.exceptions import CacheError, ExecutionError

_CHECKPOINT_NAMESPACE = "checkpoints"
_CANONICAL_FILENAME = "checkpoint"
_STAGE_PREFIX = ".eris-checkpoint-"


class CheckpointCache:
    """Content-addressed cache for verified checkpoint bytes."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        """Use ``cache_dir``, or the user cache directory for Eris when omitted."""
        self.cache_dir = Path(cache_dir) if cache_dir is not None else Path(user_cache_dir("eris"))

    def _entry_dir(self, descriptor: CheckpointDescriptor) -> Path:
        return self.cache_dir / _CHECKPOINT_NAMESPACE / descriptor.sha256

    def _canonical_path(self, descriptor: CheckpointDescriptor) -> Path:
        return self._entry_dir(descriptor) / _CANONICAL_FILENAME

    def get(self, descriptor: CheckpointDescriptor) -> VerifiedCheckpoint | None:
        """Return a verified cached checkpoint, or a non-destructive cache miss."""
        canonical_path = self._canonical_path(descriptor)
        if not canonical_path.is_file():
            return None

        try:
            return verify_checkpoint(canonical_path, descriptor)
        except ExecutionError as error:
            cause = error.__cause__
            if isinstance(cause, OSError):
                raise CacheError(
                    f"Unable to read cached checkpoint '{descriptor.identifier}' at {canonical_path}: {cause}"
                ) from cause
            return None

    def put(
        self,
        path: str | Path,
        descriptor: CheckpointDescriptor,
    ) -> VerifiedCheckpoint:
        """Safely insert verified candidate bytes and return the canonical cache entry."""
        existing = self.get(descriptor)
        if existing is not None:
            return existing

        candidate_path = Path(path)
        if not candidate_path.exists():
            raise ExecutionError(f"Checkpoint '{descriptor.identifier}' does not exist: {candidate_path}")
        if not candidate_path.is_file():
            raise ExecutionError(
                f"Checkpoint '{descriptor.identifier}' is not a regular file: {candidate_path}"
            )

        entry_dir = self._entry_dir(descriptor)
        canonical_path = self._canonical_path(descriptor)
        staged_path: Path | None = None
        try:
            try:
                entry_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    delete=False,
                    dir=entry_dir,
                    prefix=_STAGE_PREFIX,
                ) as staged_file:
                    staged_path = Path(staged_file.name)
                    with candidate_path.open("rb") as candidate_file:
                        shutil.copyfileobj(candidate_file, staged_file)
                    staged_file.flush()
                    os.fsync(staged_file.fileno())
            except OSError as error:
                raise CacheError(
                    f"Unable to stage checkpoint '{descriptor.identifier}' "
                    f"from {candidate_path} in {entry_dir}: {error}"
                ) from error

            try:
                verified_stage = verify_checkpoint(staged_path, descriptor)
            except ExecutionError as error:
                cause = error.__cause__
                if isinstance(cause, OSError):
                    raise CacheError(
                        f"Unable to verify staged checkpoint '{descriptor.identifier}' "
                        f"at {staged_path}: {cause}"
                    ) from cause
                raise

            try:
                staged_path.replace(canonical_path)
            except OSError as error:
                raise CacheError(
                    f"Unable to install checkpoint '{descriptor.identifier}' at {canonical_path}: {error}"
                ) from error

            return VerifiedCheckpoint(
                path=canonical_path.resolve(),
                descriptor=descriptor,
                sha256=verified_stage.sha256,
            )
        finally:
            if staged_path is not None:
                with contextlib.suppress(OSError):
                    staged_path.unlink(missing_ok=True)


__all__ = ["CheckpointCache"]
