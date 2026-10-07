from __future__ import annotations

import os
import tempfile
from http.client import HTTPException
from pathlib import Path
from urllib import error, request
from urllib.parse import urlsplit

from eris.core.learned.checkpoint_cache import CheckpointCache
from eris.core.learned.checkpoints import CheckpointDescriptor, VerifiedCheckpoint
from eris.exceptions import CacheError, ExecutionError

_CHUNK_SIZE = 1024 * 1024
_DOWNLOAD_PREFIX = ".eris-checkpoint-download-"
_TIMEOUT_SECONDS = 60
_USER_AGENT = "Eris/0.1 official-checkpoint-acquisition"


def _require_safe_final_url(url: str, *, identifier: str) -> None:
    parsed = urlsplit(url)
    try:
        hostname = parsed.hostname
        parsed.port
    except ValueError as error:
        raise ExecutionError(
            f"Official checkpoint '{identifier}' redirected to an invalid URL. "
            "Provide a local checkpoint_path (CLI: --checkpoint) instead."
        ) from error
    if (
        parsed.scheme != "https"
        or hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ExecutionError(
            f"Official checkpoint '{identifier}' redirected to an unsafe non-HTTPS URL. "
            "Provide a local checkpoint_path (CLI: --checkpoint) instead."
        )


class _HttpsOnlyRedirectHandler(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _require_safe_final_url(newurl, identifier="registered checkpoint")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_registered_url(download_request: request.Request):
    opener = request.build_opener(_HttpsOnlyRedirectHandler())
    return opener.open(download_request, timeout=_TIMEOUT_SECONDS)


def acquire_registered_checkpoint(
    descriptor: CheckpointDescriptor,
    cache: CheckpointCache,
) -> VerifiedCheckpoint:
    """Stream a registered official checkpoint into the verified canonical cache."""

    download_url = descriptor.download_url
    if download_url is None:
        raise ExecutionError(
            f"No cached checkpoint or registered official download is available for "
            f"'{descriptor.identifier}'; provide a local checkpoint_path (CLI: --checkpoint)"
        )

    temporary_path: Path | None = None
    try:
        try:
            download_request = request.Request(
                download_url,
                headers={"User-Agent": _USER_AGENT},
            )
            with _open_registered_url(download_request) as response:
                _require_safe_final_url(response.geturl(), identifier=descriptor.identifier)
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    delete=False,
                    prefix=_DOWNLOAD_PREFIX,
                ) as temporary_file:
                    temporary_path = Path(temporary_file.name)
                    while chunk := response.read(_CHUNK_SIZE):
                        temporary_file.write(chunk)
                    temporary_file.flush()
                    os.fsync(temporary_file.fileno())
        except error.HTTPError as download_error:
            raise ExecutionError(
                f"Automatic download of official checkpoint '{descriptor.identifier}' failed "
                f"with HTTP {download_error.code}. Retry with network access or provide a local "
                "checkpoint_path (CLI: --checkpoint)."
            ) from download_error
        except (error.URLError, HTTPException, TimeoutError, OSError) as download_error:
            reason = getattr(download_error, "reason", download_error)
            raise ExecutionError(
                f"Automatic download of official checkpoint '{descriptor.identifier}' failed: "
                f"{reason}. Retry with network access or provide a local checkpoint_path "
                "(CLI: --checkpoint)."
            ) from download_error

        try:
            return cache.put(temporary_path, descriptor)
        except ExecutionError as verification_error:
            raise ExecutionError(
                f"Automatic download of official checkpoint '{descriptor.identifier}' failed "
                f"verification: {verification_error}. Retry the download or provide a verified "
                "local checkpoint_path (CLI: --checkpoint)."
            ) from verification_error
        except CacheError as cache_error:
            raise CacheError(
                f"Automatic download of official checkpoint '{descriptor.identifier}' could not "
                f"be installed in the checkpoint cache: {cache_error}"
            ) from cache_error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
