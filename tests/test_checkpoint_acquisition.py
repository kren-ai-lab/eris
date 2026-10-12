from __future__ import annotations

import hashlib
import io
import tempfile
from pathlib import Path
from typing import Any, cast
from urllib.error import URLError

import pytest

from eris.core.learned import checkpoint_acquisition
from eris.core.learned.checkpoint_cache import CheckpointCache
from eris.core.learned.checkpoints import CheckpointDescriptor, StructuralModelDescriptor
from eris.core.learned.runner import StructuralEmbedderRunner
from eris.exceptions import ConfigError, ExecutionError

CHECKPOINT_BYTES = b"official checkpoint fixture\n"
CHECKPOINT_SHA256 = hashlib.sha256(CHECKPOINT_BYTES).hexdigest()
DOWNLOAD_URL = "https://models.example.test/checkpoints/model.pth"


class _StreamingResponse:
    def __init__(self, data: bytes, *, final_url: str = DOWNLOAD_URL) -> None:
        self._stream = io.BytesIO(data)
        self._final_url = final_url
        self.read_sizes: list[int] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback

    def geturl(self) -> str:
        return self._final_url

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return self._stream.read(size)


class _InterruptedResponse(_StreamingResponse):
    def read(self, size: int = -1) -> bytes:
        if self.read_sizes:
            raise OSError("connection interrupted")
        return super().read(size)


def _descriptor(*, download_url: str | None = DOWNLOAD_URL) -> CheckpointDescriptor:
    return CheckpointDescriptor(
        identifier="model.pth",
        sha256=CHECKPOINT_SHA256,
        revision="official-v1",
        source="https://models.example.test/releases/v1",
        download_url=download_url,
    )


def _model(*, download_url: str | None = DOWNLOAD_URL) -> StructuralModelDescriptor:
    return StructuralModelDescriptor(
        architecture="TestModel",
        checkpoint=_descriptor(download_url=download_url),
    )


def _runner(tmp_path: Path) -> StructuralEmbedderRunner:
    return StructuralEmbedderRunner(checkpoint_cache=CheckpointCache(cache_dir=tmp_path / "cache"))


def _force_download_temp_dir(monkeypatch, download_dir: Path) -> None:
    original = tempfile.NamedTemporaryFile

    def named_temporary_file(*args, **kwargs):
        kwargs["dir"] = download_dir
        return original(*args, **kwargs)

    monkeypatch.setattr(
        checkpoint_acquisition.tempfile,
        "NamedTemporaryFile",
        named_temporary_file,
    )


def test_cached_checkpoint_has_priority_and_does_not_open_network(monkeypatch, tmp_path):
    candidate = tmp_path / "candidate.pth"
    candidate.write_bytes(CHECKPOINT_BYTES)
    runner = _runner(tmp_path)
    descriptor = _descriptor()
    cached = runner._checkpoint_cache().put(candidate, descriptor)  # noqa: SLF001

    def fail_urlopen(*args, **kwargs):
        del args, kwargs
        raise AssertionError("network must not be used for a verified cache hit")

    monkeypatch.setattr(checkpoint_acquisition, "_open_registered_url", fail_urlopen)

    resolved = runner._resolve_checkpoint(_model(), None)  # noqa: SLF001

    assert resolved.path == cached.path
    assert resolved.sha256 == CHECKPOINT_SHA256


def test_explicit_checkpoint_has_priority_over_registered_download(monkeypatch, tmp_path):
    candidate = tmp_path / "candidate.pth"
    candidate.write_bytes(CHECKPOINT_BYTES)
    runner = _runner(tmp_path)

    def fail_urlopen(*args, **kwargs):
        del args, kwargs
        raise AssertionError("network must not be used when an explicit checkpoint is available")

    monkeypatch.setattr(checkpoint_acquisition, "_open_registered_url", fail_urlopen)

    resolved = runner._resolve_checkpoint(_model(), candidate)  # noqa: SLF001

    assert resolved.sha256 == CHECKPOINT_SHA256
    assert resolved.path != candidate.resolve()
    assert runner._checkpoint_cache().get(_descriptor()) == resolved  # noqa: SLF001


def test_cache_miss_streams_registered_download_through_verified_cache(monkeypatch, tmp_path):
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()
    _force_download_temp_dir(monkeypatch, download_dir)
    response = _StreamingResponse(CHECKPOINT_BYTES)
    seen_request = {}

    def fake_urlopen(download_request):
        seen_request["url"] = download_request.full_url
        return response

    monkeypatch.setattr(checkpoint_acquisition, "_open_registered_url", fake_urlopen)
    runner = _runner(tmp_path)

    resolved = runner._resolve_checkpoint(_model(), None)  # noqa: SLF001

    assert seen_request == {"url": DOWNLOAD_URL}
    assert response.read_sizes == [1024 * 1024, 1024 * 1024]
    assert resolved.sha256 == CHECKPOINT_SHA256
    assert resolved.path.read_bytes() == CHECKPOINT_BYTES
    assert runner._checkpoint_cache().get(_descriptor()) == resolved  # noqa: SLF001
    assert list(download_dir.iterdir()) == []


def test_bad_download_digest_is_rejected_without_install_or_temp_file(monkeypatch, tmp_path):
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()
    _force_download_temp_dir(monkeypatch, download_dir)
    monkeypatch.setattr(
        checkpoint_acquisition,
        "_open_registered_url",
        lambda *args, **kwargs: _StreamingResponse(b"tampered bytes"),  # noqa: ARG005
    )
    runner = _runner(tmp_path)

    with pytest.raises(ExecutionError, match="failed verification"):
        runner._resolve_checkpoint(_model(), None)  # noqa: SLF001

    assert runner._checkpoint_cache().get(_descriptor()) is None  # noqa: SLF001
    assert list(download_dir.iterdir()) == []


def test_network_failure_is_actionable_and_leaves_no_temp_file(monkeypatch, tmp_path):
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()
    _force_download_temp_dir(monkeypatch, download_dir)

    def fail_urlopen(*args, **kwargs):
        del args, kwargs
        raise URLError("offline")

    monkeypatch.setattr(checkpoint_acquisition, "_open_registered_url", fail_urlopen)
    runner = _runner(tmp_path)

    with pytest.raises(ExecutionError) as raised:
        runner._resolve_checkpoint(_model(), None)  # noqa: SLF001

    message = str(raised.value)
    assert "Automatic download" in message
    assert "offline" in message
    assert "--checkpoint" in message
    assert runner._checkpoint_cache().get(_descriptor()) is None  # noqa: SLF001
    assert list(download_dir.iterdir()) == []


def test_interrupted_transfer_removes_partial_temp_file(monkeypatch, tmp_path):
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()
    _force_download_temp_dir(monkeypatch, download_dir)
    monkeypatch.setattr(
        checkpoint_acquisition,
        "_open_registered_url",
        lambda *args, **kwargs: _InterruptedResponse(CHECKPOINT_BYTES),  # noqa: ARG005
    )

    with pytest.raises(ExecutionError, match="connection interrupted"):
        _runner(tmp_path)._resolve_checkpoint(_model(), None)  # noqa: SLF001

    assert list(download_dir.iterdir()) == []


def test_descriptor_without_registered_url_never_attempts_network(monkeypatch, tmp_path):
    def fail_urlopen(*args, **kwargs):
        del args, kwargs
        raise AssertionError("network must not be used without a registered URL")

    monkeypatch.setattr(checkpoint_acquisition, "_open_registered_url", fail_urlopen)

    with pytest.raises(ExecutionError, match="registered official download") as raised:
        _runner(tmp_path)._resolve_checkpoint(_model(download_url=None), None)  # noqa: SLF001

    assert "--checkpoint" in str(raised.value)


def test_acquisition_url_is_excluded_from_artifact_and_cache_identity():
    first = _model(download_url="https://one.example.test/model.pth")
    second = _model(download_url="https://two.example.test/model.pth")

    assert first.to_dict() == second.to_dict()
    assert first.to_cache_identity() == second.to_cache_identity()
    assert "download_url" not in cast("dict[str, Any]", first.to_dict())["checkpoint"]


@pytest.mark.parametrize(
    "download_url",
    [
        "http://models.example.test/model.pth",
        "https://user:secret@models.example.test/model.pth",
        "https://models.example.test/model.pth#fragment",
        "https://models.example.test:invalid/model.pth",
    ],
)
def test_registered_download_url_must_be_safe_https(download_url):
    with pytest.raises(ConfigError, match="HTTPS URL"):
        _descriptor(download_url=download_url)


def test_redirect_to_non_https_url_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(
        checkpoint_acquisition,
        "_open_registered_url",
        lambda *args, **kwargs: _StreamingResponse(  # noqa: ARG005
            CHECKPOINT_BYTES,
            final_url="http://models.example.test/model.pth",
        ),
    )

    with pytest.raises(ExecutionError, match="unsafe non-HTTPS URL"):
        _runner(tmp_path)._resolve_checkpoint(_model(), None)  # noqa: SLF001
