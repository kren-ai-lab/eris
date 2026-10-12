"""SHA-256 helpers."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def sha256_text(text: str) -> str:
    """Return the hex SHA-256 of UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
