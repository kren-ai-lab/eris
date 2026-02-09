from __future__ import annotations

from pathlib import Path
from typing import Optional

from platformdirs import user_cache_dir


def default_cache_dir() -> Path:
    p = Path(user_cache_dir("morpheo"))
    p.mkdir(parents=True, exist_ok=True)
    return p


class DiskCache:
    """Placeholder cache implementation (to be extended)."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = cache_dir or default_cache_dir()
