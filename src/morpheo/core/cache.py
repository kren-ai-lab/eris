from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import json
import numpy as np
from platformdirs import user_cache_dir

from morpheo.core.hashing import sha256_text


def default_cache_dir() -> Path:
    p = Path(user_cache_dir("morpheo"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_cache_key(*parts: str) -> str:
    return sha256_text("|".join(parts))


class DiskCache:
    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = cache_dir or default_cache_dir()

    def _paths(self, key: str) -> Tuple[Path, Path]:
        base = self.cache_dir / key
        return base.with_suffix(".npz"), base.with_suffix(".json")

    def get(self, key: str) -> Optional[Tuple[Dict[str, np.ndarray], Dict[str, Any]]]:
        npz_path, meta_path = self._paths(key)
        if not npz_path.exists() or not meta_path.exists():
            return None
        npz = np.load(npz_path)
        data = {k: npz[k] for k in npz.files}
        meta = json.loads(meta_path.read_text())
        return data, meta

    def set(self, key: str, data: Dict[str, np.ndarray], meta: Dict[str, Any]) -> None:
        npz_path, meta_path = self._paths(key)
        np.savez_compressed(npz_path, **data)
        meta_path.write_text(json.dumps(meta, indent=2))
