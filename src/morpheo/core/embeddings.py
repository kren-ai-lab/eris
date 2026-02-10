from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def load_node_embeddings(path: Path, key: str = "node_emb") -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        return arr.astype(np.float32)
    if path.suffix.lower() == ".npz":
        npz = np.load(path)
        if key not in npz.files:
            raise KeyError(f"Key '{key}' not found in {path}. Available: {npz.files}")
        return npz[key].astype(np.float32)
    raise ValueError(f"Unsupported embedding file: {path} (use .npy or .npz)")
