from __future__ import annotations

from typing import List, Set, Sequence

import numpy as np


def parse_mode(mode: str) -> Set[str]:
    m = (mode or "").strip()
    if not m:
        return set()
    return {t.strip().lower() for t in m.split("+") if t.strip()}


def safe_float(x: np.ndarray, *, missing: str = "zero") -> np.ndarray:
    y = np.asarray(x, dtype=np.float32)
    if missing == "zero":
        return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if missing == "nan":
        return y.astype(np.float32)
    raise ValueError("missing must be 'zero' or 'nan'")


def onehot(index: np.ndarray, num_classes: int) -> np.ndarray:
    idx = np.asarray(index, dtype=np.int64).reshape(-1)
    n = idx.shape[0]
    out = np.zeros((n, num_classes), dtype=np.float32)
    ok = (idx >= 0) & (idx < num_classes)
    out[np.arange(n)[ok], idx[ok]] = 1.0
    return out


def ensure_1d(x: np.ndarray, n: int, name: str) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim != 1 or a.shape[0] != n:
        raise ValueError(f"{name} must be shape (N,), got {a.shape}")
    return a

def validate_bin_edges(edges: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(edges), dtype=np.float32).reshape(-1)
    if arr.size < 2:
        raise ValueError("edges must have at least 2 values (bin boundaries).")
    if not np.all(np.isfinite(arr)):
        raise ValueError("edges contains non-finite values.")
    if not np.all(arr[1:] > arr[:-1]):
        raise ValueError("edges must be strictly increasing.")
    return arr
