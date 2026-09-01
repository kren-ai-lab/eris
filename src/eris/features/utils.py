from __future__ import annotations

from typing import List, Set, Sequence, Any, Tuple, Dict

import numpy as np


def parse_mode(mode: Any) -> List[Tuple[str, Dict[str, Any]]]:
    if isinstance(mode, str):
        return [(t.strip(), {}) for t in mode.split("+") if t.strip()]

    if isinstance(mode, list):
        result = []

        for item in mode:
            if isinstance(item, str):
                t = item.strip()
                if t:
                    result.append((t, {}))

            elif isinstance(item, dict):
                if len(item) != 1:
                    raise ValueError(f"Invalid feature config: {item}")

                token, params = next(iter(item.items()))

                if not isinstance(token, str):
                    raise ValueError(f"Feature token must be string: {item}")

                if params is None:
                    params = {}

                if not isinstance(params, dict):
                    raise ValueError(f"Params for '{token}' must be dict")

                result.append((token.strip(), params))

            else:
                raise ValueError(f"Invalid mode element: {item}")

        return result

    raise ValueError(
        "Mode must be string ('a+b'), list of strings ['a','b'], or list of dict [{'a': {...}}]"
    )


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
