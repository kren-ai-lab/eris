from __future__ import annotations

from typing import List, Tuple

import numpy as np

_NEIGHBOR_OFFSETS = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]


def cell_index(pos: np.ndarray, cell_size: float) -> np.ndarray:
    return np.floor(pos / float(cell_size)).astype(np.int64)


def build_cell_dict(cells: np.ndarray) -> dict[Tuple[int, int, int], List[int]]:
    d: dict[Tuple[int, int, int], List[int]] = {}
    for i, c in enumerate(cells):
        key = (int(c[0]), int(c[1]), int(c[2]))
        d.setdefault(key, []).append(i)
    return d
