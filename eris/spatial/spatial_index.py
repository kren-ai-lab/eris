"""Cell-based spatial indexing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

_NEIGHBOR_OFFSETS = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]


def cell_index(pos: np.ndarray, cell_size: float) -> np.ndarray:
    """Return the integer grid cell of each position."""
    return np.floor(pos / float(cell_size)).astype(np.int64)


def build_cell_dict(cells: np.ndarray) -> dict[tuple[int, int, int], list[int]]:
    """Map each grid cell to the indexes of the positions in it."""
    d: dict[tuple[int, int, int], list[int]] = {}
    for i, c in enumerate(cells):
        key = (int(c[0]), int(c[1]), int(c[2]))
        d.setdefault(key, []).append(i)
    return d


def iter_pairs_within(pos: np.ndarray, cutoff: float) -> Iterator[tuple[int, int, float]]:
    """Yield each unordered pair ``(i, j, dist)`` with ``dist <= cutoff`` once, in cell-list order."""
    r = float(cutoff)
    r2 = r * r
    cell_dict = build_cell_dict(cell_index(pos, r))

    for cell_key, idxs in cell_dict.items():
        cx, cy, cz = cell_key
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            nkey = (cx + dx, cy + dy, cz + dz)
            # ordering to avoid double counting across cells
            if nkey not in cell_dict or nkey < cell_key:
                continue

            if nkey == cell_key:
                # within same cell: compare i<j
                pairs = ((idxs[a], idxs[b]) for a in range(len(idxs)) for b in range(a + 1, len(idxs)))
            else:
                # across two different cells: full cross
                pairs = ((i, j) for i in idxs for j in cell_dict[nkey])

            for i, j in pairs:
                diff = pos[i] - pos[j]
                dsq = float(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
                if dsq <= r2:
                    yield i, j, float(np.sqrt(dsq))
