from __future__ import annotations

from typing import Any, Dict, ClassVar, Tuple

import numpy as np

from eris.spatial.spatial_index import cell_index, build_cell_dict, _NEIGHBOR_OFFSETS


def neighbor_search_edges(pos: np.ndarray, cutoff: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build residue-residue edges using spatial hashing (cell lists).
    Returns (edge_index (2,E), edge_dist (E,))
    Undirected edges are returned as *both* directions (i->j and j->i).
    """
    n = pos.shape[0]
    if n == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    r = float(cutoff)
    r2 = r * r

    cells = cell_index(pos, r)
    cell_dict = build_cell_dict(cells)

    # collect unique undirected pairs i<j
    pairs_i: list[int] = []
    pairs_j: list[int] = []
    dists: list[float] = []

    for cell_key, idxs in cell_dict.items():
        cx, cy, cz = cell_key

        # only generate pairs where (cell_key, neighbor_key) is ordered to avoid duplicates
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            nkey = (cx + dx, cy + dy, cz + dz)
            if nkey not in cell_dict:
                continue

            # ordering to avoid double counting across cells
            if nkey < cell_key:
                continue

            jdxs = cell_dict[nkey]

            if nkey == cell_key:
                # within same cell: compare i<j
                for a in range(len(idxs)):
                    i = idxs[a]
                    pi = pos[i]
                    for b in range(a + 1, len(idxs)):
                        j = idxs[b]
                        diff = pi - pos[j]
                        dsq = float(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
                        if dsq <= r2:
                            pairs_i.append(i)
                            pairs_j.append(j)
                            dists.append(np.sqrt(dsq))
            else:
                # across two different cells: full cross
                for i in idxs:
                    pi = pos[i]
                    for j in jdxs:
                        diff = pi - pos[j]
                        dsq = float(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
                        if dsq <= r2:
                            pairs_i.append(i)
                            pairs_j.append(j)
                            dists.append(np.sqrt(dsq))

    if not pairs_i:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    # make directed edges: i->j and j->i
    i_arr = np.asarray(pairs_i, dtype=np.int64)
    j_arr = np.asarray(pairs_j, dtype=np.int64)
    dist_arr = np.asarray(dists, dtype=np.float32)

    edge_i = np.concatenate([i_arr, j_arr], axis=0)
    edge_j = np.concatenate([j_arr, i_arr], axis=0)
    edge_dist = np.concatenate([dist_arr, dist_arr], axis=0)

    edge_index = np.stack([edge_i, edge_j], axis=0).astype(np.int64)
    return edge_index, edge_dist