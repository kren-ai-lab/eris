from __future__ import annotations

from typing import Tuple

import numpy as np

from eris.spatial.spatial_index import iter_pairs_within


def neighbor_search_edges(pos: np.ndarray, cutoff: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build residue-residue edges using spatial hashing (cell lists).
    Returns (edge_index (2,E), edge_dist (E,))
    Undirected edges are returned as *both* directions (i->j and j->i).
    """
    n = pos.shape[0]
    if n == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    pairs = list(iter_pairs_within(pos, cutoff))
    pairs_i = [i for i, _, _ in pairs]
    pairs_j = [j for _, j, _ in pairs]
    dists = [d for _, _, d in pairs]

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
