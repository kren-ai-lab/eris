from __future__ import annotations

import numpy as np

from eris.spatial.spatial_index import iter_pairs_within


def _collect_local_neighbor_candidates(pos: np.ndarray, cutoff: float) -> list[list[tuple[int, float]]]:
    """Collect neighbor candidates within cutoff for each node using spatial hashing.

    Returns
    -------
    neigh : list of length N
        neigh[i] is a list of (j, dist_ij) with dist_ij <= cutoff and j != i.

    """
    neigh: list[list[tuple[int, float]]] = [[] for _ in range(pos.shape[0])]
    for i, j, dist in iter_pairs_within(pos, cutoff):
        neigh[i].append((j, dist))
        neigh[j].append((i, dist))

    return neigh


def _collect_global_neighbor_candidates(pos: np.ndarray) -> list[list[tuple[int, float]]]:
    """Collect all neighbor candidates without considering cutoff.

    Returns
    -------
    neigh : list of length N
        neigh[i] is a list of (j, dist_ij).

    """
    n = pos.shape[0]
    neigh: list[list[tuple[int, float]]] = [[] for _ in range(n)]

    for i in range(n):
        pi = pos[i]
        for j in range(n):
            if i == j:
                continue
            diff = pi - pos[j]
            dist = float(np.sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]))
            neigh[i].append((j, dist))

    return neigh


def _knn_edges_from_candidates(
    candidates: list[list[tuple[int, float]]],
    k: int,
    symmetric: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build KNN edges from per-node candidate lists.

    Parameters
    ----------
    candidates
        candidates[i] = list of (j, dist)
    k
        number of nearest neighbors per node
    symmetric
        if True, ensure undirectedness by adding reverse edges for all edges

    Returns
    -------
    edge_index : (2,E) int64
    edge_dist : (E,) float32

    """
    n = len(candidates)
    k = int(k)

    edges_i: list[int] = []
    edges_j: list[int] = []
    dists: list[float] = []

    for i in range(n):
        neigh = candidates[i]
        if not neigh:
            continue

        # Sort by distance with node index as an explicit deterministic tie-breaker.
        neigh_sorted = sorted(neigh, key=lambda t: (t[1], t[0]))[:k]

        for j, dist in neigh_sorted:
            edges_i.append(i)
            edges_j.append(int(j))
            dists.append(float(dist))

    if not edges_i:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    if symmetric:
        original_edges = list(zip(edges_i, edges_j, dists))
        closure = original_edges + [(j, i, dist) for i, j, dist in original_edges]

        seen: set[tuple[int, int]] = set()
        unique_i: list[int] = []
        unique_j: list[int] = []
        unique_dists: list[float] = []

        for i, j, dist in closure:
            edge = (i, j)
            if edge in seen:
                continue
            seen.add(edge)
            unique_i.append(i)
            unique_j.append(j)
            unique_dists.append(dist)

        edges_i = unique_i
        edges_j = unique_j
        dists = unique_dists

    edge_index = np.stack([np.asarray(edges_i, dtype=np.int64), np.asarray(edges_j, dtype=np.int64)], axis=0)
    edge_dist = np.asarray(dists, dtype=np.float32)
    return edge_index, edge_dist


def knn_edges(
    pos: np.ndarray,
    k: int,
    *,
    cutoff: float | None = None,
    symmetric: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """KNN builder.

    Parameters
    ----------
    pos : (N,3)
    k : int
    cutoff : optional
        If provided → uses spatial candidates (optimized)
        If None → uses global brute force

    """
    if cutoff is not None:
        candidates = _collect_local_neighbor_candidates(pos, cutoff)
    else:
        candidates = _collect_global_neighbor_candidates(pos)

    return _knn_edges_from_candidates(candidates, k, symmetric=symmetric)
