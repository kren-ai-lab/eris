from __future__ import annotations

import numpy as np


def _safe_float(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(x)


def compute_graph_stats(
    edge_index: np.ndarray,
    edge_dist: np.ndarray | None,
    n_nodes: int,
) -> dict[str, float]:
    """
    Deterministic graph-level statistics for graph representations.

    Parameters
    ----------
    edge_index:
        (2, E) int array
    edge_dist:
        (E,) float array or None
    n_nodes:
        number of nodes N

    Returns
    -------
    Dict[str, float]
        Flat dict with scalar stats. Safe (no NaNs), deterministic.
    """
    ei = np.asarray(edge_index, dtype=np.int64)
    if ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError(f"edge_index must be (2,E), got {ei.shape}")

    E = int(ei.shape[1])
    N = int(n_nodes)

    if N <= 0:
        raise ValueError(f"n_nodes must be > 0, got {N}")

    # Graph relationships use the simple undirected projection of non-loop edges.
    deg = np.zeros((N,), dtype=np.int64)
    src = ei[0]
    dst = ei[1]

    # clamp indices defensively
    if (
        src.min(initial=0) < 0
        or dst.min(initial=0) < 0
        or src.max(initial=0) >= N
        or dst.max(initial=0) >= N
    ):
        raise ValueError("edge_index contains node ids outside [0, n_nodes).")

    loop_mask = src == dst
    non_loop_mask = ~loop_mask
    n_self_loops = int(np.count_nonzero(loop_mask))

    non_loop_src = src[non_loop_mask]
    non_loop_dst = dst[non_loop_mask]

    if non_loop_src.size > 0:
        unordered_pairs = np.column_stack(
            (
                np.minimum(non_loop_src, non_loop_dst),
                np.maximum(non_loop_src, non_loop_dst),
            )
        )
        unique_pairs = np.unique(unordered_pairs, axis=0)
        np.add.at(deg, unique_pairs[:, 0], 1)
        np.add.at(deg, unique_pairs[:, 1], 1)
        n_unique_pairs = int(unique_pairs.shape[0])
    else:
        n_unique_pairs = 0

    deg_f = deg.astype(np.float32)
    deg_mean = float(deg_f.mean())
    deg_std = float(deg_f.std())
    deg_max = float(deg_f.max())
    deg_min = float(deg_f.min())

    # Density of the simple undirected projection, excluding self-loops.
    denom = N * (N - 1)
    density = float((2.0 * n_unique_pairs) / denom) if denom > 0 else 0.0

    stats: dict[str, float] = {
        "n_nodes": float(N),
        "n_edges": float(E),
        "n_self_loops": float(n_self_loops),
        "density": _safe_float(density),
        "degree_mean": _safe_float(deg_mean),
        "degree_std": _safe_float(deg_std),
        "degree_min": _safe_float(deg_min),
        "degree_max": _safe_float(deg_max),
    }

    if edge_dist is not None:
        ed = np.asarray(edge_dist, dtype=np.float32).reshape(-1)
        if ed.shape[0] != E:
            raise ValueError(f"edge_dist must have length E={E}, got {ed.shape}")
        # remove any weird values safely
        ed = np.nan_to_num(ed, nan=0.0, posinf=0.0, neginf=0.0)
        non_loop_dist = ed[non_loop_mask]

        if non_loop_dist.size > 0:
            stats.update(
                {
                    "edge_dist_mean": _safe_float(float(non_loop_dist.mean())),
                    "edge_dist_std": _safe_float(float(non_loop_dist.std())),
                    "edge_dist_min": _safe_float(float(non_loop_dist.min())),
                    "edge_dist_max": _safe_float(float(non_loop_dist.max())),
                }
            )
        else:
            stats.update(
                {
                    "edge_dist_mean": 0.0,
                    "edge_dist_std": 0.0,
                    "edge_dist_min": 0.0,
                    "edge_dist_max": 0.0,
                }
            )
    else:
        stats.update(
            {
                "edge_dist_mean": 0.0,
                "edge_dist_std": 0.0,
                "edge_dist_min": 0.0,
                "edge_dist_max": 0.0,
            }
        )

    # Optional: simple connectivity proxy via isolated nodes fraction
    isolated = float(np.sum(deg == 0))
    stats["isolated_frac"] = _safe_float(isolated / float(N))

    return stats


def pack_graph_stats(stats: dict[str, float]) -> tuple[np.ndarray, list[str]]:
    """
    Convert dict stats -> (graph_attr vector, keys) in stable key order.
    """
    keys = sorted(stats.keys())
    vec = np.asarray([stats[k] for k in keys], dtype=np.float32)
    return vec, keys
