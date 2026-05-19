from __future__ import annotations

import numpy as np

from eris.features.graph_stats import compute_graph_stats, pack_graph_stats


def test_graph_stats_basic():
    # Simple chain of 3 nodes: 0-1-2
    edge_index = np.array([[0, 1], [1, 2]], dtype=np.int64)
    edge_dist = np.array([5.0, 6.0], dtype=np.float32)

    stats = compute_graph_stats(edge_index=edge_index, edge_dist=edge_dist, n_nodes=3)
    assert stats["n_nodes"] == 3.0
    assert stats["n_edges"] == 2.0
    assert stats["degree_max"] >= 1.0
    assert stats["edge_dist_mean"] > 0.0

    vec, keys = pack_graph_stats(stats)
    assert vec.ndim == 1
    assert len(keys) == vec.shape[0]
