from __future__ import annotations

import pytest


def test_pool_graph_basic():
    torch = pytest.importorskip("torch")
    from morpheo.pooling import pool_graph

    # 3 nodes, 2-dim features
    x = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]])

    # 2 edges, 1-dim features
    edge_attr = torch.tensor([[10.0],
                              [20.0]])

    graph_attr = torch.tensor([7.0, 8.0])

    sample = {"x": x, "edge_attr": edge_attr, "graph_attr": graph_attr}

    g = pool_graph(sample, stats=("mean", "min", "max"), include_edges=True, include_graph_attr=True)
    # dims: node (3 stats * 2) + edge (3 stats * 1) + graph (2) = 6 + 3 + 2 = 11
    assert g.ndim == 1
    assert g.numel() == 11

    # quick sanity on first block (node mean, min, max)
    # mean: [3,4], min: [1,2], max: [5,6]
    assert torch.allclose(g[0:2], torch.tensor([3.0, 4.0]))
    assert torch.allclose(g[2:4], torch.tensor([1.0, 2.0]))
    assert torch.allclose(g[4:6], torch.tensor([5.0, 6.0]))

    # graph_attr appended at end
    assert torch.allclose(g[-2:], torch.tensor([7.0, 8.0]))


def test_pool_graph_no_edges_ok():
    torch = pytest.importorskip("torch")
    from morpheo.pooling import pool_graph

    x = torch.randn(5, 4)
    sample = {"x": x}  # no edges, no graph_attr

    g = pool_graph(sample, include_edges=True, include_graph_attr=True)
    # default stats 4 -> 4*4 = 16
    assert g.numel() == 16
