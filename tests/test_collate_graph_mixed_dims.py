from __future__ import annotations

import pytest


def test_collate_graph_pads_mixed_x_and_edge_attr():
    torch = pytest.importorskip("torch")

    # Import directly from your collate module path if needed
    from morpheo.datasets.collate import collate_graph

    # sample 1: x has F=22, edge_attr Fe=8
    x1 = torch.zeros((5, 22), dtype=torch.float32)
    ei1 = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int64)  # E=3
    ea1 = torch.zeros((3, 8), dtype=torch.float32)
    ga1 = torch.zeros((10,), dtype=torch.float32)

    s1 = {
        "x": x1,
        "edge_index": ei1,
        "edge_attr": ea1,
        "graph_attr": ga1,
        "graph_attr_keys": [f"k{i}" for i in range(10)],
        "id": "a",
        "encoder": "residue_contact_graph_ns",
    }

    # sample 2: x has F=30, edge_attr Fe=16
    x2 = torch.ones((7, 30), dtype=torch.float32)
    ei2 = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)  # E=2
    ea2 = torch.ones((2, 16), dtype=torch.float32)
    ga2 = torch.ones((10,), dtype=torch.float32)

    s2 = {
        "x": x2,
        "edge_index": ei2,
        "edge_attr": ea2,
        "graph_attr": ga2,
        "graph_attr_keys": [f"k{i}" for i in range(10)],
        "id": "b",
        "encoder": "residue_knn_graph_ns",
    }

    out = collate_graph([s1, s2])

    # x padded to Fmax=30, total nodes = 5+7=12
    assert out["x"].shape == (12, 30)

    # verify padding on first sample: last 8 dims (22->30) must be zeros
    assert torch.allclose(out["x"][:5, 22:], torch.zeros((5, 8)))

    # edge_attr padded to Femax=16, total edges = 3 + 2 = 5
    assert "edge_attr" in out
    assert out["edge_attr"].shape == (5, 16)

    # first sample edge_attr padding (8->16) must be zeros in tail
    assert torch.allclose(out["edge_attr"][:3, 8:], torch.zeros((3, 8)))

    # graph_attr should be stacked (B,Fg) => (2,10)
    assert "graph_attr" in out
    assert out["graph_attr"].shape == (2, 10)

    # keys should be present and kept (best-effort)
    assert out.get("graph_attr_keys") is not None
    assert len(out["graph_attr_keys"]) == 10

    # batch vector has length total nodes
    assert out["batch"].shape == (12,)
    assert int(out["batch"].min()) == 0
    assert int(out["batch"].max()) == 1


def test_collate_graph_strict_raises_on_mixed_dims():
    torch = pytest.importorskip("torch")
    from morpheo.datasets.collate import collate_graph_strict

    x1 = torch.zeros((3, 4))
    x2 = torch.zeros((3, 5))
    ei1 = torch.zeros((2, 1), dtype=torch.int64)
    ei2 = torch.zeros((2, 1), dtype=torch.int64)

    s1 = {"x": x1, "edge_index": ei1}
    s2 = {"x": x2, "edge_index": ei2}

    with pytest.raises(RuntimeError):
        collate_graph_strict([s1, s2])
