from __future__ import annotations

import pytest


def test_collate_graph_pads_mixed_x_and_edge_attr():
    torch = pytest.importorskip("torch")
    from eris.datasets.collate import collate_graph

    # sample 1: N=5, x F=22, E=3, edge_attr Fe=8
    x1 = torch.zeros((5, 22), dtype=torch.float32)
    ei1 = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int64)  # E=3
    ea1 = torch.zeros((3, 8), dtype=torch.float32)
    ga1 = torch.zeros((10,), dtype=torch.float32)
    keys = [f"k{i}" for i in range(10)]

    s1 = {
        "x": x1,
        "edge_index": ei1,
        "edge_attr": ea1,
        "graph_attr": ga1,
        "graph_attr_keys": keys,
        "id": "a",
        "encoder": "residue_contact_graph_ns",
    }

    # sample 2: N=7, x F=30, E=2, edge_attr Fe=16
    x2 = torch.ones((7, 30), dtype=torch.float32)
    ei2 = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)  # E=2 (local indexing)
    ea2 = torch.ones((2, 16), dtype=torch.float32)
    ga2 = torch.ones((10,), dtype=torch.float32)

    s2 = {
        "x": x2,
        "edge_index": ei2,
        "edge_attr": ea2,
        "graph_attr": ga2,
        "graph_attr_keys": keys,  # enforce same keys -> stable behavior
        "id": "b",
        "encoder": "residue_knn_graph_ns",
    }

    out = collate_graph([s1, s2])

    # total nodes = 12, pad x to Fmax=30
    assert out["x"].dtype == torch.float32
    assert out["x"].shape == (12, 30)

    # first sample padding (22->30) must be zeros
    assert torch.allclose(out["x"][:5, 22:], torch.zeros((5, 8), dtype=torch.float32))

    # edge_attr padded to Femax=16, total edges = 5
    assert "edge_attr" in out
    assert out["edge_attr"].dtype == torch.float32
    assert out["edge_attr"].shape == (5, 16)

    # first sample edge_attr padding (8->16) must be zeros
    assert torch.allclose(out["edge_attr"][:3, 8:], torch.zeros((3, 8), dtype=torch.float32))

    # graph_attr stacked to (B, Fg)
    assert "graph_attr" in out
    assert out["graph_attr"].dtype == torch.float32
    assert out["graph_attr"].shape == (2, 10)

    # keys preserved when equal across samples
    assert out.get("graph_attr_keys") is not None
    assert list(out["graph_attr_keys"]) == keys

    # batch vector length equals total nodes
    assert out["batch"].shape == (12,)
    assert int(out["batch"].min()) == 0
    assert int(out["batch"].max()) == 1

    # edge_index should be offset for second sample by N1=5
    assert out["edge_index"].dtype == torch.int64
    assert out["edge_index"].shape == (2, 5)

    # First 3 edges: same as ei1
    assert torch.equal(out["edge_index"][:, :3], ei1)

    # Next 2 edges: ei2 + offset(5)
    expected_ei2 = ei2 + 5
    assert torch.equal(out["edge_index"][:, 3:], expected_ei2)


def test_collate_graph_strict_raises_on_mixed_dims():
    torch = pytest.importorskip("torch")
    from eris.datasets.collate import collate_graph_strict

    x1 = torch.zeros((3, 4), dtype=torch.float32)
    x2 = torch.zeros((3, 5), dtype=torch.float32)
    ei1 = torch.zeros((2, 1), dtype=torch.int64)
    ei2 = torch.zeros((2, 1), dtype=torch.int64)

    s1 = {"x": x1, "edge_index": ei1}
    s2 = {"x": x2, "edge_index": ei2}

    with pytest.raises((RuntimeError, ValueError)):
        collate_graph_strict([s1, s2])
