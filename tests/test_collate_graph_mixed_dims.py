from __future__ import annotations

import pytest


def _metadata(sample_id: str, name: str = "residue_knn_graph"):
    return {
        "id": sample_id,
        "artifact_dir": f"artifacts/{sample_id}/{name}",
        "meta_path": f"artifacts/{sample_id}/{name}/meta.json",
        "representation_name": name,
        "representation_family": "explicit",
        "representation_type": "graph",
        "granularity": "residue",
        "parameters": {"k": 8},
        "feature_keys": {
            "node_attr_keys": ["feature"],
            "edge_attr_keys": ["edge_feature"],
            "graph_attr_keys": ["graph_feature"],
        },
        "outputs": [{"name": "node_attr"}, {"name": "edge_index"}],
    }


def test_collate_graph_rejects_mixed_x_and_edge_attr_dimensions():
    torch = pytest.importorskip("torch")
    from eris.datasets.collate import collate_graph

    # sample 1: N=5, x F=22, E=3, edge_attr Fe=8
    x1 = torch.zeros((5, 22), dtype=torch.float32)
    ei1 = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int64)  # E=3
    ea1 = torch.zeros((3, 8), dtype=torch.float32)
    ga1 = torch.zeros((10,), dtype=torch.float32)

    s1 = {
        **_metadata("a"),
        "x": x1,
        "edge_index": ei1,
        "edge_attr": ea1,
        "graph_attr": ga1,
    }

    # sample 2: N=7, x F=30, E=2, edge_attr Fe=16
    x2 = torch.ones((7, 30), dtype=torch.float32)
    ei2 = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)  # E=2 (local indexing)
    ea2 = torch.ones((2, 16), dtype=torch.float32)
    ga2 = torch.ones((10,), dtype=torch.float32)

    s2 = {
        **_metadata("b"),
        "x": x2,
        "edge_index": ei2,
        "edge_attr": ea2,
        "graph_attr": ga2,
    }

    with pytest.raises(ValueError, match="feature width mismatch"):
        collate_graph([s1, s2])


def test_collate_graph_strict_raises_on_mixed_dims():
    torch = pytest.importorskip("torch")
    from eris.datasets.collate import collate_graph_strict

    x1 = torch.zeros((3, 4), dtype=torch.float32)
    x2 = torch.zeros((3, 5), dtype=torch.float32)
    ei1 = torch.zeros((2, 1), dtype=torch.int64)
    ei2 = torch.zeros((2, 1), dtype=torch.int64)

    s1 = {**_metadata("a"), "x": x1, "edge_index": ei1}
    s2 = {**_metadata("b"), "x": x2, "edge_index": ei2}

    with pytest.raises((RuntimeError, ValueError)):
        collate_graph_strict([s1, s2])
