from __future__ import annotations

from pathlib import Path

import pytest


def _get_data(obj):
    """
    ErisFolderDataset may return either:
      - a StructureArtifact (obj.data)
      - a dict sample already (obj itself)
    """
    if hasattr(obj, "data") and isinstance(getattr(obj, "data"), dict):
        return obj.data
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unexpected dataset item type: {type(obj)}")


def test_collate_graph_includes_graph_attr():
    pytest.importorskip("torch")
    from torch.utils.data import DataLoader, Subset

    from eris.datasets import ErisFolderDataset, collate_graph

    idx = Path("out/index.csv")
    if not idx.exists():
        pytest.skip("Missing out/index.csv (run eris pack --root out --out out/index.csv)")

    ds = ErisFolderDataset(str(idx), prefer="graph")

    # Find at least 2 samples that include graph_attr
    picks = []
    for i in range(len(ds)):
        item = ds[i]
        data = _get_data(item)

        # Depending on your dataset, graph_attr may be already tensor or np array,
        # but key presence is what matters for this test.
        if "graph_attr" in data:
            picks.append(i)
            if len(picks) == 2:
                break

    if len(picks) < 2:
        pytest.skip("Need at least 2 graph samples with graph_attr in out/index.csv")

    subset = Subset(ds, picks)

    dl = DataLoader(subset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_graph)
    b = next(iter(dl))

    assert "graph_attr" in b
    assert b["graph_attr"].ndim == 2
    assert b["graph_attr"].shape[0] == 2
