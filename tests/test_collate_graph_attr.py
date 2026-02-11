from __future__ import annotations
from pathlib import Path
import pytest

def test_collate_graph_includes_graph_attr():
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader
    from morpheo.datasets import MorpheoFolderDataset, collate_graph

    # Requires packed csv already; skip if missing
    idx = Path("out/index.csv")
    if not idx.exists():
        pytest.skip("Missing out/index.csv (run morpheo pack --root out --out out/index.csv)")

    ds = MorpheoFolderDataset(str(idx), prefer="graph")
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_graph)
    b = next(iter(dl))

    assert "graph_attr" in b
    assert b["graph_attr"].ndim == 2
    assert b["graph_attr"].shape[0] == 2
