from __future__ import annotations
from pathlib import Path
import pytest

def test_export_torch_graph_attr_present():
    torch = pytest.importorskip("torch")
    from morpheo.export.torch import load_torch

    artifact_dir = Path("out/1AQ2/residue_contact_graph_ns")
    if not artifact_dir.exists():
        pytest.skip("Run: morpheo encode --input src/tests/data/1AQ2.pdb --encoder residue_contact_graph_ns --out out")

    s = load_torch(str(artifact_dir), prefer="graph")
    assert "graph_attr" in s
    assert s["graph_attr"].ndim == 1
    assert int(s["graph_attr"].numel()) > 0
    assert "graph_attr_keys" in s
    assert isinstance(s["graph_attr_keys"], list)
