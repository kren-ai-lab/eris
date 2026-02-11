from __future__ import annotations

from pathlib import Path

import pytest


def test_export_torch_graph_attr_present():
    torch = pytest.importorskip("torch")
    from morpheo.export.torch import load_torch

    out_root = Path("out")
    if not out_root.exists():
        pytest.skip("Missing out/ directory. Run morpheo encode + morpheo pack first.")

    # Prefer a known NS encoder, but don't hardcode the protein ID
    candidates = list(out_root.glob("**/residue_contact_graph_ns"))
    if not candidates:
        pytest.skip(
            "No residue_contact_graph_ns artifacts found. "
            "Run: morpheo encode --input src/tests/data/1AQ2.pdb "
            "--encoder residue_contact_graph_ns --out out"
        )

    artifact_dir = candidates[0]

    s = load_torch(str(artifact_dir), prefer="graph")

    assert "graph_attr" in s
    assert isinstance(s["graph_attr"], torch.Tensor)
    assert s["graph_attr"].ndim == 1
    assert int(s["graph_attr"].numel()) > 0

    assert "graph_attr_keys" in s
    # exporter should expose keys as python list[str]
    assert isinstance(s["graph_attr_keys"], list)
    assert len(s["graph_attr_keys"]) == int(s["graph_attr"].numel())
