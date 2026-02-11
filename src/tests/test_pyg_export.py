# src/tests/test_export_pyg.py

from __future__ import annotations

from pathlib import Path

import pytest

@pytest.mark.skipif(
    pytest.importorskip("torch", reason="torch not installed") is None,
    reason="torch not installed",
)
def test_export_pyg_graph_ns_auto(tmp_path: Path):
    """
    Ensures PyG exporter prefers node_attr/edge_attr when present (auto mode).
    """
    torch = pytest.importorskip("torch")
    pyg = pytest.importorskip("torch_geometric")

    from morpheo.export.pyg import load_pyg

    artifact_dir = Path("out/1AQ2/residue_contact_graph_ns")
    if not artifact_dir.exists():
        pytest.skip("Artifact not found. Run: morpheo encode --input src/tests/data/1AQ2.pdb --encoder residue_contact_graph_ns --out out")

    g = load_pyg(str(artifact_dir), x_mode="auto")
    assert hasattr(g, "x")
    assert hasattr(g, "edge_index")
    assert g.x.ndim == 2
    assert g.edge_index.shape[0] == 2
    assert g.x.shape[0] >= 10

    # edge_attr must exist for ns graphs
    assert getattr(g, "edge_attr", None) is not None
    assert g.edge_attr.ndim == 2
    assert g.edge_attr.shape[0] == g.edge_index.shape[1]


@pytest.mark.skipif(
    pytest.importorskip("torch", reason="torch not installed") is None,
    reason="torch not installed",
)
def test_export_pyg_graph_fallback_rbf(tmp_path: Path):
    """
    Ensures PyG exporter falls back to RBF(edge_dist) when edge_attr is missing.
    """
    torch = pytest.importorskip("torch")
    pyg = pytest.importorskip("torch_geometric")

    from morpheo.export.pyg import load_pyg

    artifact_dir = Path("out/1AQ2/residue_contact_graph")
    if not artifact_dir.exists():
        pytest.skip("Artifact not found. Run: morpheo encode --input src/tests/data/1AQ2.pdb --encoder residue_contact_graph --out out")

    g = load_pyg(str(artifact_dir), x_mode="auto", rbf_k=8, rbf_dmax=20.0)
    assert getattr(g, "edge_attr", None) is not None
    assert g.edge_attr.ndim == 2
    assert g.edge_attr.shape[0] == g.edge_index.shape[1]
    assert g.edge_attr.shape[1] == 8  # rbf_k


@pytest.mark.skipif(
    pytest.importorskip("torch", reason="torch not installed") is None,
    reason="torch not installed",
)
def test_export_pyg_missing_edge_index_raises(tmp_path: Path):
    """
    A non-graph artifact should raise on PyG export.
    """
    torch = pytest.importorskip("torch")
    pyg = pytest.importorskip("torch_geometric")

    from morpheo.core.types import StructureArtifact
    from morpheo.export.pyg import to_pyg

    art = StructureArtifact("dummy", data={"pos": [[0.0, 0.0, 0.0]]}, meta={})

    with pytest.raises(KeyError):
        to_pyg(art, x_mode="auto")
