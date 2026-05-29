# tests/test_export_pyg.py

from __future__ import annotations

from pathlib import Path

import pytest


def _find_artifact_dir(pattern: str) -> Path | None:
    out_root = Path("out")
    if not out_root.exists():
        return None
    hits = list(out_root.glob(pattern))
    return hits[0] if hits else None


def test_export_pyg_graph_ns_auto():
    """
    Ensures PyG exporter prefers node_attr/edge_attr when present (auto mode).
    """
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    from eris.export.pyg import load_pyg

    artifact_dir = _find_artifact_dir("**/residue_contact_graph_ns")
    if artifact_dir is None:
        pytest.skip(
            "Artifact not found. Run: eris encode --input tests/data/1AQ2.pdb "
            "--encoder residue_contact_graph_ns --out out"
        )

    g = load_pyg(str(artifact_dir))

    assert hasattr(g, "x")
    assert hasattr(g, "edge_index")
    assert g.x.ndim == 2
    assert g.edge_index.shape[0] == 2
    assert g.x.shape[0] >= 10

    # edge_attr must exist for ns graphs
    assert getattr(g, "edge_attr", None) is not None
    assert g.edge_attr.ndim == 2
    assert g.edge_attr.shape[0] == g.edge_index.shape[1]

    assert hasattr(g, "graph_attr")
    assert g.graph_attr.ndim == 1
    assert int(g.graph_attr.numel()) > 0


# def test_export_pyg_graph_fallback_rbf():
#     """
#     Ensures PyG exporter falls back to RBF(edge_dist) when edge_attr is missing.
#     """
#     pytest.importorskip("torch")
#     pytest.importorskip("torch_geometric")

#     from eris.export.pyg import load_pyg

#     artifact_dir = _find_artifact_dir("**/residue_contact_graph")
#     if artifact_dir is None:
#         pytest.skip(
#             "Artifact not found. Run: eris encode --input tests/data/1AQ2.pdb "
#             "--encoder residue_contact_graph --out out"
#         )

#     g = load_pyg(str(artifact_dir))

#     assert getattr(g, "edge_attr", None) is not None
#     assert g.edge_attr.ndim == 2
#     assert g.edge_attr.shape[0] == g.edge_index.shape[1]
#     assert g.edge_attr.shape[1] == 8  # rbf_k



def test_export_pyg_cloud_allowed():
    """
    A cloud artifact should be exportable to PyG (pos-only graph).
    """
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")

    from eris.core.artifacts import StructureArtifact
    from eris.export.pyg import to_pyg

    art = StructureArtifact(
        data={"pos": [[0.0, 0.0, 0.0]]},
        meta={
            "encoder_name": "dummy",
            "encoder_version": "0.0.0",
            "encoder_fingerprint": "test",
            "type": "cloud",     # explicitly not a graph
            "level": "ca",
            "params": {},
            "input": {"source": "memory", "chain_id": None, "model_index": 0, "format": None},
            "feature_keys": {},
        },
    )
    art.finalize()

    g = to_pyg(art)

    assert hasattr(g, "pos")
    assert g.pos.shape[0] == 1

