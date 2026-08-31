from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import eris
from eris.export.pyg import PyGExportError, load_pyg, to_pyg
from tests.test_m7_pyg_export import _torch_and_data_type

PDB = Path("tests/data/1AQ2.pdb")


def test_export_pyg_graph_v2_auto(tmp_path: Path, monkeypatch) -> None:
    """Historical PyG smoke test migrated from legacy out/Schema v1 to Schema v2."""

    _, Data = _torch_and_data_type(monkeypatch)
    artifact_dir = tmp_path / "artifact"
    artifact = eris.encode(
        PDB,
        representation="residue_contact_graph",
        output=artifact_dir,
        use_cache=False,
    )

    graph = load_pyg(artifact_dir)

    assert isinstance(graph, Data)
    assert graph.representation_name == "residue_contact_graph"
    assert graph.representation_family == "explicit"
    assert graph.representation_type == "graph"
    np.testing.assert_array_equal(graph.x.cpu().numpy(), artifact.data["node_attr"])
    np.testing.assert_array_equal(graph.edge_index.cpu().numpy(), artifact.data["edge_index"])
    np.testing.assert_array_equal(graph.edge_attr.cpu().numpy(), artifact.data["edge_attr"])
    np.testing.assert_array_equal(graph.edge_dist.cpu().numpy(), artifact.data["edge_dist"])
    np.testing.assert_array_equal(graph.graph_attr.cpu().numpy(), artifact.data["graph_attr"])
    assert graph.edge_attr.shape[0] == graph.edge_index.shape[1]
    assert graph.edge_dist.shape[0] == graph.edge_index.shape[1]


def test_export_pyg_cloud_is_unsupported() -> None:
    """Point clouds have coordinates but no public graph topology."""

    artifact = eris.encode(PDB, representation="residue_point_cloud", use_cache=False)

    with pytest.raises(PyGExportError, match="Unsupported representation_type='cloud'"):
        to_pyg(artifact)
