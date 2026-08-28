from __future__ import annotations

import numpy as np
import pytest
from graph_artifact_v2_fixture import save_graph_artifact_v2


def test_export_torch_graph_attr_present(tmp_path):
    torch = pytest.importorskip("torch")
    from eris.export.torch import load_torch

    artifact_dir = tmp_path / "graph-artifact"
    artifact = save_graph_artifact_v2(artifact_dir, graph_attr=(2.0, 2.0))

    s = load_torch(str(artifact_dir), prefer="graph")

    assert artifact.meta["schema_version"] == 2
    assert "graph_attr" in s
    assert isinstance(s["graph_attr"], torch.Tensor)
    assert s["graph_attr"].ndim == 1
    assert int(s["graph_attr"].numel()) > 0
    np.testing.assert_array_equal(s["graph_attr"].numpy(), artifact.data["graph_attr"])
