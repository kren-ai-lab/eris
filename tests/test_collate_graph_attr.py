from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from graph_artifact_v2_fixture import save_graph_artifact_v2


def test_collate_graph_includes_graph_attr(tmp_path):
    pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from eris.datasets import ErisFolderDataset, collate_graph

    rows = []
    expected = []
    for sample_id, graph_attr in (("sample-a", (2.0, 2.0)), ("sample-b", (3.0, 4.0))):
        artifact_dir = tmp_path / sample_id / "graph"
        artifact = save_graph_artifact_v2(artifact_dir, graph_attr=graph_attr)
        assert artifact.meta["schema_version"] == 2
        expected.append(artifact.data["graph_attr"])
        rows.append(
            {
                "id": sample_id,
                "encoder": "test_graph_fixture",
                "artifact_dir": str(artifact_dir),
                "meta_path": str(artifact_dir / "meta.json"),
            }
        )

    index = Path(tmp_path) / "index.csv"
    with index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    dataset = ErisFolderDataset(index, prefer="graph")
    dl = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_graph,
    )
    b = next(iter(dl))

    assert "graph_attr" in b
    assert b["graph_attr"].ndim == 2
    assert b["graph_attr"].shape[0] == 2
    np.testing.assert_array_equal(b["graph_attr"].numpy(), np.stack(expected))
