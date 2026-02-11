from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from morpheo.core.registry import get_encoder
from morpheo.core.types import StructureInput

DATA = Path("tests/data/1AQ2.pdb")


def _check_artifact_graph_basic(art):
    # Ensure artifact is finalized for contract fields
    if hasattr(art, "finalize"):
        art.finalize()

    # ---- data checks ----
    pos = art.data["pos"]
    edge_index = art.data["edge_index"]
    edge_dist = art.data["edge_dist"]
    residue_ids = art.data["residue_ids"]

    assert pos.ndim == 2 and pos.shape[1] == 3
    assert residue_ids.shape[0] == pos.shape[0]

    assert edge_index.ndim == 2 and edge_index.shape[0] == 2
    assert edge_index.dtype == np.int64

    assert edge_dist.ndim == 1
    assert edge_dist.shape[0] == edge_index.shape[1]

    if edge_index.shape[1] > 0:
        assert int(edge_index.max()) < int(pos.shape[0])

    # ---- meta contract (minimal) ----
    meta = art.meta
    for k in ["schema_version", "encoder_name", "encoder_version", "encoder_fingerprint",
              "type", "level", "family", "params", "input", "dtypes", "counts", "feature_keys"]:
        assert k in meta, f"missing meta key: {k}"

    assert meta["type"] == "graph"
    assert meta["level"] in {"ca", "residue"}  # depends on your convention; you’re using "ca"
    assert meta["family"] == "residue_graph"

    assert isinstance(meta["params"], dict)
    assert isinstance(meta["input"], dict)
    assert "source" in meta["input"]

    # counts must be consistent
    counts = meta["counts"]
    assert int(counts.get("n_nodes", pos.shape[0])) == int(pos.shape[0])
    assert int(counts.get("n_edges", edge_index.shape[1])) == int(edge_index.shape[1])


def test_contact_graph():
    if not DATA.exists():
        pytest.skip(f"Missing test structure: {DATA}")

    enc = get_encoder("residue_contact_graph")
    art = enc.encode(StructureInput(source=str(DATA)))
    _check_artifact_graph_basic(art)


def test_knn_graph():
    if not DATA.exists():
        pytest.skip(f"Missing test structure: {DATA}")

    enc = get_encoder("residue_knn_graph")
    art = enc.encode(StructureInput(source=str(DATA)))
    _check_artifact_graph_basic(art)
