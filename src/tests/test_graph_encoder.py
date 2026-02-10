from pathlib import Path
import numpy as np

from morpheo.core.registry import get_encoder
from morpheo.core.types import StructureInput

DATA = Path("src/tests/data/1AQ2.pdb")

def _check_artifact(art):
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
        assert edge_index.max() < pos.shape[0]

def test_contact_graph():
    enc = get_encoder("residue_contact_graph")
    art = enc.encode(StructureInput(source=str(DATA)))
    _check_artifact(art)

def test_knn_graph():
    enc = get_encoder("residue_knn_graph")
    art = enc.encode(StructureInput(source=str(DATA)))
    _check_artifact(art)
