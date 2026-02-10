from pathlib import Path
import numpy as np

from morpheo.core.types import StructureInput
from morpheo.core.registry import get_encoder

DATA = Path("src/tests/data/1AQ2.pdb")

def test_residue_contact_graph_shapes():
    enc = get_encoder("residue_contact_graph")
    art = enc.encode(StructureInput(source=str(DATA), chain_id=None, model_index=0))

    pos = art.data["pos"]
    edge_index = art.data["edge_index"]
    residue_ids = art.data["residue_ids"]

    assert pos.ndim == 2 and pos.shape[1] == 3
    assert residue_ids.shape[0] == pos.shape[0]
    assert edge_index.ndim == 2 and edge_index.shape[0] == 2
    assert edge_index.dtype == np.int64
    if edge_index.shape[1] > 0:
        assert edge_index.max() < pos.shape[0]
