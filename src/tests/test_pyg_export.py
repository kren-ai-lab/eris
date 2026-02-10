from pathlib import Path

import pytest

from morpheo.core.registry import get_encoder
from morpheo.core.types import StructureInput

DATA = Path("src/tests/data/1AQ2.pdb")

def test_pyg_export_optional():
    try:
        import torch  # noqa
        import torch_geometric  # noqa
    except Exception:
        pytest.skip("torch/torch_geometric not installed")

    from morpheo.export.pyg import to_pyg

    enc = get_encoder("residue_contact_graph")
    art = enc.encode(StructureInput(source=str(DATA)))

    g = to_pyg(art, x_mode="onehot+plddt")
    assert g.edge_index.shape[0] == 2
    assert g.pos.shape[1] == 3
    assert g.x.shape[0] == g.pos.shape[0]
