from __future__ import annotations

from pathlib import Path

from eris.core.registry import get_encoder, get_encoder_info
from eris.core.types import StructureInput

DATA = Path("tests/data/1AQ2.pdb")

def test_registry_matches_meta():
    enc = get_encoder("residue_contact_graph")
    info = get_encoder_info("residue_contact_graph")

    art = enc.encode(StructureInput(source=str(DATA)))
    art.finalize()

    assert art.meta["type"] == info.type
    assert art.meta["level"] == info.level
    assert art.meta["encoder_fingerprint"] == enc.fingerprint()
