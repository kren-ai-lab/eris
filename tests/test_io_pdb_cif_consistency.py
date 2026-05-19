from pathlib import Path
import numpy as np
from eris.core.io import extract_ca_bundle, load_structure, select_chain

DATA = Path(__file__).resolve().parent / "data"

def _load_ca(path: Path):
    st = load_structure(path)
    ch = select_chain(st, "A", 0)
    return extract_ca_bundle(ch, amino_acids_only=True, altloc="max_occupancy", missing_ca="skip")

def test_pdb_cif_ca_bundle_comparable():
    strc_pdb = _load_ca(DATA/"1AQ2.pdb")
    strc_cif = _load_ca(DATA/"1AQ2.cif")

    assert np.array_equal(strc_pdb["residue_ids"], strc_cif["residue_ids"])
    assert np.allclose(strc_pdb["pos"], strc_cif["pos"], atol=1e-3, rtol=1e-5)
    assert np.allclose(strc_pdb["bfactor_ca"], strc_cif["bfactor_ca"], atol=1e-3, rtol=1e-5)
