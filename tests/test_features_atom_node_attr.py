from __future__ import annotations

import numpy as np
import pytest

from eris.features.node.atom import build_node_attr


def _atom_inputs(n=2):
    return {
        "elem_index": np.arange(n, dtype=np.int16),
        "atom_role": np.arange(n, dtype=np.int8),
        "bfactor": np.full((n,), 20.0, dtype=np.float32),
        "occ": np.ones((n,), dtype=np.float32),
    }


def test_atom_node_builder_default_smoke():
    node_attr, keys = build_node_attr(**_atom_inputs(), return_keys=True)

    assert node_attr.shape == (2, 25)
    assert node_attr.dtype == np.float32
    assert len(keys) == node_attr.shape[1]
    assert keys[0] == "elem.0"
    assert keys[-1] == "occ"


def test_atom_node_builder_rejects_selected_array_row_mismatch():
    inputs = _atom_inputs()
    inputs["atom_role"] = np.array([0], dtype=np.int8)

    with pytest.raises(ValueError, match="expected 2, got 1"):
        build_node_attr(**inputs, mode="role")


def test_atom_node_builder_valid_mode_with_no_atoms_preserves_empty_contract():
    node_attr, keys = build_node_attr(**_atom_inputs(0), mode="elem", return_keys=True)

    assert node_attr.shape == (0, 0)
    assert node_attr.dtype == np.float32
    assert keys == []


@pytest.mark.parametrize(
    "mode",
    [
        "unknown",
        "elem+elem",
        [{"elem": {"bogus": 1}}],
        ("elem",),
    ],
)
def test_atom_node_builder_validates_mode_with_no_atoms(mode):
    with pytest.raises(ValueError):
        build_node_attr(**_atom_inputs(0), mode=mode)
