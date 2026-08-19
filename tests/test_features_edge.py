from __future__ import annotations

import numpy as np
import pytest

from eris.features.edge.edge import build_edge_attr


EDGE_INDEX = np.array([[0, 0, 1], [1, 2, 2]], dtype=np.int64)
EDGE_DIST = np.array([1.0, 2.0, 1.5], dtype=np.float32)
SEQ_POS = np.array([0, 2, 5], dtype=np.int64)


def test_edge_builder_default_mode_uses_feature_context():
    edge_attr, keys = build_edge_attr(
        EDGE_INDEX,
        EDGE_DIST,
        seq_pos=SEQ_POS,
        rbf_k=4,
        local_cutoff=2,
        return_keys=True,
    )

    assert edge_attr.shape == (3, 7)
    assert edge_attr.dtype == np.float32
    assert keys == [
        "dist",
        "rbf.0",
        "rbf.1",
        "rbf.2",
        "rbf.3",
        "seq_sep",
        "flag.is_local_le_2",
    ]


@pytest.mark.parametrize(
    ("mode", "expected_keys"),
    [
        ("dist", ["dist"]),
        ("rbf", ["rbf.0", "rbf.1", "rbf.2", "rbf.3"]),
        ("seq", ["seq_sep"]),
        ("local", ["flag.is_local_le_2"]),
    ],
)
def test_edge_builder_single_modes(mode, expected_keys):
    edge_attr, keys = build_edge_attr(
        EDGE_INDEX,
        EDGE_DIST,
        mode=mode,
        seq_pos=SEQ_POS,
        rbf_k=4,
        local_cutoff=2,
        return_keys=True,
    )

    assert edge_attr.shape == (3, len(expected_keys))
    assert edge_attr.dtype == np.float32
    assert keys == expected_keys


def test_edge_builder_seq_and_local_values():
    edge_attr, keys = build_edge_attr(
        EDGE_INDEX,
        EDGE_DIST,
        mode="seq+local",
        seq_pos=SEQ_POS,
        local_cutoff=2,
        return_keys=True,
    )

    assert keys == ["seq_sep", "flag.is_local_le_2"]
    np.testing.assert_array_equal(
        edge_attr,
        np.array([[2.0, 1.0], [5.0, 0.0], [3.0, 0.0]], dtype=np.float32),
    )


def test_edge_builder_empty_mode_uses_edge_count():
    edge_attr, keys = build_edge_attr(
        EDGE_INDEX,
        EDGE_DIST,
        mode="",
        return_keys=True,
    )

    assert edge_attr.shape == (3, 0)
    assert edge_attr.dtype == np.float32
    assert keys == []


def test_edge_builder_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown feature tokens.*unknown"):
        build_edge_attr(EDGE_INDEX, EDGE_DIST, mode="unknown")
