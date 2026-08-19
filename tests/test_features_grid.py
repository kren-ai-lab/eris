from __future__ import annotations

import numpy as np
import pytest

from eris.features.feature_builders import build_from_registry_voxel
from eris.features.feature_registry import VoxelFeatureSpec
from eris.features.grid.grid import build_grid_attr


def _grid_inputs():
    return {
        "pos": np.zeros((2, 3), dtype=np.float32),
        "centroid": np.zeros((1, 3), dtype=np.float32),
        "strc": {
            "aa_index": np.array([0, 0], dtype=np.int16),
            "plddt": np.array([50.0, 50.0], dtype=np.float32),
            "bfactor_ca": np.array([20.0, 20.0], dtype=np.float32),
        },
        "grid_size": 4,
        "voxel_size": 1.0,
        "fit": False,
        "margin": 0.0,
        "center": False,
        "pool": "max",
        "sigma": 1.0,
    }


@pytest.mark.parametrize(
    ("return_keys", "expected_keys"),
    [(True, ["occupancy", "density"]), (False, [])],
)
def test_grid_builder_preserves_shape_for_both_key_modes(return_keys, expected_keys):
    grid, keys, metadata = build_grid_attr(
        **_grid_inputs(),
        channels="occupancy+density",
        return_keys=return_keys,
    )

    assert grid.shape == (2, 4, 4, 4)
    assert grid.dtype == np.float32
    assert keys == expected_keys
    assert metadata["origin"].shape == (3,)


def test_grid_builder_propagates_normalized_pool_mode():
    canonical, _, _ = build_grid_attr(
        **_grid_inputs(),
        channels="occupancy",
        return_keys=True,
    )
    spaced_inputs = _grid_inputs()
    spaced_inputs["pool"] = " MAX "
    spaced, _, _ = build_grid_attr(
        **spaced_inputs,
        channels="occupancy",
        return_keys=True,
    )

    np.testing.assert_array_equal(spaced, canonical)
    assert float(spaced.max()) == 1.0


def test_grid_builder_rejects_aa_onehot_channel_key_mismatch():
    with pytest.raises(ValueError, match="aa_onehot.*channel/key mismatch"):
        build_grid_attr(
            **_grid_inputs(),
            channels=[{"aa_onehot": {"num_classes": 10}}],
            return_keys=True,
        )


def test_voxel_builder_rejects_emitted_channel_mismatch():
    spec = VoxelFeatureSpec(
        token="bad",
        order=10,
        requires={"pool"},
        n_channels_fn=lambda data, ctx, params: 1,
        fill_fn=lambda **kwargs: kwargs["ch_start"] + 2,
        keys_fn=lambda data, ctx, params: ["bad"],
    )

    with pytest.raises(ValueError, match="declared 1 but emitted 2"):
        build_from_registry_voxel(
            data={},
            mode="bad",
            registry={"bad": spec},
            grid_size=2,
            pool="max",
        )


@pytest.mark.parametrize(
    ("channels", "message"),
    [
        ("unknown", "Unknown voxel tokens"),
        ("occupancy+occupancy", "Duplicate voxel tokens"),
    ],
)
def test_grid_builder_rejects_invalid_channel_modes(channels, message):
    with pytest.raises(ValueError, match=message):
        build_grid_attr(
            **_grid_inputs(),
            channels=channels,
            return_keys=True,
        )
