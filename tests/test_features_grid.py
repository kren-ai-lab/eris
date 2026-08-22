from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eris.core.artifact_schema_v2 import (
    build_artifact_metadata,
    build_feature_keys,
    build_input_metadata,
    build_provenance,
)
from eris.core.artifacts import ArtifactValidationError, StructureArtifact
from eris.core.types import StructureInput
from eris.encoders.grids.voxel_grid_ca import VoxelGridCA
from eris.features.feature_builders import build_from_registry_voxel
from eris.features.feature_registry import VoxelFeatureSpec
from eris.features.grid.grid import build_grid_attr

DATA = Path("tests/data/1AQ2.pdb")
AA_ONEHOT_CHANNELS = [f"aa_onehot.{aa}" for aa in "ACDEFGHIKLMNPQRSTVWYX"]


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


def _encode_grid(**kwargs) -> StructureArtifact:
    encoder = VoxelGridCA(use_cache=False, **kwargs)
    return encoder.encode(StructureInput(source=str(DATA)))


def _grid_artifact(
    grid: np.ndarray,
    channels: list[str],
) -> StructureArtifact:
    parameters: dict = {}
    return StructureArtifact(
        data={"grid": grid},
        meta=build_artifact_metadata(
            data={"grid": grid},
            representation_name="synthetic_voxel_grid",
            representation_family="explicit",
            representation_type="grid",
            granularity="residue",
            coordinate_source="ca",
            encoder={"implementation": "tests.synthetic.VoxelEncoder", "version": "0.1.0"},
            input_metadata=build_input_metadata(
                source="memory://synthetic",
                source_sha256="a" * 64,
                input_format="pdb",
                model_index=0,
                chain_id=None,
            ),
            parameters=parameters,
            provenance=build_provenance(
                representation_name="synthetic_voxel_grid",
                parameters=parameters,
            ),
            output_declarations=[{"name": "grid", "role": "grid", "layout": "CXYZ"}],
            feature_keys=build_feature_keys(grid_channels=channels),
            counts={},
        ),
    )


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


def test_default_voxel_encoder_metadata_matches_physical_grid():
    artifact = _encode_grid()
    expected_physical_channels = [
        "occupancy",
        "density",
        "plddt",
        "bfactor",
        *AA_ONEHOT_CHANNELS,
    ]

    artifact.finalize()
    artifact.validate(strict=True)

    assert artifact.data["grid"].shape == (25, 64, 64, 64)
    assert artifact.meta["feature_keys"] == {
        "grid_channels": expected_physical_channels,
    }
    assert artifact.meta["outputs"][0]["name"] == "grid"
    assert artifact.meta["outputs"][0]["layout"] == "CXYZ"
    assert artifact.meta["parameters"]["channels"] == [
        "occupancy",
        "density",
        "aa_onehot",
        "plddt",
        "bfactor",
    ]


def test_scalar_occupancy_metadata_uses_one_token_and_one_channel():
    artifact = _encode_grid(channels="occupancy", grid_size=4)

    assert artifact.data["grid"].shape[0] == 1
    assert artifact.meta["feature_keys"]["grid_channels"] == ["occupancy"]
    assert artifact.meta["parameters"]["channels"] == ["occupancy"]


def test_aa_onehot_metadata_has_exact_physical_channel_order():
    artifact = _encode_grid(channels="aa_onehot", grid_size=4)

    assert artifact.data["grid"].shape[0] == 21
    assert artifact.meta["feature_keys"]["grid_channels"] == AA_ONEHOT_CHANNELS
    assert artifact.meta["parameters"]["channels"] == ["aa_onehot"]


def test_aa_onehot_a_and_x_channels_follow_declared_positions():
    inputs = _grid_inputs()
    inputs["pos"] = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    inputs["strc"]["aa_index"] = np.array([0, 20], dtype=np.int16)

    grid, keys, _ = build_grid_attr(
        **inputs,
        channels="aa_onehot",
        return_keys=True,
    )

    assert keys[0] == "aa_onehot.A"
    assert keys[20] == "aa_onehot.X"
    assert np.count_nonzero(grid[0]) == 1
    assert np.count_nonzero(grid[20]) == 1


def test_unordered_request_preserves_requested_tokens_and_physical_order():
    artifact = _encode_grid(
        channels="aa_onehot+occupancy+bfactor",
        grid_size=4,
    )

    assert artifact.data["grid"].shape[0] == 23
    assert artifact.meta["parameters"]["channels"] == [
        "aa_onehot",
        "occupancy",
        "bfactor",
    ]
    assert artifact.meta["feature_keys"]["grid_channels"] == [
        "occupancy",
        "bfactor",
        *AA_ONEHOT_CHANNELS,
    ]


def test_parameterized_channel_metadata_preserves_token_parameters():
    requested = [{"density": {"sigma": 2.0}}, "occupancy"]
    artifact = _encode_grid(channels=requested, grid_size=4)

    assert artifact.meta["parameters"]["channels"] == requested
    assert artifact.meta["feature_keys"]["grid_channels"] == [
        "occupancy",
        "density",
    ]


@pytest.mark.parametrize("pool", ["max", "sum", "mean"])
def test_pool_mode_does_not_change_channel_metadata(pool):
    artifact = _encode_grid(
        channels="aa_onehot+occupancy",
        grid_size=4,
        pool=pool,
    )

    assert artifact.data["grid"].shape[0] == 22
    assert artifact.meta["feature_keys"] == {
        "grid_channels": ["occupancy", *AA_ONEHOT_CHANNELS],
    }


def test_all_zero_grid_preserves_shape_channels_and_layout():
    inputs = _grid_inputs()
    inputs["pos"] = np.full((2, 3), 100.0, dtype=np.float32)
    grid, channels, _ = build_grid_attr(
        **inputs,
        channels="occupancy+bfactor",
        return_keys=True,
    )
    artifact = _grid_artifact(grid, channels)

    artifact.finalize()
    artifact.validate(strict=True)

    assert not np.any(grid)
    assert grid.shape == (2, 4, 4, 4)
    assert artifact.meta["feature_keys"] == {
        "grid_channels": ["occupancy", "bfactor"],
    }
    assert artifact.meta["outputs"][0]["layout"] == "CXYZ"


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_dtype_does_not_change_channel_metadata(dtype):
    artifact = _encode_grid(
        channels="occupancy+bfactor",
        grid_size=4,
        dtype=dtype,
    )

    assert artifact.data["grid"].dtype == np.dtype(dtype)
    assert artifact.meta["feature_keys"]["grid_channels"] == [
        "occupancy",
        "bfactor",
    ]


def test_valid_schema_v2_grid_artifact_finalizes_and_validates():
    artifact = _grid_artifact(
        np.zeros((3, 2, 2, 2), dtype=np.float32),
        ["channel.0", "channel.1", "channel.2"],
    )

    artifact.finalize()
    artifact.validate(strict=True)


def test_schema_v2_grid_rejects_truncated_channel_metadata():
    artifact = _grid_artifact(
        np.zeros((3, 2, 2, 2), dtype=np.float32),
        ["channel.0", "channel.1", "channel.2"],
    )
    artifact.finalize()
    artifact.meta["feature_keys"]["grid_channels"] = ["channel.0", "channel.1"]

    with pytest.raises(ArtifactValidationError, match=r"2 entries.*3 features"):
        artifact.validate(strict=True)


def test_schema_v2_grid_records_layout_in_outputs():
    artifact = _grid_artifact(
        np.zeros((1, 2, 2, 2), dtype=np.float32),
        ["channel.0"],
    )
    artifact.validate(strict=True)

    assert artifact.meta["feature_keys"] == {"grid_channels": ["channel.0"]}
    assert artifact.meta["outputs"][0]["layout"] == "CXYZ"


def test_schema_v2_grid_rejects_non_4d_tensor():
    artifact = _grid_artifact(
        np.zeros((1, 2, 2, 2), dtype=np.float32),
        ["channel.0"],
    )
    artifact.finalize()
    artifact.data["grid"] = np.zeros((2, 2, 2), dtype=np.float32)

    with pytest.raises(ArtifactValidationError, match="meta.shapes.grid"):
        artifact.validate(strict=True)


def test_voxel_artifact_roundtrip_preserves_channel_contract(tmp_path):
    artifact = _encode_grid(
        channels="aa_onehot+occupancy+bfactor",
        grid_size=4,
    )
    artifact_dir = tmp_path / "voxel-artifact"

    artifact.save(artifact_dir)
    loaded = StructureArtifact.load(artifact_dir, validate=True, strict=True)

    assert loaded.meta["feature_keys"] == artifact.meta["feature_keys"]
    assert loaded.meta["shapes"] == artifact.meta["shapes"]
    assert loaded.meta["dtypes"] == artifact.meta["dtypes"]
    assert loaded.meta["counts"] == {}
    assert loaded.meta["outputs"][0]["layout"] == "CXYZ"
