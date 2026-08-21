from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import eris.encoders.grids.voxel_grid_ca as voxel_module
from eris.core.artifacts import StructureArtifact
from eris.core.base_encoder.base_encoder import BaseEncoder, EncodingContext
from eris.core.config import ResiduePointCloudConfig, ResidueVoxelGridConfig
from eris.core.io import load_and_extract_residue
from eris.core.registry import get_encoder
from eris.core.types import StructureInput
from eris.encoders.grids.voxel_grid_ca import VoxelGridCA

DATA = Path("tests/data/1AQ2.pdb")
INPUT = StructureInput(source=str(DATA))
DEFAULT_GRID_HASH = "da786c06e2f353fc5c51844832c283d8be62916f1d5438d8dfdcae6145c05068"
CENTER_FALSE_GRID_HASH = "a1c4dbc89401639d9eaf8df78e6daa14fd0e34a0b6581836684d11c6ea4c69a5"
AA_ONEHOT_CHANNELS = [f"aa_onehot.{aa}" for aa in "ACDEFGHIKLMNPQRSTVWYX"]


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def _encode(encoder: VoxelGridCA) -> StructureArtifact:
    return encoder.encode(INPUT)


def _structure(pos: np.ndarray) -> dict[str, np.ndarray]:
    n = len(pos)
    return {
        "pos": np.asarray(pos, dtype=np.float32),
        "residue_ids": np.asarray([f"A:{index + 1}" for index in range(n)]),
        "resname": np.asarray(["ALA"] * n),
        "aa_index": np.zeros(n, dtype=np.int16),
        "plddt": np.full(n, 50.0, dtype=np.float32),
        "bfactor_ca": np.full(n, 20.0, dtype=np.float32),
    }


def _context(encoder: VoxelGridCA, pos: np.ndarray) -> EncodingContext:
    structure = _structure(pos)
    centroid, normalized = encoder._normalize_structure(structure["pos"])
    return EncodingContext(
        inp=StructureInput(source="memory.pdb"),
        raw_structure=structure,
        centroid=centroid,
        pos=normalized,
    )


@pytest.fixture(scope="module")
def default_pair() -> tuple[VoxelGridCA, StructureArtifact]:
    encoder = VoxelGridCA(use_cache=False)
    return encoder, _encode(encoder)


def test_default_encoder_state_is_owned_by_typed_config() -> None:
    encoder = VoxelGridCA()
    expected = ResidueVoxelGridConfig()

    assert encoder.config == expected
    assert set(encoder.__dict__) == {"config", "use_cache", "cache"}
    assert encoder.get_params() == expected.to_dict()
    assert encoder.name == "residue_voxel_grid"
    assert encoder.granularity_contract.level == "residue"


def test_config_and_flat_kwargs_are_equivalent() -> None:
    config = ResidueVoxelGridConfig(
        grid_size=32,
        voxel_size=1.5,
        channels=["occupancy", {"density": {"sigma": 2.0}}],
        pool="sum",
        dtype="float16",
    )
    canonical = VoxelGridCA(config=config)
    flat = VoxelGridCA(
        grid_size=32,
        voxel_size=1.5,
        channels=["occupancy", {"density": {"sigma": 2.0}}],
        pool="sum",
        dtype="float16",
    )

    assert canonical.config is config
    assert flat.config == config
    assert flat.get_params() == canonical.get_params()
    assert flat.fingerprint() == canonical.fingerprint()


def test_registry_supports_config_and_flat_kwargs() -> None:
    config = ResidueVoxelGridConfig(grid_size=32, channels="occupancy")
    flat = get_encoder("residue_voxel_grid", grid_size=32, channels="occupancy")
    canonical = get_encoder("residue_voxel_grid", config=config)

    assert isinstance(flat, VoxelGridCA)
    assert flat.config == canonical.config == config
    assert flat.fingerprint() == canonical.fingerprint()


def test_config_rejects_simultaneous_representation_kwargs() -> None:
    with pytest.raises(TypeError, match="config cannot be combined.*grid_size"):
        VoxelGridCA(config=ResidueVoxelGridConfig(), grid_size=32)


def test_constructor_rejects_wrong_config_type() -> None:
    with pytest.raises(TypeError, match="ResidueVoxelGridConfig"):
        VoxelGridCA(config=ResiduePointCloudConfig())


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"grid_size": 0}, ValueError, "grid_size"),
        ({"voxel_size": 0}, ValueError, "voxel_size"),
        ({"fit": 1}, TypeError, "fit"),
        ({"margin": -1}, ValueError, "margin"),
        ({"channels": "unknown"}, ValueError, "unknown grid feature"),
        ({"sigma": 0}, ValueError, "sigma"),
        ({"pool": "median"}, ValueError, "pool"),
        ({"clamp": 1}, TypeError, "clamp"),
        ({"dtype": "float64"}, ValueError, "dtype"),
        ({"residue_center": "invalid"}, ValueError, "residue_center"),
        ({"unknown": True}, TypeError, "unexpected keyword"),
    ],
)
def test_flat_kwargs_use_typed_config_validation(
    kwargs: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        VoxelGridCA(**kwargs)


def test_execution_options_are_separate_from_config_identity() -> None:
    config = ResidueVoxelGridConfig(grid_size=32)
    cache = object()
    operational = VoxelGridCA(config=config, use_cache=False, cache=cache)
    default_execution = VoxelGridCA(config=config)

    assert operational.config is config
    assert operational.use_cache is False
    assert operational.cache is cache
    assert operational.fingerprint() == default_execution.fingerprint()
    assert "use_cache" not in operational.get_params()
    assert "cache" not in operational.get_params()


def test_representation_properties_are_read_only_and_follow_config() -> None:
    encoder = VoxelGridCA(
        center=False,
        scale=2,
        residue_center="mean",
        grid_size=32,
        voxel_size=1.5,
        fit=False,
        margin=0,
        channels="occupancy+density",
        sigma=2,
        pool="sum",
        clamp=False,
        dtype="float16",
    )

    assert encoder.center is False
    assert encoder.scale == 2.0
    assert encoder.residue_center == "mean"
    assert encoder.amino_acids_only is True
    assert encoder.altloc == "first"
    assert encoder.missing_residue_center == "skip"
    assert encoder.grid_size == 32
    assert encoder.voxel_size == 1.5
    assert encoder.fit is False
    assert encoder.margin == 0.0
    assert encoder.channels == "occupancy+density"
    assert encoder.sigma == 2.0
    assert encoder.pool == "sum"
    assert encoder.clamp is False
    assert encoder.dtype == "float16"

    for name, value in (
        ("center", True),
        ("residue_center", "ca"),
        ("grid_size", 64),
        ("voxel_size", None),
        ("channels", "occupancy"),
        ("pool", "max"),
        ("dtype", "float32"),
    ):
        with pytest.raises(AttributeError):
            setattr(encoder, name, value)


def test_custom_constructor_runs_base_post_init_exactly_once(monkeypatch) -> None:
    calls: list[BaseEncoder] = []
    original = BaseEncoder.__post_init__

    def spy(encoder: BaseEncoder) -> None:
        calls.append(encoder)
        original(encoder)

    monkeypatch.setattr(BaseEncoder, "__post_init__", spy)
    encoder = VoxelGridCA()

    assert calls == [encoder]


@pytest.mark.parametrize("residue_center", ["ca", "mean", "com"])
def test_residue_policies_are_passed_explicitly_to_loader(
    monkeypatch,
    residue_center: str,
) -> None:
    captured: dict[str, Any] = {}
    sentinel = _structure(np.ones((1, 3), dtype=np.float32))

    def fake_load(inp: StructureInput, **kwargs: Any) -> dict[str, np.ndarray]:
        captured["inp"] = inp
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(voxel_module, "load_and_extract_residue", fake_load)
    encoder = VoxelGridCA(residue_center=residue_center)

    assert encoder._load_structure(EncodingContext(inp=INPUT)) is sentinel
    assert captured == {
        "inp": INPUT,
        "amino_acids_only": True,
        "mode": residue_center,
        "altloc": "first",
        "missing_ca": "skip",
    }
    assert (
        encoder.config.coordinate_source
        == {
            "ca": "ca",
            "mean": "residue_mean",
            "com": "residue_com",
        }[residue_center]
    )


@pytest.mark.parametrize(
    ("center", "scale", "expected"),
    [
        (True, None, [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        (True, 2, [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        (False, None, [[10.0, 20.0, 30.0], [12.0, 20.0, 30.0]]),
    ],
)
def test_base_is_sole_owner_of_center_and_scale(
    monkeypatch,
    center: bool,
    scale: float | None,
    expected: list[list[float]],
) -> None:
    encoder = VoxelGridCA(
        center=center,
        scale=scale,
        grid_size=3,
        channels="occupancy",
    )
    ctx = _context(
        encoder,
        np.asarray([[10, 20, 30], [12, 20, 30]], dtype=np.float32),
    )
    captured: dict[str, Any] = {}

    def fake_builder(**kwargs: Any) -> tuple[np.ndarray, list[str], dict[str, Any]]:
        captured.update(kwargs)
        return (
            np.zeros((1, 3, 3, 3), dtype=np.float32),
            ["occupancy"],
            {"voxel_size": 1.0, "origin": np.zeros(3, dtype=np.float32)},
        )

    monkeypatch.setattr(voxel_module, "build_grid_attr", fake_builder)
    encoder._create_data(ctx)

    np.testing.assert_allclose(captured["pos"], expected)
    assert captured["center"] is False
    if center:
        np.testing.assert_array_equal(ctx.centroid, [[11.0, 20.0, 30.0]])
    else:
        np.testing.assert_array_equal(ctx.centroid, [[0.0, 0.0, 0.0]])


def test_single_residue_centering_produces_an_occupied_voxel() -> None:
    encoder = VoxelGridCA(
        grid_size=3,
        voxel_size=1,
        fit=False,
        channels="occupancy",
    )
    artifact_data = encoder._create_data(
        _context(encoder, np.asarray([[10, 20, 30]], dtype=np.float32))
    )

    assert artifact_data["grid"].shape == (1, 3, 3, 3)
    assert artifact_data["grid"].sum() == 1.0
    assert artifact_data["grid"][0, 1, 1, 1] == 1.0


@pytest.mark.parametrize(
    ("voxel_size", "fit", "expected_spacing"),
    [
        (None, True, 1.5),
        (None, False, 1.0),
        (2.0, True, 2.0),
        (2.0, False, 2.0),
    ],
)
def test_requested_and_fit_spacing_matrix(
    voxel_size: float | None,
    fit: bool,
    expected_spacing: float,
) -> None:
    encoder = VoxelGridCA(
        center=False,
        grid_size=5,
        voxel_size=voxel_size,
        fit=fit,
        margin=1,
        channels="occupancy",
    )
    data = encoder._create_data(
        _context(encoder, np.asarray([[-2, 0, 0], [2, 0, 0]], dtype=np.float32))
    )

    assert encoder.get_params()["voxel_size"] == voxel_size
    assert float(data["voxel_size"][0]) == expected_spacing
    np.testing.assert_array_equal(
        data["origin"],
        np.full(3, -2 * expected_spacing, dtype=np.float32),
    )


def test_margin_zero_preserves_current_fit_formula() -> None:
    encoder = VoxelGridCA(
        center=False,
        grid_size=5,
        margin=0,
        channels="occupancy",
    )
    data = encoder._create_data(
        _context(encoder, np.asarray([[-2, 0, 0], [2, 0, 0]], dtype=np.float32))
    )

    assert float(data["voxel_size"][0]) == 1.0
    assert encoder.margin == 0.0


def test_grid_size_one_and_known_degenerate_configuration_remain_accepted() -> None:
    assert ResidueVoxelGridConfig(grid_size=1).grid_size == 1
    config = ResidueVoxelGridConfig(grid_size=1, margin=0)
    assert (config.grid_size, config.margin) == (1, 0.0)


def test_feature_mode_adapter_preserves_requested_and_physical_orders() -> None:
    requested = ["aa_onehot", "occupancy", {"density": {"sigma": 2.0}}]
    encoder = VoxelGridCA(
        grid_size=8,
        channels=requested,
        use_cache=False,
    )
    artifact = _encode(encoder)

    assert encoder.config.channels.to_list() == requested
    assert artifact.meta["params"]["channels"] == requested
    assert artifact.meta["feature_keys"]["grid_channels"] == [
        "occupancy",
        "density",
        *AA_ONEHOT_CHANNELS,
    ]
    assert artifact.data["grid"].shape == (23, 8, 8, 8)


@pytest.mark.parametrize(
    ("channels", "expected_count", "expected_keys"),
    [
        ("occupancy", 1, ["occupancy"]),
        ("aa_onehot", 21, AA_ONEHOT_CHANNELS),
    ],
)
def test_single_feature_channel_shapes_and_metadata(
    channels: str,
    expected_count: int,
    expected_keys: list[str],
) -> None:
    artifact = _encode(VoxelGridCA(grid_size=8, channels=channels, use_cache=False))

    assert artifact.data["grid"].shape == (expected_count, 8, 8, 8)
    assert artifact.meta["feature_keys"]["grid_channels"] == expected_keys


@pytest.mark.parametrize(
    ("pool", "expected"),
    [("max", 1.0), ("sum", 2.0), ("mean", 1.0)],
)
def test_pool_modes_preserve_collision_semantics(pool: str, expected: float) -> None:
    encoder = VoxelGridCA(
        center=False,
        grid_size=4,
        voxel_size=1,
        fit=False,
        channels="occupancy",
        pool=pool,
        clamp=False,
    )
    data = encoder._create_data(
        _context(encoder, np.asarray([[0, 0, 0], [0.1, 0, 0]], dtype=np.float32))
    )

    assert data["grid"].max() == expected


def test_clamp_is_applied_after_pooling() -> None:
    kwargs = {
        "center": False,
        "grid_size": 4,
        "voxel_size": 1,
        "fit": False,
        "channels": "occupancy",
        "pool": "sum",
    }
    positions = np.asarray([[0, 0, 0], [0.1, 0, 0]], dtype=np.float32)
    clamped = VoxelGridCA(**kwargs, clamp=True)
    raw = VoxelGridCA(**kwargs, clamp=False)

    assert clamped._create_data(_context(clamped, positions))["grid"].max() == 1.0
    assert raw._create_data(_context(raw, positions))["grid"].max() == 2.0


def test_sigma_changes_density_output_in_voxel_units() -> None:
    positions = np.asarray([[0, 0, 0]], dtype=np.float32)
    narrow = VoxelGridCA(
        center=False,
        grid_size=7,
        voxel_size=1,
        fit=False,
        channels="density",
        sigma=1,
    )
    wide = VoxelGridCA(
        center=False,
        grid_size=7,
        voxel_size=1,
        fit=False,
        channels="density",
        sigma=2,
    )
    narrow_grid = narrow._create_data(_context(narrow, positions))["grid"]
    wide_grid = wide._create_data(_context(wide, positions))["grid"]

    assert not np.array_equal(narrow_grid, wide_grid)
    assert wide_grid.sum() > narrow_grid.sum()


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_final_dtype_follows_validated_config(dtype: str) -> None:
    encoder = VoxelGridCA(
        grid_size=4,
        channels="occupancy",
        dtype=dtype,
    )
    data = encoder._create_data(_context(encoder, np.asarray([[0, 0, 0]], dtype=np.float32)))
    assert data["grid"].dtype == np.dtype(dtype)


def test_requested_params_and_effective_geometry_remain_separate(
    default_pair: tuple[VoxelGridCA, StructureArtifact],
) -> None:
    encoder, artifact = default_pair

    assert encoder.get_params() == encoder.config.to_dict()
    assert artifact.meta["params"] == encoder.config.to_dict()
    assert artifact.meta["params"]["scale"] is None
    assert artifact.meta["params"]["voxel_size"] is None
    assert float(artifact.data["voxel_size"][0]) > 0
    assert artifact.data["origin"].shape == (3,)
    assert "voxel_size_effective" not in artifact.meta["params"]
    assert "origin" not in artifact.meta["params"]


def test_explicit_spacing_remains_requested_in_params_and_effective_in_data() -> None:
    encoder = VoxelGridCA(
        grid_size=8,
        voxel_size=1.5,
        channels="occupancy",
        use_cache=False,
    )
    artifact = _encode(encoder)

    assert artifact.meta["params"]["voxel_size"] == 1.5
    assert float(artifact.data["voxel_size"][0]) == 1.5


def test_default_scientific_baseline_is_corrected_and_strict(
    default_pair: tuple[VoxelGridCA, StructureArtifact],
) -> None:
    encoder, artifact = default_pair
    grid = artifact.data["grid"]

    assert grid.shape == (25, 64, 64, 64)
    assert grid.dtype == np.float32
    assert _digest(grid) == DEFAULT_GRID_HASH
    assert artifact.meta["feature_keys"] == {
        "grid_key": "grid",
        "grid_layout": "CXYZ",
        "grid_channels": [
            "occupancy",
            "density",
            "plddt",
            "bfactor",
            *AA_ONEHOT_CHANNELS,
        ],
    }
    assert artifact.meta["params"] == encoder.config.to_dict()
    artifact.finalize()
    artifact.validate(strict=True)


def test_all_loaded_residues_contribute_to_corrected_default_grid(
    default_pair: tuple[VoxelGridCA, StructureArtifact],
) -> None:
    encoder, artifact = default_pair
    structure = load_and_extract_residue(
        INPUT,
        amino_acids_only=encoder.config.amino_acids_only,
        mode=encoder.config.residue_center,
        altloc=encoder.config.altloc,
        missing_ca=encoder.config.missing_residue_center,
    )
    _, normalized = encoder._normalize_structure(structure["pos"])
    spacing = float(artifact.data["voxel_size"][0])
    origin = artifact.data["origin"]
    indices = np.floor((normalized - origin[None, :]) / spacing).astype(np.int64)
    valid = np.all((indices >= 0) & (indices < encoder.grid_size), axis=1)

    assert len(structure["pos"]) == 534
    assert int(valid.sum()) == 534


def test_center_false_preserves_uncentered_scientific_baseline() -> None:
    encoder = VoxelGridCA(center=False, use_cache=False)
    artifact = _encode(encoder)

    assert _digest(artifact.data["grid"]) == CENTER_FALSE_GRID_HASH
    np.testing.assert_array_equal(artifact.data["centroid"], [[0.0, 0.0, 0.0]])
    artifact.finalize()
    artifact.validate(strict=True)


def test_corrected_representative_grid_size_regression() -> None:
    artifact = _encode(VoxelGridCA(grid_size=16, use_cache=False))

    assert _digest(artifact.data["grid"]) == (
        "e5af985dfe02f8fd842097d6758a8d2d1eff40d6969513f7fc8df5102cb7335d"
    )
    artifact.finalize()
    artifact.validate(strict=True)


def test_equivalent_channel_forms_have_canonical_identity() -> None:
    forms = (
        "occupancy+density",
        ["occupancy", "density"],
        ("occupancy", "density"),
    )
    encoders = [VoxelGridCA(channels=form) for form in forms]

    assert all(encoder.config == encoders[0].config for encoder in encoders[1:])
    assert all(encoder.get_params() == encoders[0].get_params() for encoder in encoders[1:])
    assert all(encoder.fingerprint() == encoders[0].fingerprint() for encoder in encoders[1:])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"center": False},
        {"scale": 2},
        {"residue_center": "mean"},
        {"grid_size": 32},
        {"voxel_size": 1.5},
        {"fit": False},
        {"margin": 0},
        {"channels": "occupancy"},
        {"sigma": 2},
        {"pool": "sum"},
        {"clamp": False},
        {"dtype": "float16"},
    ],
)
def test_semantic_config_changes_fingerprint(kwargs: dict[str, Any]) -> None:
    assert VoxelGridCA().fingerprint() != VoxelGridCA(**kwargs).fingerprint()


def test_fingerprint_is_deterministic_and_excludes_execution_state() -> None:
    config = ResidueVoxelGridConfig(
        grid_size=32,
        channels=[{"density": {"sigma": 2.0}}, "occupancy"],
    )
    first = VoxelGridCA(config=config, use_cache=False, cache=object())
    second = VoxelGridCA(config=config)

    assert first.fingerprint() == second.fingerprint()
    assert first.fingerprint() == first.fingerprint()


@pytest.mark.parametrize(
    "encoder",
    [
        VoxelGridCA(grid_size=8, channels="occupancy", use_cache=False),
        VoxelGridCA(
            grid_size=8,
            channels="occupancy+density",
            pool="mean",
            clamp=False,
            dtype="float16",
            use_cache=False,
        ),
    ],
)
def test_representative_variants_emit_strict_valid_artifacts(
    encoder: VoxelGridCA,
) -> None:
    artifact = _encode(encoder)

    assert artifact.meta["params"] == encoder.config.to_dict()
    artifact.finalize()
    artifact.validate(strict=True)
