from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import eris.core.base_encoder.base_atom_encoder as atom_base_module
import eris.encoders.clouds.residue_point_cloud as residue_cloud_module
from eris.core.artifacts import StructureArtifact
from eris.core.base_encoder.base_encoder import BaseEncoder, EncodingContext
from eris.core.config import AtomPointCloudConfig, ResiduePointCloudConfig
from eris.core.io import load_and_extract_atoms, load_and_extract_residue
from eris.core.registry import get_encoder
from eris.core.types import StructureInput
from eris.encoders.clouds.atom_point_cloud import AtomPointCloud
from eris.encoders.clouds.residue_point_cloud import ResiduePointCloudCA
from eris.features.node.atom import build_node_attr as build_atom_node_attr
from eris.features.node.residue import build_node_attr as build_residue_node_attr

DATA = Path("tests/data/1AQ2.pdb")
INPUT = StructureInput(source=str(DATA))


def _encode(encoder: AtomPointCloud | ResiduePointCloudCA) -> StructureArtifact:
    return encoder.encode(INPUT)


def _assert_same_data(first: StructureArtifact, second: StructureArtifact) -> None:
    assert first.data.keys() == second.data.keys()
    for key in first.data:
        np.testing.assert_array_equal(first.data[key], second.data[key])


@pytest.mark.parametrize(
    ("encoder", "expected_config", "expected_mode"),
    [
        (
            AtomPointCloud(),
            AtomPointCloudConfig(),
            "elem+role+bfactor+occ",
        ),
        (
            ResiduePointCloudCA(),
            ResiduePointCloudConfig(),
            "onehot+physchem+plddt_ca+bfactor_ca",
        ),
    ],
)
def test_default_encoder_state_is_owned_by_typed_config(
    encoder: AtomPointCloud | ResiduePointCloudCA,
    expected_config: AtomPointCloudConfig | ResiduePointCloudConfig,
    expected_mode: str,
) -> None:
    assert encoder.config == expected_config
    assert set(vars(encoder)) == {"config", "use_cache", "cache"}
    assert encoder.get_params() == expected_config.to_dict()
    assert encoder.get_params()["scale"] is None
    assert encoder.node_attr_mode == expected_mode


def test_atom_flat_kwargs_and_config_construction_are_equivalent() -> None:
    config = AtomPointCloudConfig(
        center=False,
        scale=2.0,
        include_h=True,
        altloc="max_occupancy",
        node_attr_mode="occ+elem",
    )
    flat = AtomPointCloud(
        center=False,
        scale=2.0,
        include_h=True,
        altloc="max_occupancy",
        node_attr_mode="occ+elem",
        use_cache=False,
        cache=object(),
    )
    canonical = AtomPointCloud(
        config=config,
        use_cache=False,
        cache=object(),
    )

    assert flat.config == canonical.config == config
    assert flat.get_params() == canonical.get_params()
    assert flat.fingerprint() == canonical.fingerprint()
    _assert_same_data(_encode(flat), _encode(canonical))


def test_residue_flat_kwargs_and_config_construction_are_equivalent() -> None:
    config = ResiduePointCloudConfig(
        center=False,
        scale=2.0,
        residue_center="mean",
        node_attr_mode="bfactor_ca+onehot",
    )
    flat = ResiduePointCloudCA(
        center=False,
        scale=2.0,
        residue_center="mean",
        node_attr_mode="bfactor_ca+onehot",
        use_cache=False,
        cache=object(),
    )
    canonical = ResiduePointCloudCA(
        config=config,
        use_cache=False,
        cache=object(),
    )

    assert flat.config == canonical.config == config
    assert flat.get_params() == canonical.get_params()
    assert flat.fingerprint() == canonical.fingerprint()
    _assert_same_data(_encode(flat), _encode(canonical))


@pytest.mark.parametrize(
    ("encoder_type", "config", "representation_kwargs"),
    [
        (AtomPointCloud, AtomPointCloudConfig(), {"center": False}),
        (AtomPointCloud, AtomPointCloudConfig(), {"scale": 2.0}),
        (AtomPointCloud, AtomPointCloudConfig(), {"include_h": True}),
        (AtomPointCloud, AtomPointCloudConfig(), {"node_attr_mode": "elem"}),
        (ResiduePointCloudCA, ResiduePointCloudConfig(), {"center": False}),
        (ResiduePointCloudCA, ResiduePointCloudConfig(), {"scale": 2.0}),
        (
            ResiduePointCloudCA,
            ResiduePointCloudConfig(),
            {"residue_center": "mean"},
        ),
        (
            ResiduePointCloudCA,
            ResiduePointCloudConfig(),
            {"node_attr_mode": "onehot"},
        ),
    ],
)
def test_config_rejects_simultaneous_representation_kwargs(
    encoder_type: type[AtomPointCloud] | type[ResiduePointCloudCA],
    config: AtomPointCloudConfig | ResiduePointCloudConfig,
    representation_kwargs: dict[str, Any],
) -> None:
    with pytest.raises(TypeError, match="config cannot be combined"):
        encoder_type(config=config, **representation_kwargs)


def test_constructors_reject_the_wrong_config_type() -> None:
    with pytest.raises(TypeError, match="AtomPointCloudConfig"):
        AtomPointCloud(config=ResiduePointCloudConfig())
    with pytest.raises(TypeError, match="ResiduePointCloudConfig"):
        ResiduePointCloudCA(config=AtomPointCloudConfig())


def test_flat_kwargs_use_typed_config_validation() -> None:
    with pytest.raises(ValueError, match="scale must be >"):
        AtomPointCloud(scale=0)
    with pytest.raises(ValueError, match="residue_center"):
        ResiduePointCloudCA(residue_center="invalid")
    with pytest.raises(TypeError, match="unexpected keyword"):
        AtomPointCloud(unknown_parameter=True)


@pytest.mark.parametrize(
    ("encoder_type", "config"),
    [
        (AtomPointCloud, AtomPointCloudConfig(center=False)),
        (
            ResiduePointCloudCA,
            ResiduePointCloudConfig(residue_center="mean"),
        ),
    ],
)
def test_execution_options_coexist_with_config_and_do_not_affect_identity(
    encoder_type: type[AtomPointCloud] | type[ResiduePointCloudCA],
    config: AtomPointCloudConfig | ResiduePointCloudConfig,
) -> None:
    cache = object()
    operational = encoder_type(config=config, use_cache=False, cache=cache)
    default_execution = encoder_type(config=config)

    assert operational.config is config
    assert operational.use_cache is False
    assert operational.cache is cache
    assert operational.get_params() == config.to_dict()
    assert operational.fingerprint() == default_execution.fingerprint()
    assert "use_cache" not in operational.get_params()
    assert "cache" not in operational.get_params()

    operational.use_cache = True
    assert operational.use_cache is True


def test_representation_properties_are_read_only() -> None:
    atom = AtomPointCloud()
    residue = ResiduePointCloudCA()

    with pytest.raises(AttributeError):
        atom.center = False
    with pytest.raises(AttributeError):
        atom.include_h = True
    with pytest.raises(AttributeError):
        atom.node_attr_mode = "elem"
    with pytest.raises(AttributeError):
        residue.residue_center = "mean"
    with pytest.raises(AttributeError):
        residue.node_attr_mode = "onehot"


def test_custom_constructors_run_base_post_init_exactly_once(monkeypatch) -> None:
    calls: list[BaseEncoder] = []
    original = BaseEncoder.__post_init__

    def spy(encoder: BaseEncoder) -> None:
        calls.append(encoder)
        original(encoder)

    monkeypatch.setattr(BaseEncoder, "__post_init__", spy)

    atom = AtomPointCloud()
    residue = ResiduePointCloudCA()

    assert calls == [atom, residue]


def test_atom_config_filters_are_passed_to_the_inherited_loader(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    sentinel = {"pos": np.ones((1, 3), dtype=np.float32)}

    def fake_load(inp: StructureInput, **kwargs: Any) -> dict[str, np.ndarray]:
        captured["inp"] = inp
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(atom_base_module, "load_and_extract_atoms", fake_load)
    config = AtomPointCloudConfig(
        include_hetero=True,
        include_h=True,
        drop_water=False,
        keep_ions=False,
        altloc="max_occupancy",
        amino_acids_only=True,
    )
    encoder = AtomPointCloud(config=config)
    ctx = EncodingContext(inp=INPUT)

    assert encoder._load_structure(ctx) is sentinel
    assert captured == {
        "inp": INPUT,
        "include_hetero": True,
        "include_h": True,
        "drop_water": False,
        "keep_ions": False,
        "altloc": "max_occupancy",
        "amino_acids_only": True,
    }


@pytest.mark.parametrize("residue_center", ["ca", "mean", "com"])
def test_residue_config_policies_are_passed_explicitly_to_loader(
    monkeypatch,
    residue_center: str,
) -> None:
    captured: dict[str, Any] = {}
    sentinel = {"pos": np.ones((1, 3), dtype=np.float32)}

    def fake_load(inp: StructureInput, **kwargs: Any) -> dict[str, np.ndarray]:
        captured["inp"] = inp
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(residue_cloud_module, "load_and_extract_residue", fake_load)
    config = ResiduePointCloudConfig(residue_center=residue_center)
    encoder = ResiduePointCloudCA(config=config)
    ctx = EncodingContext(inp=INPUT)

    assert encoder._load_structure(ctx) is sentinel
    assert captured == {
        "inp": INPUT,
        "amino_acids_only": True,
        "mode": residue_center,
        "altloc": "first",
        "missing_ca": "skip",
    }


def test_registry_preserves_flat_and_config_aware_point_cloud_construction() -> None:
    atom_config = AtomPointCloudConfig(center=False, include_h=True)
    residue_config = ResiduePointCloudConfig(residue_center="mean")

    atom_flat = get_encoder("atom_point_cloud", center=False, include_h=True)
    atom_canonical = get_encoder("atom_point_cloud", config=atom_config)
    residue_flat = get_encoder("residue_point_cloud", residue_center="mean")
    residue_canonical = get_encoder("residue_point_cloud", config=residue_config)

    assert atom_flat.config == atom_canonical.config == atom_config
    assert atom_flat.fingerprint() == atom_canonical.fingerprint()
    assert residue_flat.config == residue_canonical.config == residue_config
    assert residue_flat.fingerprint() == residue_canonical.fingerprint()


def test_atom_default_output_matches_pre_config_runtime_semantics(tmp_path: Path) -> None:
    config = AtomPointCloudConfig()
    raw = load_and_extract_atoms(
        INPUT,
        include_hetero=config.include_hetero,
        include_h=config.include_h,
        drop_water=config.drop_water,
        keep_ions=config.keep_ions,
        altloc=config.altloc,
        amino_acids_only=config.amino_acids_only,
    )
    centroid = raw["pos"].mean(axis=0, keepdims=True).astype(np.float32)
    expected_node_attr, expected_keys = build_atom_node_attr(
        elem_index=raw["elem_index"],
        atom_role=raw["atom_role"],
        bfactor=raw["bfactor"],
        occ=raw["occ"],
        bfactor_norm=True,
        mode="elem+role+bfactor+occ",
        return_keys=True,
    )

    encoder = AtomPointCloud(config=config, use_cache=False, cache=object())
    artifact = _encode(encoder)

    assert artifact.data["pos"].shape == (4114, 3)
    assert artifact.data["node_attr"].shape == (4114, 25)
    np.testing.assert_array_equal(artifact.data["centroid"], centroid)
    np.testing.assert_array_equal(artifact.data["pos"], raw["pos"] - centroid)
    np.testing.assert_array_equal(artifact.data["node_attr"], expected_node_attr)
    for key in (
        "elem_index",
        "atom_role",
        "bfactor",
        "occ",
        "residue_ids",
        "resname",
        "atom_name",
        "seq_pos_residue",
        "aa_index_residue",
    ):
        np.testing.assert_array_equal(artifact.data[key], raw[key])
    assert artifact.meta["parameters"] == config.to_dict()
    assert artifact.meta["parameters"]["scale"] is None
    assert artifact.meta["feature_keys"]["node_attr_keys"] == expected_keys

    outdir = tmp_path / "atom-artifact"
    artifact.save(outdir)
    loaded = StructureArtifact.load(outdir, strict=True)
    assert loaded.meta["parameters"] == config.to_dict()


@pytest.mark.parametrize(
    ("residue_center", "coordinate_source"),
    [
        ("ca", "ca"),
        ("mean", "residue_mean"),
        ("com", "residue_com"),
    ],
)
def test_residue_outputs_preserve_scientific_semantics_and_strict_artifacts(
    residue_center: str,
    coordinate_source: str,
) -> None:
    config = ResiduePointCloudConfig(residue_center=residue_center)
    raw = load_and_extract_residue(
        INPUT,
        amino_acids_only=config.amino_acids_only,
        mode=config.residue_center,
        altloc=config.altloc,
        missing_ca=config.missing_residue_center,
    )
    centroid = raw["pos"].mean(axis=0, keepdims=True).astype(np.float32)
    expected_node_attr, expected_keys = build_residue_node_attr(
        raw["aa_index"],
        plddt=raw["plddt"],
        bfactor_ca=raw["bfactor_ca"],
        mode="onehot+physchem+plddt_ca+bfactor_ca",
        return_keys=True,
    )

    encoder = ResiduePointCloudCA(config=config, use_cache=False, cache=object())
    artifact = _encode(encoder)

    assert artifact.data["pos"].shape == (534, 3)
    assert artifact.data["node_attr"].shape == (534, 38)
    np.testing.assert_array_equal(artifact.data["centroid"], centroid)
    np.testing.assert_array_equal(artifact.data["pos"], raw["pos"] - centroid)
    np.testing.assert_array_equal(artifact.data["node_attr"], expected_node_attr)
    for key in ("residue_ids", "resname", "aa_index", "plddt", "bfactor_ca"):
        np.testing.assert_array_equal(artifact.data[key], raw[key])
    assert artifact.meta["granularity"] == "residue"
    assert config.coordinate_source == coordinate_source
    assert artifact.meta["parameters"] == config.to_dict()
    assert artifact.meta["parameters"]["scale"] is None
    assert artifact.meta["feature_keys"]["node_attr_keys"] == expected_keys

    artifact.finalize()
    artifact.validate(strict=True)


@pytest.mark.parametrize(
    ("encoder", "expected_width", "requested_mode"),
    [
        (
            AtomPointCloud(
                config=AtomPointCloudConfig(node_attr_mode="occ+elem"),
                use_cache=False,
                cache=object(),
            ),
            18,
            ["occ", "elem"],
        ),
        (
            ResiduePointCloudCA(
                config=ResiduePointCloudConfig(node_attr_mode="bfactor_ca+onehot"),
                use_cache=False,
                cache=object(),
            ),
            22,
            ["bfactor_ca", "onehot"],
        ),
    ],
)
def test_feature_mode_uses_canonical_config_without_losing_requested_order(
    encoder: AtomPointCloud | ResiduePointCloudCA,
    expected_width: int,
    requested_mode: list[str],
) -> None:
    artifact = _encode(encoder)

    assert artifact.data["node_attr"].shape[1] == expected_width
    assert artifact.meta["parameters"]["node_attr_mode"] == requested_mode
