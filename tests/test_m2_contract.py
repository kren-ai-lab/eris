from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from eris.core.registry import (
    get_encoder,
    get_encoder_info,
    list_encoder_info,
    list_encoders,
)
from eris.core.types import StructureInput

DATA = Path("tests/data/1AQ2.pdb")

CANONICAL_NAMES = [
    "atom_knn_graph",
    "atom_point_cloud",
    "residue_contact_graph",
    "residue_contact_map",
    "residue_contact_map_bins",
    "residue_distogram",
    "residue_distogram_bins",
    "residue_knn_graph",
    "residue_point_cloud",
    "residue_voxel_grid",
]

HISTORICAL_ALIASES = [
    "contact_map_bins_ca",
    "contact_map_bins_residue",
    "contact_map_ca",
    "contact_map_residue",
    "distogram_bins_ca",
    "distogram_bins_residue",
    "distogram_ca",
    "distogram_residue",
    "voxel_grid_ca",
]


def test_m2_registry_identity_and_alias_contract():
    assert list_encoders() == CANONICAL_NAMES
    assert len(list_encoder_info()) == 10

    for registry_name, info in zip(list_encoders(), list_encoder_info(), strict=True):
        assert registry_name == info.name
        assert registry_name == info.encoder_type.name
        assert registry_name == info.config_type.representation_name

    for alias in HISTORICAL_ALIASES:
        with pytest.raises(KeyError, match="Unknown encoder"):
            get_encoder(alias)
        with pytest.raises(KeyError, match="No encoder info"):
            get_encoder_info(alias)


@pytest.mark.parametrize("name", CANONICAL_NAMES)
def test_m2_config_ownership_execution_separation_and_construction_equivalence(name):
    info = get_encoder_info(name)
    flat = get_encoder(name, use_cache=False)
    configured = get_encoder(name, config=info.default_config(), use_cache=False)
    with_cache = get_encoder(name, use_cache=True, cache=object())
    changed_config = get_encoder(name, center=False, use_cache=False)

    assert type(flat.config) is info.config_type
    assert flat.config == info.default_config()
    assert flat.get_params() == flat.config.to_dict()
    assert configured.config == flat.config
    assert configured.get_params() == flat.get_params()
    assert configured.fingerprint() == flat.fingerprint()
    assert with_cache.get_params() == flat.get_params()
    assert with_cache.fingerprint() == flat.fingerprint()
    assert changed_config.get_params() != flat.get_params()
    assert changed_config.fingerprint() != flat.fingerprint()


@pytest.mark.parametrize("name", CANONICAL_NAMES)
def test_m2_introspection_matches_exact_config_contract(name):
    info = get_encoder_info(name)
    schema = info.params_schema()

    assert [entry["name"] for entry in schema] == [
        config_field.name for config_field in fields(info.config_type)
    ]
    assert {entry["name"]: entry["default"] for entry in schema} == info.defaults()


@pytest.fixture(scope="module")
def default_artifacts():
    artifacts = {}
    structure_input = StructureInput(source=str(DATA))
    for name in CANONICAL_NAMES:
        encoder = get_encoder(name, use_cache=False)
        artifact = encoder.encode(structure_input)
        artifact.finalize()
        artifact.validate(strict=True)
        artifacts[name] = (encoder, artifact)
    return artifacts


@pytest.mark.parametrize("name", CANONICAL_NAMES)
def test_m2_default_artifact_metadata_is_canonical_and_strict_valid(name, default_artifacts):
    encoder, artifact = default_artifacts[name]

    assert artifact.meta["params"] == encoder.config.to_dict()
    assert artifact.meta["encoder_fingerprint"] == encoder.fingerprint()


def test_m2_requested_and_effective_state_remain_separate(default_artifacts):
    for name in ("residue_contact_map_bins", "residue_distogram_bins"):
        encoder, artifact = default_artifacts[name]
        assert encoder.config.edges is None
        assert artifact.meta["params"]["edges"] is None
        assert "bin_edges" in artifact.data
        assert "bin_edges" not in artifact.meta["params"]

    encoder, artifact = default_artifacts["residue_voxel_grid"]
    assert encoder.config.voxel_size is None
    assert artifact.meta["params"]["voxel_size"] is None
    assert "voxel_size" in artifact.data
