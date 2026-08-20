from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from eris.core.config import (
    AtomKNNGraphConfig,
    AtomPointCloudConfig,
    ResidueContactGraphConfig,
    ResidueContactMapBinsConfig,
    ResidueContactMapConfig,
    ResidueDistogramBinsConfig,
    ResidueDistogramConfig,
    ResidueKNNGraphConfig,
    ResiduePointCloudConfig,
    ResidueVoxelGridConfig,
)
from eris.core.registry import (
    get_encoder,
    get_encoder_info,
    list_encoder_info,
    list_encoders,
    register_encoder,
)
from eris.core.types import StructureInput
from eris.encoders.clouds.atom_point_cloud import AtomPointCloud
from eris.encoders.clouds.residue_point_cloud import ResiduePointCloudCA
from eris.encoders.geometry.contact_map_bins_ca import ContactMapBinsCA
from eris.encoders.geometry.contact_map_ca import ContactMapCA
from eris.encoders.geometry.distogram_bins_ca import DistogramBinsCA
from eris.encoders.geometry.distogram_ca import DistogramCA
from eris.encoders.graphs.atom_knn_graph import AtomKNNGraph
from eris.encoders.graphs.residue_contact_graph import ResidueContactGraph
from eris.encoders.graphs.residue_knn_graph import ResidueKNNGraph
from eris.encoders.grids.voxel_grid_ca import VoxelGridCA

DATA = Path("tests/data/1AQ2.pdb")

EXPECTED_ENTRIES = {
    "atom_knn_graph": (AtomKNNGraphConfig, AtomKNNGraph, "graph", "atom", "atom"),
    "atom_point_cloud": (
        AtomPointCloudConfig,
        AtomPointCloud,
        "cloud",
        "atom",
        "atom",
    ),
    "residue_contact_graph": (
        ResidueContactGraphConfig,
        ResidueContactGraph,
        "graph",
        "residue",
        "ca",
    ),
    "residue_contact_map": (
        ResidueContactMapConfig,
        ContactMapCA,
        "matrix",
        "residue",
        "ca",
    ),
    "residue_contact_map_bins": (
        ResidueContactMapBinsConfig,
        ContactMapBinsCA,
        "matrix",
        "residue",
        "ca",
    ),
    "residue_distogram": (
        ResidueDistogramConfig,
        DistogramCA,
        "matrix",
        "residue",
        "ca",
    ),
    "residue_distogram_bins": (
        ResidueDistogramBinsConfig,
        DistogramBinsCA,
        "matrix",
        "residue",
        "ca",
    ),
    "residue_knn_graph": (
        ResidueKNNGraphConfig,
        ResidueKNNGraph,
        "graph",
        "residue",
        "ca",
    ),
    "residue_point_cloud": (
        ResiduePointCloudConfig,
        ResiduePointCloudCA,
        "cloud",
        "residue",
        "ca",
    ),
    "residue_voxel_grid": (
        ResidueVoxelGridConfig,
        VoxelGridCA,
        "grid",
        "residue",
        "ca",
    ),
}
CANONICAL_NAMES = sorted(EXPECTED_ENTRIES)
HISTORICAL_NAMES = [
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


class NoncanonicalConfig(AtomPointCloudConfig):
    representation_name = "noncanonical_point_cloud"


def test_registry_has_exactly_ten_sorted_canonical_entries():
    assert list_encoders() == CANONICAL_NAMES
    assert len(list_encoder_info()) == 10
    assert not set(HISTORICAL_NAMES).intersection(list_encoders())


@pytest.mark.parametrize("name", HISTORICAL_NAMES)
def test_historical_names_are_not_registry_aliases(name):
    with pytest.raises(KeyError, match="Unknown encoder"):
        get_encoder(name)

    with pytest.raises(KeyError, match="No encoder info"):
        get_encoder_info(name)


@pytest.mark.parametrize(
    ("name", "expected"),
    EXPECTED_ENTRIES.items(),
)
def test_registry_associations_and_derived_metadata(name, expected):
    config_type, encoder_type, representation_type, granularity, coordinate_source = expected
    info = get_encoder_info(name)

    assert info.name == name
    assert info.config_type is config_type
    assert info.encoder_type is encoder_type
    assert info.family == "explicit"
    assert info.representation_type == representation_type
    assert info.granularity == granularity
    assert info.granularity != "ca"
    assert info.default_coordinate_source == coordinate_source
    assert info.default_config() == config_type()
    assert info.defaults() == config_type().to_dict()
    assert info.config_fields() == fields(config_type)
    assert not hasattr(info, "params_schema")


@pytest.mark.parametrize("name", CANONICAL_NAMES)
def test_canonical_encoder_construction_and_fingerprints(name):
    first = get_encoder(name, use_cache=False)
    second = get_encoder(name, use_cache=False)

    assert first.name == name
    assert first.fingerprint()
    assert first.fingerprint() == second.fingerprint()


def test_default_fingerprints_do_not_collide():
    fingerprints = {get_encoder(name, use_cache=False).fingerprint() for name in CANONICAL_NAMES}

    assert len(fingerprints) == len(CANONICAL_NAMES)


def test_get_encoder_preserves_direct_operational_and_encoder_kwargs():
    cache = object()
    encoder = get_encoder(
        "residue_contact_graph",
        use_cache=False,
        cache=cache,
        cutoff=9.5,
        local_cutoff=4.5,
    )

    assert encoder.use_cache is False
    assert encoder.cache is cache
    assert encoder.cutoff == 9.5
    assert encoder.local_cutoff == 4.5


def test_registration_rejects_duplicate_canonical_name():
    with pytest.raises(KeyError, match="Encoder already registered: atom_point_cloud"):
        register_encoder(AtomPointCloudConfig, AtomPointCloud)


def test_registration_rejects_noncanonical_config_subclass():
    with pytest.raises(TypeError, match="config_type is not the canonical config"):
        register_encoder(NoncanonicalConfig, AtomPointCloud)


def test_registry_defaults_have_no_stale_manual_parameters():
    atom_cloud = get_encoder_info("atom_point_cloud").defaults()
    atom_graph = get_encoder_info("atom_knn_graph").defaults()
    residue_knn = get_encoder_info("residue_knn_graph").defaults()
    distogram_bins = get_encoder_info("residue_distogram_bins").defaults()
    contact_bins = get_encoder_info("residue_contact_map_bins").defaults()
    contact = get_encoder_info("residue_contact_map").defaults()
    voxel = get_encoder_info("residue_voxel_grid").defaults()

    assert "mode" not in atom_cloud
    assert "include_self_loops" not in atom_graph
    assert "symmetrize" not in residue_knn
    assert distogram_bins["num_bins"] == contact_bins["num_bins"] == 36
    assert contact["dtype"] == "uint8"
    assert voxel["grid_size"] == 64
    assert voxel["margin"] == 4.0
    assert voxel["channels"] == [
        "occupancy",
        "density",
        "aa_onehot",
        "plddt",
        "bfactor",
    ]
    assert voxel["pool"] == "max"
    assert voxel["clamp"] is True


def test_registry_matches_schema_v1_meta():
    encoder = get_encoder("residue_contact_graph", use_cache=False)
    info = get_encoder_info("residue_contact_graph")

    artifact = encoder.encode(StructureInput(source=str(DATA)))
    artifact.finalize()

    assert artifact.meta["encoder_name"] == info.name
    assert artifact.meta["type"] == info.representation_type
    assert artifact.meta["level"] == info.granularity
    assert artifact.meta["encoder_fingerprint"] == encoder.fingerprint()
