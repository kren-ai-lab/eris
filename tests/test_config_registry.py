from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from importlib import metadata

import pytest

from eris.core.artifact_schema_v2 import compute_config_fingerprint
from eris.core.config import (
    _CONFIG_TYPES_BY_REPRESENTATION,
    AtomKNNGraphConfig,
    AtomPointCloudConfig,
    FeatureMode,
    FeatureToken,
    ResidueContactGraphConfig,
    ResidueContactMapBinsConfig,
    ResidueContactMapConfig,
    ResidueDistogramBinsConfig,
    ResidueDistogramConfig,
    ResidueKNNGraphConfig,
    ResiduePointCloudConfig,
    ResidueVoxelGridConfig,
)
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

ATOM_SELECTION = {
    "include_hetero": False,
    "include_h": False,
    "drop_water": True,
    "keep_ions": True,
    "altloc": "max_occupancy",
    "amino_acids_only": False,
}
RESIDUE_SELECTION = {
    "residue_center": "ca",
    "amino_acids_only": True,
    "altloc": "max_occupancy",
    "missing_residue_center": "skip",
    "plddt_source": "none",
}
GEOMETRY = {"center": True, "scale": None}
ATOM_NODE_MODE = ["elem", "role", "bfactor", "occ"]
RESIDUE_NODE_MODE = ["onehot", "physchem", "bfactor_ca"]
EDGE_MODE = ["dist", "rbf", "seq", "local"]


EXPECTED_DEFAULTS = {
    "atom_point_cloud": {
        **GEOMETRY,
        **ATOM_SELECTION,
        "node_attr_mode": ATOM_NODE_MODE,
    },
    "residue_point_cloud": {
        **GEOMETRY,
        **RESIDUE_SELECTION,
        "node_attr_mode": RESIDUE_NODE_MODE,
    },
    "atom_knn_graph": {
        **GEOMETRY,
        **ATOM_SELECTION,
        "node_attr_mode": ATOM_NODE_MODE,
        "edge_attr_mode": EDGE_MODE,
        "rbf_k": 16,
        "rbf_dmax": 20.0,
        "local_cutoff": 4,
        "k": 16,
        "cutoff": 12.0,
        "symmetric": True,
    },
    "residue_knn_graph": {
        **GEOMETRY,
        **RESIDUE_SELECTION,
        "node_attr_mode": RESIDUE_NODE_MODE,
        "edge_attr_mode": EDGE_MODE,
        "rbf_k": 16,
        "rbf_dmax": 20.0,
        "local_cutoff": 4,
        "knn_mode": "local",
        "k": 16,
        "cutoff": 12.0,
        "symmetric": True,
        "include_self_loops": False,
    },
    "residue_contact_graph": {
        **GEOMETRY,
        **RESIDUE_SELECTION,
        "node_attr_mode": RESIDUE_NODE_MODE,
        "edge_attr_mode": EDGE_MODE,
        "rbf_k": 16,
        "rbf_dmax": 20.0,
        "local_cutoff": 4,
        "cutoff": 8.0,
        "include_self_loops": False,
    },
    "residue_distogram": {
        **GEOMETRY,
        **RESIDUE_SELECTION,
        "include_self": True,
    },
    "residue_distogram_bins": {
        **GEOMETRY,
        **RESIDUE_SELECTION,
        "edges": None,
        "dmin": 2.0,
        "dmax": 20.0,
        "num_bins": 36,
        "add_no_contact_class": True,
        "include_self": True,
        "dtype": "uint8",
    },
    "residue_contact_map": {
        **GEOMETRY,
        **RESIDUE_SELECTION,
        "cutoff": 8.0,
        "include_self": False,
        "dtype": "uint8",
    },
    "residue_contact_map_bins": {
        **GEOMETRY,
        **RESIDUE_SELECTION,
        "edges": None,
        "dmin": 2.0,
        "dmax": 20.0,
        "num_bins": 36,
        "add_no_contact_class": True,
        "include_self": False,
        "dtype": "uint8",
    },
    "residue_voxel_grid": {
        **GEOMETRY,
        **RESIDUE_SELECTION,
        "grid_size": 64,
        "voxel_size": None,
        "fit": True,
        "margin": 4.0,
        "channels": [
            "occupancy",
            "density",
            "aa_onehot",
            {"bfactor": {"normalization": "none"}},
        ],
        "sigma": 1.0,
        "pool": "max",
        "clamp": False,
        "dtype": "float32",
    },
}


def test_all_ten_defaults_match_the_m2_1_contract():
    assert len(_CONFIG_TYPES_BY_REPRESENTATION) == 10
    actual = {
        name: config_type().to_dict()
        for name, config_type in _CONFIG_TYPES_BY_REPRESENTATION.items()
    }
    assert actual == EXPECTED_DEFAULTS


def test_identity_is_unique_and_not_a_parameter_field():
    names = tuple(_CONFIG_TYPES_BY_REPRESENTATION)
    assert names == tuple(EXPECTED_DEFAULTS)
    assert len(set(names)) == 10

    for name, config_type in _CONFIG_TYPES_BY_REPRESENTATION.items():
        config = config_type()
        field_names = {config_field.name for config_field in fields(config)}
        assert config.representation_name == name
        assert "representation_name" not in field_names
        assert "granularity" not in field_names
        assert "representation_name" not in config.to_dict()
        assert "granularity" not in config.to_dict()
        assert not hasattr(config, "representation_family")


def test_granularity_and_coordinate_source_are_independent():
    assert AtomPointCloudConfig().granularity == "atom"
    assert AtomPointCloudConfig().coordinate_source == "atom"

    expected_sources = {
        "ca": "ca",
        "mean": "residue_mean",
        "com": "residue_com",
    }
    for residue_center, coordinate_source in expected_sources.items():
        config = ResiduePointCloudConfig(residue_center=residue_center)
        assert config.granularity == "residue"
        assert config.coordinate_source == coordinate_source


@pytest.mark.parametrize(
    "factory",
    [
        lambda: AtomKNNGraphConfig(k=True),
        lambda: AtomKNNGraphConfig(k=0),
        lambda: AtomKNNGraphConfig(cutoff=float("nan")),
        lambda: ResidueKNNGraphConfig(knn_mode="invalid"),
        lambda: ResiduePointCloudConfig(residue_center="invalid"),
        lambda: ResiduePointCloudConfig(plddt_source="automatic"),
        lambda: ResidueVoxelGridConfig(pool="invalid"),
        lambda: ResidueVoxelGridConfig(dtype="uint8"),
        lambda: ResidueVoxelGridConfig(grid_size=0),
        lambda: ResidueDistogramBinsConfig(num_bins=0),
        lambda: ResidueDistogramBinsConfig(edges=[2, 2]),
    ],
)
def test_invalid_values_fail_during_construction(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ResiduePointCloudConfig(amino_acids_only=False),
        lambda: ResiduePointCloudConfig(altloc="first"),
        lambda: ResiduePointCloudConfig(missing_residue_center="nan"),
    ],
)
def test_residue_loader_policies_remain_frozen(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_valid_non_default_edge_cases_remain_representable():
    atom = AtomPointCloudConfig(include_hetero=False, keep_ions=True, altloc=" a ")
    assert atom.include_hetero is False
    assert atom.keep_ions is True
    assert atom.altloc == "A"
    assert AtomPointCloudConfig(altloc=" all ").altloc == "all"

    local = ResidueKNNGraphConfig(knn_mode="local", cutoff=None)
    global_with_cutoff = ResidueKNNGraphConfig(knn_mode=" GLOBAL ", cutoff=99)
    assert local.cutoff is None
    assert global_with_cutoff.knn_mode == "global"
    assert global_with_cutoff.cutoff is None

    voxel = ResidueVoxelGridConfig(
        grid_size=1,
        fit=False,
        voxel_size=None,
        dtype="float16",
    )
    assert voxel.grid_size == 1
    assert voxel.voxel_size is None
    assert voxel.dtype == "float16"

    assert ResiduePointCloudConfig(residue_center="mean").coordinate_source == "residue_mean"
    assert ResiduePointCloudConfig(residue_center="com").coordinate_source == "residue_com"
    assert ResidueContactMapConfig(dtype="bool").dtype == "bool"
    assert ResidueDistogramBinsConfig(edges=[2, 4, 8]).edges == (2.0, 4.0, 8.0)


def test_feature_mode_string_list_and_tuple_forms_are_equivalent():
    string_mode = FeatureMode.from_value("density+occupancy", kind="grid")
    list_mode = FeatureMode.from_value(["density", "occupancy"], kind="grid")
    tuple_mode = FeatureMode.from_value(("density", "occupancy"), kind="grid")

    assert string_mode == list_mode == tuple_mode
    assert string_mode.to_list() == ["density", "occupancy"]


def test_feature_mode_preserves_requested_order_and_grid_parameters():
    channels = [
        {
            "density": {
                "sigma": 1.5,
                "nested": {"z": [3, 2], "a": True},
            }
        },
        "occupancy",
    ]
    config = ResidueVoxelGridConfig(channels=channels)

    assert config.to_dict()["channels"] == channels
    assert [token.name for token in config.channels.tokens] == ["density", "occupancy"]


def test_nested_feature_parameter_order_is_canonical():
    first = ResidueVoxelGridConfig(
        channels=[{"density": {"z": 1, "a": {"y": 2, "x": 3}}}],
    )
    second = ResidueVoxelGridConfig(
        channels=[{"density": {"a": {"x": 3, "y": 2}, "z": 1}}],
    )

    assert first == second
    assert first.to_dict() == second.to_dict()
    assert first.to_json() == second.to_json()


def test_parameterized_residue_features_are_canonical_and_provider_pinned():
    first = FeatureMode.from_value(
        [
            {
                "aaindex": {
                    "unknown": "nan",
                    "ids": ["kytj820101"],
                    "missing": "nan",
                    "normalization": "none",
                }
            }
        ],
        kind="residue_node",
    )
    second = FeatureMode.from_value(
        [{"aaindex": {"ids": ["KYTJ820101"]}}],
        kind="residue_node",
    )

    assert first == second
    assert first.to_list() == [
        {
            "aaindex": {
                "ids": ["KYTJ820101"],
                "missing": "nan",
                "normalization": "none",
                "provider": "aaindex",
                "provider_version": metadata.version("aaindex"),
                "source": "AAIndex1",
                "unknown": "nan",
            }
        }
    ]


def test_backbone_geometry_parameters_canonicalize_and_fingerprint_scientific_changes():
    minimal = ResiduePointCloudConfig(
        node_attr_mode=[{"backbone_geometry": {"mode": " BOTH ", "max_peptide_bond_distance": 2}}]
    )
    explicit = ResiduePointCloudConfig(
        node_attr_mode=[
            {
                "backbone_geometry": {
                    "mode": "both",
                    "max_peptide_bond_distance": 2.0,
                    "torsion_convention": "standard_signed_atan2[-pi,pi)",
                    "units": "radians",
                    "missing_value_policy": "nan",
                }
            }
        ]
    )
    changed_mode = ResiduePointCloudConfig(node_attr_mode=[{"backbone_geometry": {"mode": "raw"}}])
    changed_threshold = ResiduePointCloudConfig(
        node_attr_mode=[{"backbone_geometry": {"mode": "both", "max_peptide_bond_distance": 1.9}}]
    )

    assert minimal == explicit
    assert minimal.to_dict() == explicit.to_dict()
    baseline = compute_config_fingerprint(
        minimal.representation_name,
        minimal.to_fingerprint_dict(),
        chain_ids=["A"],
    )
    assert baseline != compute_config_fingerprint(
        changed_mode.representation_name,
        changed_mode.to_fingerprint_dict(),
        chain_ids=["A"],
    )
    assert baseline != compute_config_fingerprint(
        changed_threshold.representation_name,
        changed_threshold.to_fingerprint_dict(),
        chain_ids=["A"],
    )


def test_sidechain_geometry_parameters_canonicalize_and_fingerprint_modes():
    minimal = ResiduePointCloudConfig(node_attr_mode=["sidechain_geometry"])
    explicit = ResiduePointCloudConfig(
        node_attr_mode=[
            {
                "sidechain_geometry": {
                    "mode": " SINCOS ",
                    "torsion_convention": "standard_signed_atan2[-pi,pi)",
                    "units": "radians",
                    "atom_definition_convention": ("standard_residue_specific_chi1_through_chi4"),
                    "atom_definition_table": "eris.frozen_sidechain_chi_atoms.v1",
                    "missing_value_policy": "nan",
                }
            }
        ]
    )
    raw = ResiduePointCloudConfig(node_attr_mode=[{"sidechain_geometry": {"mode": "raw"}}])
    both = ResiduePointCloudConfig(node_attr_mode=[{"sidechain_geometry": {"mode": "both"}}])

    assert minimal == explicit
    assert minimal.to_dict() == explicit.to_dict()
    baseline = compute_config_fingerprint(
        minimal.representation_name,
        minimal.to_fingerprint_dict(),
        chain_ids=["A"],
    )
    raw_fingerprint = compute_config_fingerprint(
        raw.representation_name,
        raw.to_fingerprint_dict(),
        chain_ids=["A"],
    )
    both_fingerprint = compute_config_fingerprint(
        both.representation_name,
        both.to_fingerprint_dict(),
        chain_ids=["A"],
    )
    assert len({baseline, raw_fingerprint, both_fingerprint}) == 3


def test_secondary_structure_parameters_are_canonical_and_fingerprinted():
    minimal = ResiduePointCloudConfig(node_attr_mode=["secondary_structure"])
    explicit = ResiduePointCloudConfig(
        node_attr_mode=[
            {
                "secondary_structure": {
                    "provider": "biotite",
                    "provider_version": metadata.version("biotite"),
                    "algorithm": "P-SEA",
                    "coordinate_basis": "CA geometry",
                    "state_order": ["helix", "sheet", "coil", "unassigned"],
                    "state_mapping": {
                        "a": "helix",
                        "b": "sheet",
                        "c": "coil",
                        "": "unassigned",
                    },
                    "segment_handling": "selected_chain_and_peptide_continuity",
                    "max_peptide_bond_distance": 2.0,
                    "missing_ca_policy": "unassigned",
                }
            }
        ]
    )
    changed = ResiduePointCloudConfig(
        node_attr_mode=[{"secondary_structure": {"max_peptide_bond_distance": 1.9}}]
    )

    assert minimal == explicit
    baseline = compute_config_fingerprint(
        minimal.representation_name,
        minimal.to_fingerprint_dict(),
        chain_ids=["A"],
    )
    assert baseline != compute_config_fingerprint(
        changed.representation_name,
        changed.to_fingerprint_dict(),
        chain_ids=["A"],
    )


def test_sasa_rsa_parameters_are_frozen_and_canonical():
    config = ResiduePointCloudConfig(node_attr_mode=["sasa", "rsa"])
    sasa = config.node_attr_mode.to_list()[0]["sasa"]
    rsa = config.node_attr_mode.to_list()[1]["rsa"]

    assert sasa["provider"] == "biotite"
    assert sasa["provider_version"] == metadata.version("biotite")
    assert sasa["probe_radius_angstrom"] == 1.4
    assert sasa["sphere_point_count"] == 1000
    assert sasa["sphere_point_distribution"] == "Fibonacci"
    assert sasa["vdw_radii"] == "ProtOr"
    assert rsa["maximum_asa_table"]["ALA"] == 129.0
    assert rsa["maximum_asa_table"]["TRP"] == 285.0
    assert rsa["clipping"] == "none"

    with pytest.raises(ValueError, match="probe_radius_angstrom"):
        ResiduePointCloudConfig(node_attr_mode=[{"sasa": {"probe_radius_angstrom": 1.5}}])
    with pytest.raises(ValueError, match="clipping"):
        ResiduePointCloudConfig(node_attr_mode=[{"rsa": {"clipping": "unit_interval"}}])


def test_relative_direction_parameters_are_canonical_and_residue_graph_only():
    minimal = FeatureMode.from_value(["relative_direction"], kind="edge")
    explicit = FeatureMode.from_value(
        [
            {
                "relative_direction": {
                    "direction_convention": "source_frame_transpose_unit_ca_displacement",
                    "frame_atoms": ["N", "CA", "C"],
                    "frame_definition": "x=C-CA;y=orthogonalized(N-CA);z=x_cross_y",
                    "frame_epsilon": 1e-6,
                    "frame_handedness": "right",
                    "frame_matrix_layout": "columns_xyz_global",
                    "frame_origin": "CA",
                    "missing_frame_policy": "nan",
                }
            }
        ],
        kind="edge",
    )

    assert minimal == explicit
    minimal_config = ResidueKNNGraphConfig(edge_attr_mode=["relative_direction"])
    explicit_config = ResidueKNNGraphConfig(edge_attr_mode=explicit)
    assert minimal_config.to_fingerprint_dict() == explicit_config.to_fingerprint_dict()
    orientation_fingerprint = compute_config_fingerprint(
        minimal_config.representation_name,
        minimal_config.to_fingerprint_dict(),
        chain_ids=["A"],
    )
    distance_config = ResidueKNNGraphConfig(edge_attr_mode=["dist"])
    assert orientation_fingerprint != compute_config_fingerprint(
        distance_config.representation_name,
        distance_config.to_fingerprint_dict(),
        chain_ids=["A"],
    )
    assert ResidueKNNGraphConfig(
        edge_attr_mode=["same_chain", "inter_chain", "relative_direction"]
    ).edge_attr_mode.to_list() == [
        "same_chain",
        "inter_chain",
        minimal.to_list()[0],
    ]
    for token in ("same_chain", "inter_chain", "relative_direction"):
        with pytest.raises(ValueError, match="residue-only edge features"):
            AtomKNNGraphConfig(edge_attr_mode=[token])
    with pytest.raises(ValueError, match="frame_epsilon"):
        FeatureMode.from_value(
            [{"relative_direction": {"frame_epsilon": 1e-5}}],
            kind="edge",
        )


def test_interface_features_require_a_matched_residue_graph_pair():
    config = ResidueContactGraphConfig(
        node_attr_mode=[{"interface": {"cutoff": 4.5}}],
        edge_attr_mode=[{"interface_contact": {"cutoff": 4.5}}],
    )

    node_params = config.node_attr_mode.to_list()[0]["interface"]
    edge_params = config.edge_attr_mode.to_list()[0]["interface_contact"]
    assert node_params == edge_params
    assert node_params == {
        "atom_selection": "selected_polymer_heavy_atoms",
        "chain_relation": "different_selected_chains_only",
        "cutoff": 4.5,
        "cutoff_inclusive": True,
        "distance_definition": "minimum_inter_residue_atom_distance",
    }

    with pytest.raises(ValueError, match="requires both"):
        ResidueKNNGraphConfig(node_attr_mode=["interface"])
    with pytest.raises(ValueError, match="same cutoff"):
        ResidueKNNGraphConfig(
            node_attr_mode=[{"interface": {"cutoff": 4.0}}],
            edge_attr_mode=[{"interface_contact": {"cutoff": 5.0}}],
        )
    with pytest.raises(ValueError, match="only by residue graphs"):
        ResiduePointCloudConfig(node_attr_mode=["interface"])
    with pytest.raises(ValueError, match="residue-only edge features"):
        AtomKNNGraphConfig(edge_attr_mode=["interface_contact"])


@pytest.mark.parametrize(
    "value",
    [
        [{"onehot": {"invalid": True}}],
        [{"aaindex": {}}],
        [{"aaindex": {"ids": []}}],
        [{"aaindex": {"ids": ["KYTJ820101", "kytj820101"]}}],
        [{"aaindex": {"ids": ["NOT_A_REAL_ID"]}}],
        [{"aaindex": {"ids": ["KYTJ820101"], "extra": True}}],
        [{"backbone_geometry": {"mode": "degrees"}}],
        [{"backbone_geometry": {"max_peptide_bond_distance": 0.0}}],
        [{"backbone_geometry": {"missing_value_policy": "zero"}}],
        [{"sidechain_geometry": {"mode": "degrees"}}],
        [{"sidechain_geometry": {"missing_value_policy": "zero"}}],
        [{"sidechain_geometry": {"atom_definition_table": "other"}}],
    ],
)
def test_residue_feature_parameters_fail_during_config_construction(value):
    with pytest.raises((TypeError, ValueError)):
        FeatureMode.from_value(value, kind="residue_node")


@pytest.mark.parametrize(
    ("value", "kind"),
    [
        ("occupancy+occupancy", "grid"),
        ([{"dist": {"invalid": True}}], "edge"),
    ],
)
def test_feature_mode_rejects_duplicates_unknowns_and_unsupported_params(value, kind):
    with pytest.raises(ValueError):
        FeatureMode.from_value(value, kind=kind)


def test_explicit_edges_override_generation_fields_canonically():
    with_ignored_generation = ResidueDistogramBinsConfig(
        edges=[2, 4, 8],
        dmin=-100,
        dmax=-99,
        num_bins=99,
    )
    canonical = ResidueDistogramBinsConfig(edges=(2.0, 4.0, 8.0))

    assert with_ignored_generation == canonical
    assert with_ignored_generation.to_dict() == canonical.to_dict()
    assert with_ignored_generation.to_json() == canonical.to_json()
    assert canonical.to_dict()["edges"] == [2.0, 4.0, 8.0]
    assert canonical.dmin is canonical.dmax is canonical.num_bins is None


def test_configs_and_nested_state_are_immutable():
    config = ResidueDistogramBinsConfig(edges=[2, 4, 8])
    mode = ResidueVoxelGridConfig(
        channels=[{"density": {"nested": [1, 2]}}],
    ).channels

    with pytest.raises(FrozenInstanceError):
        config.edges = (1.0, 2.0)
    with pytest.raises(TypeError):
        config.edges[0] = 1.0
    with pytest.raises(FrozenInstanceError):
        mode.tokens = ()
    with pytest.raises(FrozenInstanceError):
        mode.tokens[0].params.items = ()


def test_direct_feature_construction_cannot_retain_mutable_containers():
    raw_params = {"nested": [1, 2]}
    token = FeatureToken(name=" density ", params=raw_params)
    mode = FeatureMode(tokens=[token])
    raw_params["nested"].append(3)

    assert token.name == "density"
    assert mode.tokens == (token,)
    assert token.to_value() == {"density": {"nested": [1, 2]}}


def test_catalog_is_immutable():
    with pytest.raises(TypeError):
        _CONFIG_TYPES_BY_REPRESENTATION["extra"] = AtomPointCloudConfig


def test_serialization_preserves_none_and_excludes_non_parameter_state():
    config = ResidueVoxelGridConfig()
    payload = config.to_dict()

    assert payload["scale"] is None
    assert payload["voxel_size"] is None
    assert config.to_json() == config.to_json()
    for excluded in (
        "use_cache",
        "cache",
        "name",
        "representation",
        "granularity_contract",
        "representation_name",
        "granularity",
        "coordinate_source",
        "voxel_size_effective",
    ):
        assert excluded not in payload


def _plain_value(value):
    if isinstance(value, FeatureMode):
        return "+".join(token.name for token in value.tokens)
    return value


@pytest.mark.parametrize(
    ("config", "encoder", "connected_fields"),
    [
        (
            AtomPointCloudConfig(),
            AtomPointCloud(),
            (*GEOMETRY, *ATOM_SELECTION, "node_attr_mode"),
        ),
        (
            ResiduePointCloudConfig(),
            ResiduePointCloudCA(),
            (*GEOMETRY, "residue_center", "node_attr_mode"),
        ),
        (
            AtomKNNGraphConfig(),
            AtomKNNGraph(),
            (
                *GEOMETRY,
                *ATOM_SELECTION,
                "node_attr_mode",
                "edge_attr_mode",
                "rbf_k",
                "rbf_dmax",
                "local_cutoff",
                "k",
                "cutoff",
                "symmetric",
            ),
        ),
        (
            ResidueKNNGraphConfig(),
            ResidueKNNGraph(),
            (
                *GEOMETRY,
                "residue_center",
                "node_attr_mode",
                "edge_attr_mode",
                "rbf_k",
                "rbf_dmax",
                "local_cutoff",
                "knn_mode",
                "k",
                "cutoff",
                "symmetric",
                "include_self_loops",
            ),
        ),
        (
            ResidueContactGraphConfig(),
            ResidueContactGraph(),
            (
                *GEOMETRY,
                "residue_center",
                "node_attr_mode",
                "edge_attr_mode",
                "rbf_k",
                "rbf_dmax",
                "local_cutoff",
                "cutoff",
                "include_self_loops",
            ),
        ),
        (
            ResidueDistogramConfig(),
            DistogramCA(),
            (*GEOMETRY, "residue_center", "include_self"),
        ),
        (
            ResidueDistogramBinsConfig(),
            DistogramBinsCA(),
            (
                *GEOMETRY,
                "residue_center",
                "edges",
                "dmin",
                "dmax",
                "num_bins",
                "add_no_contact_class",
                "include_self",
                "dtype",
            ),
        ),
        (
            ResidueContactMapConfig(),
            ContactMapCA(),
            (*GEOMETRY, "residue_center", "cutoff", "include_self", "dtype"),
        ),
        (
            ResidueContactMapBinsConfig(),
            ContactMapBinsCA(),
            (
                *GEOMETRY,
                "residue_center",
                "edges",
                "dmin",
                "dmax",
                "num_bins",
                "add_no_contact_class",
                "include_self",
                "dtype",
            ),
        ),
        (
            ResidueVoxelGridConfig(),
            VoxelGridCA(),
            (
                *GEOMETRY,
                "residue_center",
                "grid_size",
                "voxel_size",
                "fit",
                "margin",
                "channels",
                "sigma",
                "pool",
                "clamp",
                "dtype",
            ),
        ),
    ],
)
def test_config_defaults_match_current_connected_encoder_fields(
    config,
    encoder,
    connected_fields,
):
    for field_name in connected_fields:
        assert _plain_value(getattr(config, field_name)) == getattr(encoder, field_name)


def test_config_does_not_copy_current_identity_field_leaks():
    assert "name" not in {item.name for item in fields(ResidueKNNGraphConfig)}
    assert "name" not in {item.name for item in fields(ResidueVoxelGridConfig)}
    assert "representation" not in {item.name for item in fields(ResidueContactGraphConfig)}


def test_atom_knn_edge_mode_is_canonical_config_despite_runtime_class_attribute():
    config = AtomKNNGraphConfig()
    encoder = AtomKNNGraph()

    assert config.edge_attr_mode.to_list() == ["dist", "rbf", "seq", "local"]
    assert "edge_attr_mode" not in encoder.__dict__
    assert encoder.edge_attr_mode == "dist+rbf+seq+local"
