from __future__ import annotations

from dataclasses import FrozenInstanceError
from importlib import metadata

import pytest

from eris.core.artifact_schema import canonical_json, compute_config_fingerprint
from eris.core.config import (
    _CONFIG_TYPES_BY_REPRESENTATION,
    AtomKNNGraphConfig,
    AtomPointCloudConfig,
    FeatureMode,
    FeatureToken,
    ResidueContactGraphConfig,
    ResidueContactMapConfig,
    ResidueDistogramBinsConfig,
    ResidueKNNGraphConfig,
    ResiduePointCloudConfig,
    ResidueVoxelGridConfig,
)


def test_each_registered_config_identifies_its_representation_outside_parameters():
    for name, config_type in _CONFIG_TYPES_BY_REPRESENTATION.items():
        config = config_type()
        assert config.representation_name == name
        assert "representation_name" not in config.to_dict()
        assert "granularity" not in config.to_dict()


def test_granularity_and_coordinate_source_are_independent():
    assert AtomPointCloudConfig().granularity == "atom"
    assert AtomPointCloudConfig().coordinate_source == "atom"

    expected_sources = {
        "ca": "ca",
        "mean": "residue_mean",
        "com": "residue_com",
    }
    for residue_center, coordinate_source in expected_sources.items():
        # pyrefly: ignore [bad-argument-type]
        config = ResiduePointCloudConfig(residue_center=residue_center)
        assert config.granularity == "residue"
        assert config.coordinate_source == coordinate_source


@pytest.mark.parametrize(
    ("factory", "error", "match"),
    [
        (lambda: AtomKNNGraphConfig(k=True), TypeError, "k must be an int"),
        (lambda: AtomKNNGraphConfig(k=0), ValueError, "k must be > 0"),
        (lambda: AtomKNNGraphConfig(cutoff=float("nan")), ValueError, "cutoff must be finite"),
        # pyrefly: ignore [bad-argument-type]
        (lambda: ResidueKNNGraphConfig(knn_mode="invalid"), ValueError, "knn_mode must be one of"),
        # pyrefly: ignore [bad-argument-type]
        (lambda: ResidueDistogramBinsConfig(edges=[2, 2]), ValueError, "strictly increasing"),
        # pyrefly: ignore [bad-argument-type]
        (lambda: ResiduePointCloudConfig(altloc="first"), ValueError, "altloc must be one of"),
        # pyrefly: ignore [bad-argument-type]
        (lambda: ResiduePointCloudConfig(amino_acids_only=False), ValueError, "amino_acids_only must remain"),
    ],
)
def test_invalid_values_fail_during_construction(factory, error, match):
    with pytest.raises(error, match=match):
        factory()


def test_valid_non_default_edge_cases_remain_representable():
    atom = AtomPointCloudConfig(include_hetero=False, keep_ions=True, altloc=" a ")
    assert atom.include_hetero is False
    assert atom.keep_ions is True
    assert atom.altloc == "A"
    assert AtomPointCloudConfig(altloc=" all ").altloc == "all"

    local = ResidueKNNGraphConfig(knn_mode="local", cutoff=None)
    # pyrefly: ignore [bad-argument-type]
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
    # pyrefly: ignore [bad-argument-type]
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
    # pyrefly: ignore [bad-argument-type]
    config = ResidueVoxelGridConfig(channels=channels)

    assert config.to_dict()["channels"] == channels
    assert [token.name for token in config.channels.tokens] == ["density", "occupancy"]


def test_nested_feature_parameter_order_is_canonical():
    first = ResidueVoxelGridConfig(
        # pyrefly: ignore [bad-argument-type]
        channels=[{"density": {"z": 1, "a": {"y": 2, "x": 3}}}],
    )
    second = ResidueVoxelGridConfig(
        # pyrefly: ignore [bad-argument-type]
        channels=[{"density": {"a": {"x": 3, "y": 2}, "z": 1}}],
    )

    assert first == second
    assert first.to_dict() == second.to_dict()
    assert canonical_json(first.to_dict()) == canonical_json(second.to_dict())


def test_parameterized_residue_features_are_canonical_and_record_provider_version():
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
    assert first.to_list()[0]["aaindex"]["provider_version"] == metadata.version("aaindex")


def test_backbone_geometry_parameters_canonicalize_and_fingerprint_scientific_changes():
    minimal = ResiduePointCloudConfig(
        # pyrefly: ignore [bad-argument-type]
        node_attr_mode=[{"backbone_geometry": {"mode": " BOTH ", "max_peptide_bond_distance": 2}}]
    )
    explicit = ResiduePointCloudConfig(
        # pyrefly: ignore [bad-argument-type]
        node_attr_mode=[{"backbone_geometry": {"mode": "both", "max_peptide_bond_distance": 2.0}}]
    )
    # pyrefly: ignore [bad-argument-type]
    changed_mode = ResiduePointCloudConfig(node_attr_mode=[{"backbone_geometry": {"mode": "raw"}}])
    changed_threshold = ResiduePointCloudConfig(
        # pyrefly: ignore [bad-argument-type]
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
    # pyrefly: ignore [bad-argument-type]
    minimal = ResiduePointCloudConfig(node_attr_mode=["sidechain_geometry"])
    # pyrefly: ignore [bad-argument-type]
    explicit = ResiduePointCloudConfig(node_attr_mode=[{"sidechain_geometry": {"mode": " SINCOS "}}])
    # pyrefly: ignore [bad-argument-type]
    raw = ResiduePointCloudConfig(node_attr_mode=[{"sidechain_geometry": {"mode": "raw"}}])
    # pyrefly: ignore [bad-argument-type]
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
    # pyrefly: ignore [bad-argument-type]
    minimal = ResiduePointCloudConfig(node_attr_mode=["secondary_structure"])
    explicit = ResiduePointCloudConfig(
        # pyrefly: ignore [bad-argument-type]
        node_attr_mode=[{"secondary_structure": {"max_peptide_bond_distance": 2}}]
    )
    changed = ResiduePointCloudConfig(
        # pyrefly: ignore [bad-argument-type]
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


def test_sasa_rsa_parameters_reject_unsupported_values():
    with pytest.raises(ValueError, match="probe_radius_angstrom"):
        # pyrefly: ignore [bad-argument-type]
        ResiduePointCloudConfig(node_attr_mode=[{"sasa": {"probe_radius_angstrom": 1.5}}])
    with pytest.raises(ValueError, match="clipping"):
        # pyrefly: ignore [bad-argument-type]
        ResiduePointCloudConfig(node_attr_mode=[{"rsa": {"clipping": "unit_interval"}}])


def test_relative_direction_parameters_are_canonical_and_residue_graph_only():
    minimal = FeatureMode.from_value(["relative_direction"], kind="edge")
    explicit = FeatureMode.from_value([{"relative_direction": {"frame_epsilon": 1e-6}}], kind="edge")

    assert minimal == explicit
    # pyrefly: ignore [bad-argument-type]
    minimal_config = ResidueKNNGraphConfig(edge_attr_mode=["relative_direction"])
    explicit_config = ResidueKNNGraphConfig(edge_attr_mode=explicit)
    assert minimal_config.to_fingerprint_dict() == explicit_config.to_fingerprint_dict()
    orientation_fingerprint = compute_config_fingerprint(
        minimal_config.representation_name,
        minimal_config.to_fingerprint_dict(),
        chain_ids=["A"],
    )
    # pyrefly: ignore [bad-argument-type]
    distance_config = ResidueKNNGraphConfig(edge_attr_mode=["dist"])
    assert orientation_fingerprint != compute_config_fingerprint(
        distance_config.representation_name,
        distance_config.to_fingerprint_dict(),
        chain_ids=["A"],
    )
    assert ResidueKNNGraphConfig(
        edge_attr_mode=["same_chain", "inter_chain", "relative_direction"]  # pyrefly: ignore [bad-argument-type]
    ).edge_attr_mode.to_list() == [
        "same_chain",
        "inter_chain",
        minimal.to_list()[0],
    ]
    with pytest.raises(ValueError, match="residue-only edge features"):
        # pyrefly: ignore [bad-argument-type]
        AtomKNNGraphConfig(edge_attr_mode=["relative_direction"])
    with pytest.raises(ValueError, match="frame_epsilon"):
        FeatureMode.from_value(
            [{"relative_direction": {"frame_epsilon": 1e-5}}],
            kind="edge",
        )


def test_interface_features_require_a_matched_residue_graph_pair():
    config = ResidueContactGraphConfig(
        # pyrefly: ignore [bad-argument-type]
        node_attr_mode=[{"interface": {"cutoff": 4.5}}],
        # pyrefly: ignore [bad-argument-type]
        edge_attr_mode=[{"interface_contact": {"cutoff": 4.5}}],
    )

    node_params = config.node_attr_mode.to_list()[0]["interface"]
    edge_params = config.edge_attr_mode.to_list()[0]["interface_contact"]
    assert node_params == edge_params
    assert node_params["cutoff"] == 4.5

    with pytest.raises(ValueError, match="requires both"):
        # pyrefly: ignore [bad-argument-type]
        ResidueKNNGraphConfig(node_attr_mode=["interface"])
    with pytest.raises(ValueError, match="same cutoff"):
        ResidueKNNGraphConfig(
            # pyrefly: ignore [bad-argument-type]
            node_attr_mode=[{"interface": {"cutoff": 4.0}}],
            # pyrefly: ignore [bad-argument-type]
            edge_attr_mode=[{"interface_contact": {"cutoff": 5.0}}],
        )
    with pytest.raises(ValueError, match="only by residue graphs"):
        # pyrefly: ignore [bad-argument-type]
        ResiduePointCloudConfig(node_attr_mode=["interface"])
    with pytest.raises(ValueError, match="residue-only edge features"):
        # pyrefly: ignore [bad-argument-type]
        AtomKNNGraphConfig(edge_attr_mode=["interface_contact"])


@pytest.mark.parametrize(
    ("value", "kind", "match"),
    [
        ([{"onehot": {"invalid": True}}], "residue_node", "does not support parameters"),
        ([{"aaindex": {"ids": ["NOT_A_REAL_ID"]}}], "residue_node", "Unknown AAIndex1 accession"),
        ([{"aaindex": {"ids": ["KYTJ820101", "kytj820101"]}}], "residue_node", "Duplicate AAIndex1"),
        ([{"aaindex": {"ids": ["KYTJ820101"], "extra": True}}], "residue_node", "Unknown aaindex parameters"),
        ([{"aaindex": {}}], "residue_node", "'ids' is required"),
        ([{"aaindex": {"ids": []}}], "residue_node", "'ids' must not be empty"),
        ([{"backbone_geometry": {"mode": "degrees"}}], "residue_node", "'mode' must be one of"),
        ([{"backbone_geometry": {"max_peptide_bond_distance": 0.0}}], "residue_node", "finite positive"),
        ([{"backbone_geometry": {"missing_value_policy": "zero"}}], "residue_node", "must be 'nan'"),
        ([{"sidechain_geometry": {"mode": "degrees"}}], "residue_node", "'mode' must be one of"),
        (
            [{"sidechain_geometry": {"atom_definition_table": "other"}}],
            "residue_node",
            "atom_definition_table",
        ),
        ("occupancy+occupancy", "grid", "duplicate grid feature tokens"),
    ],
)
def test_feature_parameters_fail_during_config_construction(value, kind, match):
    with pytest.raises(ValueError, match=match):
        FeatureMode.from_value(value, kind=kind)


def test_explicit_edges_override_generation_fields_canonically():
    with_ignored_generation = ResidueDistogramBinsConfig(
        # pyrefly: ignore [bad-argument-type]
        edges=[2, 4, 8],
        dmin=-100,
        dmax=-99,
        num_bins=99,
    )
    canonical = ResidueDistogramBinsConfig(edges=(2.0, 4.0, 8.0))

    assert with_ignored_generation == canonical
    assert with_ignored_generation.to_dict() == canonical.to_dict()
    assert canonical_json(with_ignored_generation.to_dict()) == canonical_json(canonical.to_dict())
    assert canonical.to_dict()["edges"] == [2.0, 4.0, 8.0]
    assert canonical.dmin is canonical.dmax is canonical.num_bins is None


def test_configs_and_nested_state_are_immutable():
    # pyrefly: ignore [bad-argument-type]
    config = ResidueDistogramBinsConfig(edges=[2, 4, 8])
    mode = ResidueVoxelGridConfig(
        # pyrefly: ignore [bad-argument-type]
        channels=[{"density": {"nested": [1, 2]}}],
    ).channels

    with pytest.raises(FrozenInstanceError):
        # pyrefly: ignore [read-only]
        config.edges = (1.0, 2.0)
    with pytest.raises(TypeError):
        # pyrefly: ignore [unsupported-operation]
        config.edges[0] = 1.0
    with pytest.raises(FrozenInstanceError):
        # pyrefly: ignore [read-only]
        mode.tokens = ()
    with pytest.raises(FrozenInstanceError):
        # pyrefly: ignore [read-only]
        mode.tokens[0].params.items = ()


def test_direct_feature_construction_cannot_retain_mutable_containers():
    raw_params = {"nested": [1, 2]}
    # pyrefly: ignore [bad-argument-type]
    token = FeatureToken(name=" density ", params=raw_params)
    # pyrefly: ignore [bad-argument-type]
    mode = FeatureMode(tokens=[token])
    raw_params["nested"].append(3)

    assert token.name == "density"
    assert mode.tokens == (token,)
    assert token.to_value() == {"density": {"nested": [1, 2]}}
