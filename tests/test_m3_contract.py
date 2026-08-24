from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eris.core.artifact_schema_v2 import compute_config_fingerprint
from eris.core.artifacts import StructureArtifact
from eris.core.hashing import sha256_file
from eris.core.registry import get_encoder, list_encoders
from eris.core.types import StructureInput

PDB = Path("tests/data/1AQ2.pdb")
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
REQUIRED_TOP_LEVEL = {
    "schema_version",
    "representation_name",
    "representation_family",
    "representation_type",
    "granularity",
    "coordinate_source",
    "encoder",
    "input",
    "parameters",
    "provenance",
    "outputs",
    "feature_keys",
    "shapes",
    "dtypes",
    "counts",
}
LEGACY_TOP_LEVEL = {
    "encoder_name",
    "encoder_version",
    "encoder_fingerprint",
    "type",
    "level",
    "params",
}
LEGACY_FEATURE_FIELDS = {
    "matrix_key",
    "matrix_attr_keys",
    "grid_key",
    "grid_layout",
}
LEGACY_COUNTS = {"n_points", "grid_c", "grid_g"}


@pytest.fixture(scope="module")
def canonical_artifacts():
    artifacts = {}
    structure_input = StructureInput(source=str(PDB))
    for name in CANONICAL_NAMES:
        encoder = get_encoder(name, use_cache=False)
        artifacts[name] = (encoder, encoder.encode(structure_input))
    return artifacts


def test_all_ten_registry_and_immediate_schema_v2_contract(canonical_artifacts):
    assert list_encoders() == CANONICAL_NAMES
    assert len(CANONICAL_NAMES) == 10

    expected_input = {
        "source": str(PDB),
        "source_sha256": sha256_file(PDB),
        "format": "pdb",
        "model_index": 0,
        "chain_id": None,
    }
    for registry_name, (encoder, artifact) in canonical_artifacts.items():
        metadata = artifact.meta
        parameters = encoder.config.to_dict()

        assert registry_name == encoder.name
        assert registry_name == encoder.config.representation_name
        assert registry_name == metadata["representation_name"]
        assert metadata["schema_version"] == 2
        assert metadata["representation_family"] == "explicit"
        assert set(metadata) == REQUIRED_TOP_LEVEL
        assert LEGACY_TOP_LEVEL.isdisjoint(metadata)
        assert metadata["parameters"] == parameters
        assert metadata["provenance"]["config_fingerprint"] == (
            compute_config_fingerprint(registry_name, parameters)
        )
        assert metadata["encoder"] == {
            "implementation": encoder.encoder_implementation,
            "version": encoder.version,
        }
        assert metadata["input"] == expected_input
        artifact.validate(strict=True)


def test_all_ten_outputs_features_counts_and_dimensions(canonical_artifacts):
    for _name, (_encoder, artifact) in canonical_artifacts.items():
        metadata = artifact.meta
        representation_type = metadata["representation_type"]
        feature_keys = metadata["feature_keys"]
        counts = metadata["counts"]
        output_names = [output["name"] for output in metadata["outputs"]]

        assert output_names == list(artifact.data)
        assert set(metadata["shapes"]) == set(output_names)
        assert set(metadata["dtypes"]) == set(output_names)
        assert LEGACY_FEATURE_FIELDS.isdisjoint(feature_keys)
        assert LEGACY_COUNTS.isdisjoint(counts)

        if representation_type == "cloud":
            assert set(feature_keys) == {"node_attr_keys"}
            assert "n_nodes" in counts
        elif representation_type == "graph":
            assert set(feature_keys) == {
                "node_attr_keys",
                "edge_attr_keys",
                "graph_attr_keys",
            }
            assert {"n_nodes", "n_edges", "n_self_loops"} <= counts.keys()
            assert len(feature_keys["edge_attr_keys"]) == 19
            assert len(feature_keys["graph_attr_keys"]) == 13
        elif representation_type == "matrix":
            assert feature_keys == {}
            assert "n_nodes" in counts
        else:
            assert representation_type == "grid"
            assert set(feature_keys) == {"grid_channels"}
            assert counts == {}
            assert len(feature_keys["grid_channels"]) == 25
            assert metadata["shapes"]["grid"][0] == 25

        if representation_type in {"cloud", "graph"}:
            expected_node_features = 25 if metadata["granularity"] == "atom" else 38
            assert len(feature_keys["node_attr_keys"]) == expected_node_features
            assert metadata["shapes"]["node_attr"][1] == expected_node_features


def test_all_ten_save_load_round_trip_exactly(canonical_artifacts, tmp_path):
    for name, (_encoder, artifact) in canonical_artifacts.items():
        output_dir = tmp_path / name

        artifact.save(output_dir)
        loaded = StructureArtifact.load(output_dir)

        artifact.validate(strict=True)
        loaded.validate(strict=True)
        assert loaded.meta["schema_version"] == 2
        assert loaded.meta == artifact.meta
        assert list(loaded.data) == list(artifact.data)
        assert loaded.meta["shapes"] == artifact.meta["shapes"]
        assert loaded.meta["dtypes"] == artifact.meta["dtypes"]
        for output_name in artifact.data:
            assert np.array_equal(loaded.data[output_name], artifact.data[output_name])
            assert loaded.data[output_name].shape == artifact.data[output_name].shape
            assert loaded.data[output_name].dtype == artifact.data[output_name].dtype


def test_schema_v2_finalize_is_idempotent(canonical_artifacts):
    artifact = canonical_artifacts["residue_point_cloud"][1]
    metadata_before = artifact.meta.copy()

    artifact.finalize()

    assert artifact.meta == metadata_before
