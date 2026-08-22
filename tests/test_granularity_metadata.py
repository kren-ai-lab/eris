from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eris.core.artifacts import ArtifactValidationError, StructureArtifact
from eris.core.registry import get_encoder
from eris.core.types import StructureInput

DATA = Path("tests/data/1AQ2.pdb")


def _minimal_artifact(level: str) -> StructureArtifact:
    return StructureArtifact(
        data={"pos": np.zeros((1, 3), dtype=np.float32)},
        meta={
            "schema_version": 1,
            "encoder_name": "test_encoder",
            "encoder_version": "0.1.0",
            "encoder_fingerprint": "test",
            "type": "cloud",
            "level": level,
            "params": {},
            "input": {
                "source": "memory",
                "chain_id": None,
                "model_index": 0,
                "format": None,
            },
            "feature_keys": {},
        },
    )


@pytest.mark.parametrize("level", ["atom", "residue", "structure", "mixed"])
def test_schema_v1_granularity_variants_are_unsupported(level):
    with pytest.raises(ArtifactValidationError, match="Schema v1 is no longer supported"):
        _minimal_artifact(level).validate(strict=True)


def test_schema_v1_ca_level_is_unsupported_before_legacy_validation():
    artifact = _minimal_artifact("ca")

    with pytest.raises(ArtifactValidationError, match="Schema v1 is no longer supported"):
        artifact.validate(strict=True)


@pytest.mark.parametrize("mode", ["ca", "mean", "com"])
def test_residue_coordinate_source_is_separate_from_level(mode):
    encoder = get_encoder("residue_point_cloud", residue_center=mode, use_cache=False)

    artifact = encoder.encode(StructureInput(source=str(DATA), model_index=0))

    assert artifact.meta["granularity"] == "residue"
    assert artifact.meta["coordinate_source"] == ("ca" if mode == "ca" else f"residue_{mode}")
    assert artifact.meta["parameters"]["residue_center"] == mode
    assert "residue_center" not in artifact.meta
    assert artifact.meta["input"]["model_index"] == 0


def test_residue_coordinate_source_changes_fingerprint_not_level():
    ca_encoder = get_encoder("residue_point_cloud", residue_center="ca", use_cache=False)
    mean_encoder = get_encoder("residue_point_cloud", residue_center="mean", use_cache=False)

    ca_artifact = ca_encoder.encode(StructureInput(source=str(DATA)))
    mean_artifact = mean_encoder.encode(StructureInput(source=str(DATA)))

    assert ca_encoder.fingerprint() != mean_encoder.fingerprint()
    assert ca_artifact.meta["granularity"] == mean_artifact.meta["granularity"] == "residue"
    assert ca_artifact.meta["coordinate_source"] == "ca"
    assert mean_artifact.meta["coordinate_source"] == "residue_mean"


@pytest.mark.parametrize(
    ("registry_name", "expected_level"),
    [
        ("atom_point_cloud", "atom"),
        ("atom_knn_graph", "atom"),
        ("residue_point_cloud", "residue"),
        ("residue_knn_graph", "residue"),
        ("residue_contact_graph", "residue"),
        ("residue_distogram", "residue"),
        ("residue_distogram_bins", "residue"),
        ("residue_contact_map", "residue"),
        ("residue_contact_map_bins", "residue"),
        ("residue_voxel_grid", "residue"),
    ],
)
def test_explicit_encoder_level_follows_represented_entity(registry_name, expected_level):
    encoder = get_encoder(registry_name)
    encoder.use_cache = False

    artifact = encoder.encode(StructureInput(source=str(DATA)))

    assert artifact.meta["representation_name"] == registry_name
    assert artifact.meta["granularity"] == expected_level
    if expected_level == "residue":
        assert artifact.meta["parameters"]["residue_center"] == "ca"
    else:
        assert "residue_center" not in artifact.meta["parameters"]
