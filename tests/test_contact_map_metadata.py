from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eris.core.artifacts import StructureArtifact
from eris.core.registry import get_encoder
from eris.core.types import StructureInput

DATA = Path("tests/data/1AQ2.pdb")


def _encode(registry_name: str) -> StructureArtifact:
    encoder = get_encoder(registry_name)
    encoder.use_cache = False
    encoder.cache = object()
    return encoder.encode(StructureInput(source=str(DATA)))


def test_continuous_contact_map_metadata_matches_payload_and_survives_finalize():
    artifact = _encode("contact_map_residue")
    expected = {
        "matrix_key": "contact",
        "matrix_attr_keys": ["contact"],
    }

    assert "contact" in artifact.data
    assert "contact_bins" not in artifact.data
    assert artifact.data["contact"].ndim == 2
    assert artifact.data["contact"].shape[0] == artifact.data["contact"].shape[1]
    assert artifact.data["contact"].dtype == np.uint8
    assert artifact.meta["feature_keys"] == expected

    artifact.finalize()

    assert artifact.meta["feature_keys"] == expected
    assert artifact.meta["shapes"]["contact"] == list(artifact.data["contact"].shape)
    assert artifact.meta["dtypes"]["contact"] == "uint8"


def test_binned_contact_map_metadata_remains_distinct_and_consistent():
    artifact = _encode("contact_map_bins_residue")
    expected = {
        "matrix_key": "contact_bins",
        "matrix_attr_keys": ["contact_bins", "bin_edges"],
    }

    assert "contact_bins" in artifact.data
    assert "contact" not in artifact.data
    assert "bin_edges" in artifact.data
    assert artifact.data["contact_bins"].ndim == 2
    assert artifact.data["contact_bins"].shape[0] == artifact.data["contact_bins"].shape[1]
    assert artifact.data["contact_bins"].dtype == np.uint8
    assert artifact.data["bin_edges"].dtype == np.float32
    assert artifact.meta["feature_keys"] == expected

    artifact.finalize()

    assert artifact.meta["feature_keys"] == expected
    assert artifact.meta["shapes"]["contact_bins"] == list(
        artifact.data["contact_bins"].shape
    )
    assert artifact.meta["dtypes"]["contact_bins"] == "uint8"
    assert artifact.meta["dtypes"]["bin_edges"] == "float32"


@pytest.mark.parametrize(
    ("registry_name", "primary_key", "expected"),
    [
        (
            "contact_map_residue",
            "contact",
            {"matrix_key": "contact", "matrix_attr_keys": ["contact"]},
        ),
        (
            "contact_map_bins_residue",
            "contact_bins",
            {
                "matrix_key": "contact_bins",
                "matrix_attr_keys": ["contact_bins", "bin_edges"],
            },
        ),
    ],
)
def test_contact_map_metadata_survives_save_load_roundtrip(
    tmp_path,
    registry_name,
    primary_key,
    expected,
):
    artifact = _encode(registry_name)
    artifact_dir = tmp_path / registry_name

    artifact.save(artifact_dir)
    loaded = StructureArtifact.load(artifact_dir, validate=True, strict=True)

    assert primary_key in loaded.data
    assert loaded.meta["feature_keys"] == expected
