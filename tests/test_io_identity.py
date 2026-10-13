from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eris.core.artifacts import StructureArtifact
from eris.core.config import ResiduePointCloudConfig
from eris.core.io import StructureLoadError
from eris.core.types import StructureInput
from eris.encoders.clouds.residue_point_cloud import ResiduePointCloudCA
from eris.encoders.grids.voxel_grid_ca import VoxelGridCA
from tests.helpers import PDB, atom_line, write_pdb


def _write_missing_ca_structure(path: Path) -> StructureInput:
    write_pdb(
        path,
        [
            atom_line(1, "CA", "ALA", 15, 10.0),
            atom_line(2, "N", "GLY", 16, 12.0),
            atom_line(3, "CA", "SER", 20, 14.0, insertion_code="A"),
        ],
    )
    return StructureInput(source=str(path), chain_id="A")


def test_shared_residue_config_exposes_only_skip_and_error() -> None:
    assert ResiduePointCloudConfig().missing_residue_center == "skip"
    assert ResiduePointCloudConfig(missing_residue_center="error").missing_residue_center == "error"
    with pytest.raises(ValueError, match="missing_residue_center"):
        # pyrefly: ignore [bad-argument-type]
        ResiduePointCloudConfig(missing_residue_center="nan")


def test_residue_encoder_traces_skipped_residues(tmp_path: Path) -> None:
    structure_input = _write_missing_ca_structure(tmp_path / "missing-ca.pdb")
    artifact = ResiduePointCloudCA(missing_residue_center="skip", use_cache=False).encode(structure_input)

    assert artifact.meta["parameters"]["missing_residue_center"] == "skip"
    assert artifact.data["residue_ids"].tolist() == ["A:15::ALA", "A:20:A:SER"]
    assert artifact.data["omitted_residue_ids"].tolist() == ["A:16::GLY"]
    assert artifact.meta["counts"]["n_omitted_residues"] == 1
    assert "omitted_residue_ids" in [output["name"] for output in artifact.meta["outputs"]]
    artifact.validate()


def test_residue_encoder_fails_without_partial_artifact_for_error(tmp_path: Path) -> None:
    structure_input = _write_missing_ca_structure(tmp_path / "missing-ca.pdb")

    with pytest.raises(StructureLoadError, match=r"A:16::GLY"):
        ResiduePointCloudCA(missing_residue_center="error", use_cache=False).encode(structure_input)


def test_zero_omission_output_is_deterministic_and_round_trips(tmp_path: Path) -> None:
    source = tmp_path / "complete.pdb"
    write_pdb(source, [atom_line(1, "CA", "ALA", 1, 0.0), atom_line(2, "CA", "GLY", 2, 4.0)])
    artifact = ResiduePointCloudCA(use_cache=False).encode(StructureInput(source=str(source)))
    target = tmp_path / "artifact"
    artifact.save(target)
    loaded = StructureArtifact.load(target)

    assert artifact.data["omitted_residue_ids"].shape == (0,)
    assert artifact.data["omitted_residue_ids"].dtype.kind == "U"
    assert artifact.meta["counts"]["n_omitted_residues"] == 0
    np.testing.assert_array_equal(loaded.data["omitted_residue_ids"], artifact.data["omitted_residue_ids"])
    assert loaded.meta["counts"]["n_omitted_residues"] == 0


def test_residue_voxel_metadata_reconstructs_the_existing_transform(tmp_path: Path) -> None:
    source = tmp_path / "voxel.pdb"
    write_pdb(source, [atom_line(1, "CA", "ALA", 1, 10.0), atom_line(2, "CA", "GLY", 2, 14.0)])
    artifact = VoxelGridCA(
        grid_size=5,
        voxel_size=2.0,
        fit=False,
        channels="occupancy",
        use_cache=False,
    ).encode(StructureInput(source=str(source)))

    raw = np.asarray([[10.0, 0.0, 0.0], [14.0, 0.0, 0.0]], dtype=np.float32)
    normalized = raw - artifact.data["centroid"]
    indices = np.floor(
        (normalized - artifact.data["origin"][None, :]) / artifact.data["voxel_size"][0]
    ).astype(np.int64)

    assert artifact.meta["parameters"]["center"] is True
    assert artifact.meta["parameters"]["fit"] is False
    assert artifact.meta["parameters"]["voxel_size"] == 2.0
    assert artifact.meta["parameters"]["grid_size"] == 5
    np.testing.assert_array_equal(artifact.data["centroid"], [[12.0, 0.0, 0.0]])
    np.testing.assert_array_equal(indices, [[1, 2, 2], [3, 2, 2]])
    assert artifact.data["grid"][0, 1, 2, 2] == 1.0
    assert artifact.data["grid"][0, 3, 2, 2] == 1.0


@pytest.mark.parametrize("representation", ["residue_point_cloud", "atom_point_cloud"])
def test_encode_parses_the_structure_once_on_cache_miss(monkeypatch, representation):
    import gemmi

    import eris

    calls = []
    real_read = gemmi.read_structure

    def counting_read(*args, **kwargs):
        calls.append(args[0])
        return real_read(*args, **kwargs)

    monkeypatch.setattr(gemmi, "read_structure", counting_read)
    eris.encode(PDB, representation=representation, use_cache=False)

    assert len(calls) == 1
