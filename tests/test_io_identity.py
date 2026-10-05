from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from eris.core.artifact_schema_v2 import compute_config_fingerprint
from eris.core.artifacts import StructureArtifact
from eris.core.cache import DiskCache, compute_cache_key_v2
from eris.core.config import ResiduePointCloudConfig
from eris.core.hashing import sha256_file
from eris.core.io import StructureLoadError
from eris.core.types import StructureInput
from eris.encoders.clouds.residue_point_cloud import ResiduePointCloudCA
from eris.encoders.grids.voxel_grid_ca import VoxelGridCA


def _atom_line(
    serial: int,
    atom_name: str,
    residue_name: str,
    residue_number: int,
    x: float,
    *,
    insertion_code: str = " ",
) -> str:
    return (
        f"ATOM  {serial:5d} {atom_name:>4s} {residue_name:>3s} A"
        f"{residue_number:4d}{insertion_code}   {x:8.3f}{0.0:8.3f}{0.0:8.3f}"
        f"{1.0:6.2f}{10.0:6.2f}           C  "
    )


def _write_missing_ca_structure(path: Path) -> StructureInput:
    path.write_text(
        "\n".join(
            (
                _atom_line(1, "CA", "ALA", 15, 10.0),
                _atom_line(2, "N", "GLY", 16, 12.0),
                _atom_line(3, "CA", "SER", 20, 14.0, insertion_code="A"),
                "TER",
                "END",
            )
        )
        + "\n",
        encoding="ascii",
    )
    return StructureInput(source=str(path), chain_id="A")


def test_shared_residue_config_exposes_only_skip_and_error() -> None:
    assert ResiduePointCloudConfig().missing_residue_center == "skip"
    assert ResiduePointCloudConfig(missing_residue_center="error").missing_residue_center == "error"
    with pytest.raises(ValueError, match="missing_residue_center"):
        ResiduePointCloudConfig(missing_residue_center="nan")


def test_residue_encoder_traces_skipped_residues(tmp_path: Path) -> None:
    structure_input = _write_missing_ca_structure(tmp_path / "missing-ca.pdb")
    artifact = ResiduePointCloudCA(missing_residue_center="skip", use_cache=False).encode(
        structure_input
    )

    assert ResiduePointCloudCA.version == "0.1.2"
    assert artifact.meta["parameters"]["missing_residue_center"] == "skip"
    assert artifact.data["residue_ids"].tolist() == ["A:15::ALA", "A:20:A:SER"]
    assert artifact.data["omitted_residue_ids"].tolist() == ["A:16::GLY"]
    assert artifact.meta["counts"]["n_omitted_residues"] == 1
    assert "omitted_residue_ids" in [output["name"] for output in artifact.meta["outputs"]]
    artifact.validate(strict=True)


def test_residue_encoder_fails_without_partial_artifact_for_error(tmp_path: Path) -> None:
    structure_input = _write_missing_ca_structure(tmp_path / "missing-ca.pdb")

    with pytest.raises(StructureLoadError, match=r"A:16::GLY"):
        ResiduePointCloudCA(missing_residue_center="error", use_cache=False).encode(structure_input)


def test_zero_omission_output_is_deterministic_and_round_trips(tmp_path: Path) -> None:
    source = tmp_path / "complete.pdb"
    source.write_text(
        "\n".join(
            (
                _atom_line(1, "CA", "ALA", 1, 0.0),
                _atom_line(2, "CA", "GLY", 2, 4.0),
                "TER",
                "END",
            )
        )
        + "\n",
        encoding="ascii",
    )
    artifact = ResiduePointCloudCA(use_cache=False).encode(StructureInput(source=str(source)))
    target = tmp_path / "artifact"
    artifact.save(target)
    loaded = StructureArtifact.load(target, strict=True)

    assert artifact.data["omitted_residue_ids"].shape == (0,)
    assert artifact.data["omitted_residue_ids"].dtype.kind == "U"
    assert artifact.meta["counts"]["n_omitted_residues"] == 0
    np.testing.assert_array_equal(
        loaded.data["omitted_residue_ids"], artifact.data["omitted_residue_ids"]
    )
    assert loaded.meta["counts"]["n_omitted_residues"] == 0


def test_residue_voxel_metadata_reconstructs_the_existing_transform(tmp_path: Path) -> None:
    source = tmp_path / "voxel.pdb"
    source.write_text(
        "\n".join(
            (
                _atom_line(1, "CA", "ALA", 1, 10.0),
                _atom_line(2, "CA", "GLY", 2, 14.0),
                "TER",
                "END",
            )
        )
        + "\n",
        encoding="ascii",
    )
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


def test_old_residue_cache_key_cannot_satisfy_new_output_contract(tmp_path: Path) -> None:
    structure_input = _write_missing_ca_structure(tmp_path / "missing-ca.pdb")
    cache = DiskCache(tmp_path / "cache")
    current = ResiduePointCloudCA(use_cache=False).encode(structure_input)
    stale_meta = deepcopy(current.meta)
    stale_data = dict(current.data)
    stale_data.pop("omitted_residue_ids")
    stale_meta["encoder"]["version"] = "0.1.0"
    stale_meta["outputs"] = [
        output for output in stale_meta["outputs"] if output["name"] != "omitted_residue_ids"
    ]
    stale_meta["shapes"].pop("omitted_residue_ids")
    stale_meta["dtypes"].pop("omitted_residue_ids")
    stale_meta["counts"].pop("n_omitted_residues")
    stale = StructureArtifact(data=stale_data, meta=stale_meta)
    stale.validate(strict=True)

    encoder = ResiduePointCloudCA(use_cache=True, cache=cache)
    parameters = encoder.config.to_dict()
    old_key = compute_cache_key_v2(
        representation_name=encoder.name,
        encoder_implementation=encoder.encoder_implementation,
        encoder_version="0.1.0",
        config_fingerprint=compute_config_fingerprint(
            encoder.name, parameters, chain_ids=structure_input.chain_id
        ),
        source_sha256=sha256_file(Path(structure_input.source)),
        parser_hint=".pdb",
        model_index=structure_input.model_index,
        chain_ids=structure_input.chain_id,
    )
    current_key = compute_cache_key_v2(
        representation_name=encoder.name,
        encoder_implementation=encoder.encoder_implementation,
        encoder_version=encoder.version,
        config_fingerprint=compute_config_fingerprint(
            encoder.name, parameters, chain_ids=structure_input.chain_id
        ),
        source_sha256=sha256_file(Path(structure_input.source)),
        parser_hint=".pdb",
        model_index=structure_input.model_index,
        chain_ids=structure_input.chain_id,
    )
    cache.set_v2(old_key, stale)

    assert encoder.version == "0.1.2"
    assert old_key != current_key
    assert cache.get_v2(old_key) is not None
    result = encoder.encode(structure_input)
    assert result.data["omitted_residue_ids"].tolist() == ["A:16::GLY"]
    assert cache.get_v2(current_key) is not None
