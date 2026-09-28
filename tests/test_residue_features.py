from __future__ import annotations

from importlib import metadata
from pathlib import Path

import numpy as np
import pytest

from eris.core.artifacts import StructureArtifact
from eris.core.config import ResiduePointCloudConfig, ResidueVoxelGridConfig
from eris.core.io import StructureLoadError, load_and_extract_residue
from eris.core.types import StructureInput
from eris.encoders.clouds.residue_point_cloud import ResiduePointCloudCA
from eris.encoders.grids.voxel_grid_ca import VoxelGridCA
from eris.features.node.constants import AA_VOCAB
from eris.features.node.residue import build_node_attr


def _write_ca_structure(
    path: Path,
    residues: list[tuple[str, float]],
) -> StructureInput:
    lines = []
    for serial, (resname, bfactor) in enumerate(residues, start=1):
        lines.append(
            f"ATOM  {serial:5d}  CA  {resname:>3s} A{serial:4d}    "
            f"{float(serial - 1):8.3f}{0.0:8.3f}{0.0:8.3f}"
            f"{1.0:6.2f}{bfactor:6.2f}           C  "
        )
    path.write_text("\n".join([*lines, "TER", "END", ""]), encoding="ascii")
    return StructureInput(source=str(path), chain_id="A")


def _aa_indices(sequence: str) -> np.ndarray:
    return np.asarray([AA_VOCAB.index(letter) for letter in sequence], dtype=np.int16)


def _aaindex_mode(
    ids: list[str],
    *,
    unknown: str = "nan",
    missing: str = "nan",
) -> list[dict[str, dict[str, object]]]:
    return [
        {
            "aaindex": {
                "ids": ids,
                "normalization": "none",
                "unknown": unknown,
                "missing": missing,
            }
        }
    ]


def test_bfactor_is_not_automatically_interpreted_as_plddt(tmp_path: Path) -> None:
    inp = _write_ca_structure(tmp_path / "experimental.pdb", [("ALA", 42.0)])

    loaded = load_and_extract_residue(inp)

    np.testing.assert_array_equal(loaded["bfactor_ca"], np.asarray([42.0], dtype=np.float32))
    assert np.isnan(loaded["plddt"]).all()


def test_explicit_bfactor_plddt_source_is_raw_and_serialized(tmp_path: Path) -> None:
    inp = _write_ca_structure(tmp_path / "predicted.pdb", [("ALA", 87.5)])
    config = ResiduePointCloudConfig(
        plddt_source="bfactor",
        node_attr_mode=["plddt_ca", "bfactor_ca"],
    )

    artifact = ResiduePointCloudCA(config=config, use_cache=False).encode(inp)

    np.testing.assert_array_equal(artifact.data["plddt"], np.asarray([87.5], dtype=np.float32))
    np.testing.assert_array_equal(artifact.data["bfactor_ca"], np.asarray([87.5], dtype=np.float32))
    np.testing.assert_array_equal(
        artifact.data["node_attr"], np.asarray([[87.5, 87.5]], dtype=np.float32)
    )
    assert artifact.meta["parameters"]["plddt_source"] == "bfactor"
    assert artifact.meta["parameters"]["node_attr_mode"][0] == {
        "plddt_ca": {"normalization": "none"}
    }


def test_explicit_bfactor_plddt_source_rejects_out_of_range_value(tmp_path: Path) -> None:
    inp = _write_ca_structure(tmp_path / "invalid-confidence.pdb", [("ALA", 100.01)])

    with pytest.raises(StructureLoadError, match=r"within \[0, 100\].*100.01"):
        load_and_extract_residue(inp, plddt_source="bfactor")


def test_plddt_and_bfactor_feature_normalization_is_explicit() -> None:
    aa_index = _aa_indices("AI")
    plddt = np.asarray([25.0, np.nan], dtype=np.float32)
    bfactor = np.asarray([25.0, 250.0], dtype=np.float32)

    raw, raw_keys = build_node_attr(
        aa_index,
        plddt_ca=plddt,
        bfactor_ca=bfactor,
        mode=["plddt_ca", "bfactor_ca"],
        missing="nan",
        return_keys=True,
    )
    unit = build_node_attr(
        aa_index,
        plddt_ca=plddt,
        mode=[{"plddt_ca": {"normalization": "unit_interval"}}],
        missing="nan",
    )

    assert raw_keys == ["plddt_ca", "bfactor_ca"]
    np.testing.assert_equal(raw[:, 0], np.asarray([25.0, np.nan], dtype=np.float32))
    np.testing.assert_equal(raw[:, 1], bfactor)
    np.testing.assert_equal(unit[:, 0], np.asarray([0.25, np.nan], dtype=np.float32))


def test_default_residue_and_voxel_features_do_not_request_plddt() -> None:
    residue = ResiduePointCloudConfig().to_dict()
    voxel = ResidueVoxelGridConfig().to_dict()

    assert residue["plddt_source"] == "none"
    assert "plddt_ca" not in residue["node_attr_mode"]
    assert all(channel != "plddt" for channel in voxel["channels"])
    assert voxel["clamp"] is False


def test_direct_builder_cannot_bypass_feature_parameter_validation() -> None:
    with pytest.raises(ValueError, match="does not support parameters"):
        build_node_attr(
            _aa_indices("A"),
            mode=[{"onehot": {"invalid": True}}],
        )


def test_voxel_bfactor_is_raw_unless_normalization_is_requested(tmp_path: Path) -> None:
    inp = _write_ca_structure(tmp_path / "voxel.pdb", [("ALA", 42.0)])
    raw = VoxelGridCA(
        config=ResidueVoxelGridConfig(
            grid_size=3,
            fit=False,
            voxel_size=1.0,
            channels=["bfactor"],
        ),
        use_cache=False,
    ).encode(inp)
    unit = VoxelGridCA(
        config=ResidueVoxelGridConfig(
            grid_size=3,
            fit=False,
            voxel_size=1.0,
            channels=[{"bfactor": {"normalization": "unit_interval"}}],
        ),
        use_cache=False,
    ).encode(inp)

    assert float(raw.data["grid"].max()) == pytest.approx(42.0)
    assert float(unit.data["grid"].max()) == pytest.approx(0.42)


def test_voxel_plddt_normalization_is_explicit(tmp_path: Path) -> None:
    inp = _write_ca_structure(tmp_path / "confidence-voxel.pdb", [("ALA", 42.0)])
    raw = VoxelGridCA(
        config=ResidueVoxelGridConfig(
            plddt_source="bfactor",
            grid_size=3,
            fit=False,
            voxel_size=1.0,
            channels=["plddt"],
        ),
        use_cache=False,
    ).encode(inp)
    unit = VoxelGridCA(
        config=ResidueVoxelGridConfig(
            plddt_source="bfactor",
            grid_size=3,
            fit=False,
            voxel_size=1.0,
            channels=[{"plddt": {"normalization": "unit_interval"}}],
        ),
        use_cache=False,
    ).encode(inp)

    assert float(raw.data["grid"].max()) == pytest.approx(42.0)
    assert float(unit.data["grid"].max()) == pytest.approx(0.42)


def test_aaindex_known_values_shape_dtype_and_keys() -> None:
    features, keys = build_node_attr(
        _aa_indices("ADI"),
        mode=_aaindex_mode(["KYTJ820101"]),
        return_keys=True,
    )

    assert features.shape == (3, 1)
    assert features.dtype == np.float32
    assert keys == ["aaindex.KYTJ820101"]
    np.testing.assert_array_equal(features[:, 0], np.asarray([1.8, -3.5, 4.5], np.float32))


def test_aaindex_preserves_multiple_accession_order() -> None:
    features, keys = build_node_attr(
        _aa_indices("AD"),
        mode=_aaindex_mode(["HOPT810101", "KYTJ820101"]),
        return_keys=True,
    )

    assert features.shape == (2, 2)
    assert keys == ["aaindex.HOPT810101", "aaindex.KYTJ820101"]
    np.testing.assert_array_equal(features[:, 1], np.asarray([1.8, -3.5], np.float32))


@pytest.mark.parametrize(
    ("policy", "expected"),
    [("nan", np.nan), ("zero", 0.0)],
)
def test_aaindex_unknown_residue_policy(policy: str, expected: float) -> None:
    features = build_node_attr(
        _aa_indices("AX"),
        mode=_aaindex_mode(["KYTJ820101"], unknown=policy),
    )

    if np.isnan(expected):
        assert np.isnan(features[1, 0])
    else:
        assert features[1, 0] == expected


def test_aaindex_unknown_residue_error_policy() -> None:
    with pytest.raises(ValueError, match="unknown residue X"):
        build_node_attr(
            _aa_indices("X"),
            mode=_aaindex_mode(["KYTJ820101"], unknown="error"),
        )


@pytest.mark.parametrize(
    ("policy", "expected"),
    [("nan", np.nan), ("zero", 0.0)],
)
def test_aaindex_missing_record_value_policy(policy: str, expected: float) -> None:
    features = build_node_attr(
        _aa_indices("AP"),
        mode=_aaindex_mode(["AVBF000101"], missing=policy),
    )

    if np.isnan(expected):
        assert np.isnan(features[1, 0])
    else:
        assert features[1, 0] == expected


def test_aaindex_missing_record_value_error_policy() -> None:
    with pytest.raises(ValueError, match="no finite value for residue P"):
        build_node_attr(
            _aa_indices("P"),
            mode=_aaindex_mode(["AVBF000101"], missing="error"),
        )


def test_aaindex_composes_and_feature_keys_round_trip(tmp_path: Path) -> None:
    inp = _write_ca_structure(
        tmp_path / "composition.pdb",
        [("ALA", 12.0), ("ASP", 34.0), ("ILE", 56.0)],
    )
    node_mode = ["onehot", "physchem", *_aaindex_mode(["KYTJ820101"]), "bfactor_ca"]
    artifact = ResiduePointCloudCA(
        config=ResiduePointCloudConfig(node_attr_mode=node_mode),
        use_cache=False,
    ).encode(inp)

    keys = artifact.meta["feature_keys"]["node_attr_keys"]
    assert artifact.data["node_attr"].shape == (3, 38)
    assert keys[-2:] == ["aaindex.KYTJ820101", "bfactor_ca"]
    np.testing.assert_array_equal(
        artifact.data["node_attr"][:, -2], np.asarray([1.8, -3.5, 4.5], dtype=np.float32)
    )

    outdir = tmp_path / "artifact"
    artifact.save(outdir)
    loaded = StructureArtifact.load(outdir, strict=True)
    assert loaded.meta["feature_keys"]["node_attr_keys"] == keys
    assert loaded.meta["parameters"]["node_attr_mode"][2]["aaindex"]["provider"] == "aaindex"
    assert loaded.meta["parameters"]["node_attr_mode"][2]["aaindex"][
        "provider_version"
    ] == metadata.version("aaindex")
