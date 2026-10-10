from __future__ import annotations

from copy import deepcopy
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest

import eris
from eris.core.artifacts import StructureArtifact
from eris.core.cache import DiskCache
from eris.core.config import ResiduePointCloudConfig, ResidueVoxelGridConfig
from eris.core.io import StructureLoadError, load_and_extract_residue
from eris.core.types import StructureInput
from eris.encoders.clouds.atom_point_cloud import AtomPointCloud
from eris.encoders.clouds.residue_point_cloud import ResiduePointCloudCA
from eris.encoders.graphs.atom_knn_graph import AtomKNNGraph
from eris.encoders.graphs.residue_knn_graph import ResidueKNNGraph
from eris.encoders.grids.voxel_grid_ca import VoxelGridCA
from eris.features.geometry import (
    SIDECHAIN_CHI_DEFINITIONS,
    backbone_torsions,
    local_frames,
    peptide_continuity,
    sidechain_torsions,
    signed_dihedral,
)
from eris.features.node.constants import AA_VOCAB, PHYSchem_KEYS
from eris.features.node.external import ExternalFeatureError
from eris.features.node.residue import build_node_attr
from eris.pipelines.manifest import StructureManifestRow

PDB = Path("tests/data/1AQ2.pdb")
MMCIF = Path("tests/data/1AQ2.cif")


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


def _backbone_atom_line(
    serial: int,
    atom_name: str,
    residue_name: str,
    chain_id: str,
    residue_number: int,
    insertion_code: str,
    coordinate: tuple[float, float, float],
) -> str:
    x, y, z = coordinate
    return (
        f"ATOM  {serial:5d} {atom_name:>4s} {residue_name:>3s} {chain_id}"
        f"{residue_number:4d}{insertion_code:1s}   {x:8.3f}{y:8.3f}{z:8.3f}"
        f"{1.0:6.2f}{10.0:6.2f}           {atom_name[0]:>2s}  "
    )


def _altloc_atom_line(
    serial: int,
    atom_name: str,
    altloc: str,
    coordinate: tuple[float, float, float],
    occupancy: float,
) -> str:
    x, y, z = coordinate
    return (
        f"ATOM  {serial:5d} {atom_name:>4s}{altloc:1s}ALA A   1    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{10.0:6.2f}"
        f"           {atom_name[0]:>2s}  "
    )


def _write_backbone_structure(
    path: Path,
    residues: list[
        tuple[
            str,
            int,
            str,
            str,
            dict[str, tuple[float, float, float]],
        ]
    ],
) -> StructureInput:
    lines: list[str] = []
    serial = 1
    previous_chain: str | None = None
    for chain_id, residue_number, insertion_code, residue_name, atoms in residues:
        if previous_chain is not None and chain_id != previous_chain:
            lines.append("TER")
        for atom_name in ("N", "CA", "C"):
            if atom_name in atoms:
                lines.append(
                    _backbone_atom_line(
                        serial,
                        atom_name,
                        residue_name,
                        chain_id,
                        residue_number,
                        insertion_code,
                        atoms[atom_name],
                    )
                )
                serial += 1
        previous_chain = chain_id
    path.write_text("\n".join([*lines, "TER", "END", ""]), encoding="ascii")
    return StructureInput(source=str(path))


def _write_heavy_atom_structure(
    path: Path,
    residues: list[
        tuple[
            str,
            int,
            str,
            dict[str, tuple[float, float, float]],
        ]
    ],
    *,
    header: list[str] | None = None,
) -> StructureInput:
    lines = list(header or [])
    serial = 1
    previous_chain: str | None = None
    for chain_id, residue_number, residue_name, atoms in residues:
        if previous_chain is not None and chain_id != previous_chain:
            lines.append("TER")
        record = "HETATM" if residue_name in {"MSE", "HOH", "LIG", "ZN", "FAD"} else "ATOM  "
        for atom_name, coordinate in atoms.items():
            x, y, z = coordinate
            element = atom_name if atom_name in {"SE", "ZN"} else atom_name[0]
            lines.append(
                f"{record}{serial:5d} {atom_name:>4s} {residue_name:>3s} {chain_id}"
                f"{residue_number:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                f"{1.0:6.2f}{10.0:6.2f}          {element:>2s}  "
            )
            serial += 1
        previous_chain = chain_id
    path.write_text("\n".join([*lines, "TER", "END", ""]), encoding="ascii")
    return StructureInput(source=str(path))


def _complete_gly_atoms(
    offset: tuple[float, float, float],
) -> dict[str, tuple[float, float, float]]:
    x, y, z = offset
    return {
        "N": (x, y, z),
        "CA": (x + 1.45, y, z),
        "C": (x + 2.45, y + 1.0, z),
        "O": (x + 2.30, y + 2.2, z),
    }


_CHI_TEST_COORDINATES = {
    "N": (1.0, 0.0, 0.0),
    "CA": (0.0, 0.0, 0.0),
    "CB": (0.0, 0.0, 1.0),
    "CG": (0.0, 1.0, 1.0),
    "CG1": (0.0, 1.0, 1.0),
    "OG": (0.0, 1.0, 1.0),
    "OG1": (0.0, 1.0, 1.0),
    "SG": (0.0, 1.0, 1.0),
    "CD": (1.0, 1.0, 2.0),
    "CD1": (1.0, 1.0, 2.0),
    "ND1": (1.0, 1.0, 2.0),
    "OD1": (1.0, 1.0, 2.0),
    "SD": (1.0, 1.0, 2.0),
    "CE": (1.0, 2.0, 3.0),
    "NE": (1.0, 2.0, 3.0),
    "OE1": (1.0, 2.0, 3.0),
    "CZ": (2.0, 3.0, 3.0),
    "NZ": (2.0, 3.0, 3.0),
}


def _chi_atoms(residue_name: str) -> dict[str, tuple[float, float, float]]:
    definitions = SIDECHAIN_CHI_DEFINITIONS.get(residue_name, ())
    names = sorted({name for definition in definitions for name in definition})
    return {name: _CHI_TEST_COORDINATES[name] for name in names}


def _flatten_test_atoms(
    residues: list[tuple[str, dict[str, tuple[float, float, float]]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coordinates: list[tuple[float, float, float]] = []
    residue_indices: list[int] = []
    atom_names: list[str] = []
    residue_names: list[str] = []
    for residue_index, (residue_name, atoms) in enumerate(residues):
        residue_names.append(residue_name)
        for atom_name, coordinate in atoms.items():
            coordinates.append(coordinate)
            residue_indices.append(residue_index)
            atom_names.append(atom_name)
    return (
        np.asarray(coordinates, dtype=np.float32).reshape((-1, 3)),
        np.asarray(residue_indices, dtype=np.int64),
        np.asarray(atom_names, dtype=np.str_),
        np.asarray(residue_names, dtype=np.str_),
    )


def _aa_indices(sequence: str) -> np.ndarray:
    return np.asarray([AA_VOCAB.index(letter) for letter in sequence], dtype=np.int16)


def _write_external_features(
    path: Path,
    *,
    values: list[list[float]] | np.ndarray,
    residue_ids: list[str],
    source_name: str = "custom",
    feature_keys: list[str] | None = None,
    chain_ids: list[str] | None = None,
    model_index: int = 0,
) -> Path:
    if feature_keys is None:
        feature_keys = ["score_a", "score_b"]
    np.savez(
        path,
        values=np.asarray(values, dtype=np.float32),
        residue_ids=np.asarray(residue_ids),
        source_name=np.asarray(source_name),
        feature_keys=np.asarray(feature_keys),
        chain_ids=np.asarray(["A"] if chain_ids is None else chain_ids),
        model_index=np.asarray(model_index, dtype=np.int64),
    )
    return path


def _rigid_transform_pdb(source: Path, target: Path) -> None:
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    translation = np.asarray([13.0, -8.0, 4.0], dtype=np.float64)
    lines = []
    for line in source.read_text(encoding="ascii").splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            coordinate = np.asarray(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=np.float64,
            )
            x, y, z = coordinate @ rotation.T + translation
            line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
        lines.append(line)
    target.write_text("\n".join([*lines, ""]), encoding="ascii")


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


def test_atom_point_cloud_bfactor_is_raw_by_default(tmp_path: Path) -> None:
    inp = _write_ca_structure(
        tmp_path / "atom-cloud.pdb",
        [("ALA", 42.0), ("GLY", 142.0)],
    )

    artifact = AtomPointCloud(node_attr_mode=["bfactor"], use_cache=False).encode(inp)

    assert artifact.meta["feature_keys"]["node_attr_keys"] == ["bfactor"]
    np.testing.assert_array_equal(
        artifact.data["node_attr"][:, 0],
        np.asarray([42.0, 142.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(artifact.data["node_attr"][:, 0], artifact.data["bfactor"])


def test_atom_graph_bfactor_is_raw_by_default(tmp_path: Path) -> None:
    inp = _write_ca_structure(
        tmp_path / "atom-graph.pdb",
        [("ALA", 42.0), ("GLY", 142.0)],
    )

    artifact = AtomKNNGraph(k=1, node_attr_mode=["bfactor"], use_cache=False).encode(inp)

    assert artifact.meta["feature_keys"]["node_attr_keys"] == ["bfactor"]
    np.testing.assert_array_equal(
        artifact.data["node_attr"][:, 0],
        np.asarray([42.0, 142.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(artifact.data["node_attr"][:, 0], artifact.data["bfactor"])


def test_atom_altloc_first_selects_one_record_and_all_is_explicit(tmp_path: Path) -> None:
    source = tmp_path / "atom-altloc.pdb"
    source.write_text(
        "\n".join(
            [
                _altloc_atom_line(1, "N", "A", (0.0, 0.0, 0.0), 0.4),
                _altloc_atom_line(2, "N", "B", (10.0, 0.0, 0.0), 0.8),
                _altloc_atom_line(3, "CA", " ", (1.0, 0.0, 0.0), 1.0),
                "TER",
                "END",
                "",
            ]
        ),
        encoding="ascii",
    )
    structure = StructureInput(source=str(source), chain_id="A")

    default = AtomPointCloud(center=False, use_cache=False).encode(structure)
    first = AtomPointCloud(center=False, altloc="first", use_cache=False).encode(structure)
    all_locations = AtomPointCloud(center=False, altloc="all", use_cache=False).encode(structure)

    assert default.data["atom_name"].tolist() == ["N", "CA"]
    np.testing.assert_array_equal(default.data["pos"][0], [10.0, 0.0, 0.0])
    assert first.data["atom_name"].tolist() == ["N", "CA"]
    np.testing.assert_array_equal(first.data["pos"][0], [0.0, 0.0, 0.0])
    assert all_locations.data["atom_name"].tolist() == ["N", "N", "CA"]
    np.testing.assert_array_equal(all_locations.data["pos"][:2, 0], [0.0, 10.0])
    assert default.meta["parameters"]["altloc"] == "max_occupancy"
    assert first.meta["parameters"]["altloc"] == "first"
    assert all_locations.meta["parameters"]["altloc"] == "all"
    assert default.meta["encoder"]["version"] == "0.1.2"


def test_residue_center_of_mass_uses_noncarbon_element_mass(tmp_path: Path) -> None:
    structure = _write_heavy_atom_structure(
        tmp_path / "mse-center-of-mass.pdb",
        [
            (
                "A",
                1,
                "MSE",
                {
                    "N": (0.0, 0.0, 0.0),
                    "CA": (0.0, 0.0, 0.0),
                    "C": (0.0, 0.0, 0.0),
                    "O": (0.0, 0.0, 0.0),
                    "SE": (10.0, 0.0, 0.0),
                },
            )
        ],
    )

    artifact = ResiduePointCloudCA(
        residue_center="com",
        center=False,
        use_cache=False,
    ).encode(structure)

    expected_x = 10.0 * 78.96 / (14.007 + 12.011 + 12.011 + 15.999 + 78.96)
    np.testing.assert_allclose(artifact.data["pos"][0], [expected_x, 0.0, 0.0], rtol=1e-6)
    assert artifact.meta["coordinate_source"] == "residue_com"
    assert artifact.meta["encoder"]["version"] == "0.1.3"


def test_residue_bfactor_is_raw_by_default(tmp_path: Path) -> None:
    inp = _write_ca_structure(
        tmp_path / "residue-cloud.pdb",
        [("ALA", 42.0), ("GLY", 142.0)],
    )

    artifact = ResiduePointCloudCA(node_attr_mode=["bfactor_ca"], use_cache=False).encode(inp)

    assert artifact.meta["feature_keys"]["node_attr_keys"] == ["bfactor_ca"]
    np.testing.assert_array_equal(
        artifact.data["node_attr"][:, 0],
        np.asarray([42.0, 142.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(artifact.data["node_attr"][:, 0], artifact.data["bfactor_ca"])


def test_bfactor_is_not_automatically_interpreted_as_plddt(tmp_path: Path) -> None:
    inp = _write_ca_structure(tmp_path / "experimental.pdb", [("ALA", 42.0)])

    loaded = load_and_extract_residue(inp)

    np.testing.assert_array_equal(loaded["bfactor_ca"], np.asarray([42.0], dtype=np.float32))
    assert np.isnan(loaded["plddt"]).all()


def test_plddt_requires_explicit_source_and_preserves_raw_bfactor(tmp_path: Path) -> None:
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


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (0.0, 0.0),
        (np.pi / 2.0, np.pi / 2.0),
        (-np.pi / 2.0, -np.pi / 2.0),
        (np.pi, -np.pi),
    ],
)
def test_signed_dihedral_known_angles_and_boundary(angle: float, expected: float) -> None:
    points = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [np.cos(angle), np.sin(angle), 1.0],
        ],
        dtype=np.float64,
    )

    assert signed_dihedral(*points) == pytest.approx(expected, abs=1e-12)


def test_signed_dihedral_is_translation_and_rotation_invariant() -> None:
    points = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    expected = signed_dihedral(*points)

    translated = points + np.asarray([12.5, -7.25, 3.0])
    rotated = points @ rotation.T

    assert signed_dihedral(*translated) == pytest.approx(expected, abs=1e-12)
    assert signed_dihedral(*rotated) == pytest.approx(expected, abs=1e-12)


def test_local_frames_are_right_handed_orthonormal_and_rotation_equivariant() -> None:
    ca_coord = np.asarray([[2.0, 3.0, 4.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    c_coord = np.asarray([[3.0, 3.0, 4.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    n_coord = np.asarray([[2.0, 4.0, 4.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    valid = np.ones((2,), dtype=bool)

    frames, frame_valid = local_frames(
        n_coord,
        ca_coord,
        c_coord,
        valid,
        valid,
        valid,
    )

    np.testing.assert_array_equal(frame_valid, [True, True])
    np.testing.assert_allclose(frames[0], np.eye(3), atol=1e-7)
    for frame in frames:
        np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-7)
        np.testing.assert_allclose(np.cross(frame[:, 0], frame[:, 1]), frame[:, 2], atol=1e-7)
        assert np.linalg.det(frame) == pytest.approx(1.0, abs=1e-7)

    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    translation = np.asarray([7.0, -3.0, 2.0], dtype=np.float32)
    transformed, transformed_valid = local_frames(
        n_coord @ rotation.T + translation,
        ca_coord @ rotation.T + translation,
        c_coord @ rotation.T + translation,
        valid,
        valid,
        valid,
    )
    np.testing.assert_array_equal(transformed_valid, frame_valid)
    np.testing.assert_allclose(transformed, rotation @ frames, atol=1e-7)


def test_local_frames_preserve_nan_for_missing_nonfinite_and_degenerate_geometry() -> None:
    ca_coord = np.zeros((4, 3), dtype=np.float32)
    c_coord = np.asarray(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [np.nan, 0.0, 0.0], [5e-7, 0.0, 0.0]],
        dtype=np.float32,
    )
    n_coord = np.asarray(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    n_valid = np.asarray([False, True, True, True])
    valid = np.ones((4,), dtype=bool)

    frames, frame_valid = local_frames(
        n_coord,
        ca_coord,
        c_coord,
        n_valid,
        valid,
        valid,
    )

    np.testing.assert_array_equal(frame_valid, [False, False, False, False])
    assert np.isnan(frames).all()


def test_physical_peptide_continuity_is_inclusive_and_never_crosses_chains() -> None:
    c_coord = np.asarray(
        [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0], [12.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    n_coord = np.asarray(
        [[-1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [6.01, 0.0, 0.0], [10.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    valid = np.ones((4,), dtype=bool)

    continuity = peptide_continuity(
        c_coord,
        n_coord,
        valid,
        valid,
        np.asarray(["A", "A", "A", "B"]),
        np.asarray([0, 1, 2, 0]),
        max_peptide_bond_distance=2.0,
    )

    # Exactly 2.0 A is connected; 2.01 A and a chain boundary are not.
    np.testing.assert_array_equal(continuity, np.asarray([True, False, False]))


def test_backbone_torsions_respect_missing_atoms_physical_breaks_and_chains() -> None:
    n_coord = np.asarray(
        [[-1.0, 0.0, 0.0], [1.7, 0.8, 0.1], [4.0, 1.2, 1.1], [6.5, 0.7, 2.1]],
        dtype=np.float32,
    )
    ca_coord = np.asarray(
        [[0.0, 0.0, 0.0], [2.2, 1.5, 0.7], [4.6, 0.5, 1.5], [7.0, 1.4, 2.8]],
        dtype=np.float32,
    )
    c_coord = np.asarray(
        [[0.8, 0.8, 0.0], [3.1, 1.0, 1.0], [5.5, 0.9, 2.0], [7.9, 1.0, 3.1]],
        dtype=np.float32,
    )
    valid = np.ones((4,), dtype=bool)

    complete = backbone_torsions(
        n_coord,
        ca_coord,
        c_coord,
        valid,
        valid,
        valid,
        np.asarray(["A", "A", "A", "A"]),
        np.arange(4),
    )
    assert np.isnan(complete[0, 0])
    assert np.isfinite(complete[0, 1:]).all()
    assert np.isfinite(complete[1:3]).all()
    assert np.isfinite(complete[3, 0])
    assert np.isnan(complete[3, 1:]).all()

    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    translation = np.asarray([11.0, -4.0, 2.5], dtype=np.float32)
    transformed = backbone_torsions(
        n_coord @ rotation.T + translation,
        ca_coord @ rotation.T + translation,
        c_coord @ rotation.T + translation,
        valid,
        valid,
        valid,
        np.asarray(["A", "A", "A", "A"]),
        np.arange(4),
    )
    np.testing.assert_allclose(transformed, complete, atol=1e-6, equal_nan=True)

    missing_n = valid.copy()
    missing_n[1] = False
    incomplete = backbone_torsions(
        n_coord,
        ca_coord,
        c_coord,
        missing_n,
        valid,
        valid,
        np.asarray(["A", "A", "A", "A"]),
        np.arange(4),
    )
    assert np.isnan(incomplete[0, 1:]).all()
    assert np.isnan(incomplete[1, 0:2]).all()
    assert np.isfinite(incomplete[1, 2])
    assert np.isfinite(incomplete[2]).all()

    broken_c = c_coord.copy()
    broken_c[1] = [20.0, 20.0, 20.0]
    broken = backbone_torsions(
        n_coord,
        ca_coord,
        broken_c,
        valid,
        valid,
        valid,
        np.asarray(["A", "A", "A", "A"]),
        np.arange(4),
    )
    assert np.isnan(broken[1, 1:]).all()
    assert np.isnan(broken[2, 0])

    multichain = backbone_torsions(
        n_coord,
        ca_coord,
        c_coord,
        valid,
        valid,
        valid,
        np.asarray(["A", "A", "B", "B"]),
        np.asarray([0, 1, 0, 1]),
    )
    assert np.isnan(multichain[1, 1:]).all()
    assert np.isnan(multichain[2, 0])


def test_backbone_extraction_preserves_number_gaps_insertions_and_missing_atoms(
    tmp_path: Path,
) -> None:
    residues = [
        (
            "A",
            10,
            " ",
            "ALA",
            {"N": (-1.0, 0.0, 0.0), "CA": (0.0, 0.0, 0.0), "C": (0.8, 0.8, 0.0)},
        ),
        (
            "A",
            12,
            " ",
            "GLY",
            {"N": (1.7, 0.8, 0.1), "CA": (2.2, 1.5, 0.7), "C": (3.1, 1.0, 1.0)},
        ),
        (
            "A",
            12,
            "A",
            "SER",
            {"N": (4.0, 1.2, 1.1), "CA": (4.6, 0.5, 1.5)},
        ),
    ]
    inp = _write_backbone_structure(tmp_path / "ordering.pdb", residues)

    loaded = load_and_extract_residue(inp)
    assert loaded["residue_ids"].tolist() == ["A:10::ALA", "A:12::GLY", "A:12:A:SER"]
    np.testing.assert_array_equal(loaded["structural_residue_index"], [0, 1, 2])
    np.testing.assert_array_equal(loaded["backbone_c_mask"], [True, True, False])
    assert loaded["backbone_c"].shape == (3, 3)
    assert np.isnan(loaded["backbone_c"][2]).all()

    continuity = peptide_continuity(
        loaded["backbone_c"],
        loaded["backbone_n"],
        loaded["backbone_c_mask"],
        loaded["backbone_n_mask"],
        loaded["chain_id"],
        loaded["structural_residue_index"],
    )
    # The 10 -> 12 numbering gap is physically connected, and 12A remains structurally adjacent.
    np.testing.assert_array_equal(continuity, [True, True])

    artifact = ResiduePointCloudCA(
        node_attr_mode=[{"backbone_geometry": {"mode": "both"}}],
        use_cache=False,
    ).encode(inp)
    assert artifact.data["node_attr"].shape == (3, 9)
    assert np.isnan(artifact.data["node_attr"][0, [0, 1, 6]]).all()
    assert np.isfinite(artifact.data["node_attr"][0, [2, 3, 4, 5, 7, 8]]).all()
    assert np.isfinite(artifact.data["node_attr"][1]).all()
    assert np.isnan(artifact.data["node_attr"][2]).all()

    default = ResiduePointCloudCA(
        node_attr_mode=["backbone_geometry"],
        use_cache=False,
    ).encode(inp)
    assert default.meta["feature_keys"]["node_attr_keys"] == [
        "backbone.phi_sin",
        "backbone.phi_cos",
        "backbone.psi_sin",
        "backbone.psi_cos",
        "backbone.omega_sin",
        "backbone.omega_cos",
    ]
    assert default.data["node_attr"].shape == (3, 6)


def test_backbone_extraction_uses_frozen_altloc_selection(tmp_path: Path) -> None:
    source = tmp_path / "altloc.pdb"
    source.write_text(
        "\n".join(
            [
                _altloc_atom_line(1, "N", "A", (1.0, 0.0, 0.0), 0.4),
                _altloc_atom_line(2, "N", "B", (2.0, 0.0, 0.0), 0.8),
                _altloc_atom_line(3, "CA", " ", (3.0, 0.0, 0.0), 0.5),
                _altloc_atom_line(4, "CA", "A", (4.0, 0.0, 0.0), 0.5),
                _altloc_atom_line(5, "C", "C", (5.0, 0.0, 0.0), 0.5),
                _altloc_atom_line(6, "C", "B", (6.0, 0.0, 0.0), 0.5),
                "TER",
                "END",
                "",
            ]
        ),
        encoding="ascii",
    )

    loaded = load_and_extract_residue(StructureInput(source=str(source), chain_id="A"))

    np.testing.assert_array_equal(loaded["backbone_n"], [[2.0, 0.0, 0.0]])
    np.testing.assert_array_equal(loaded["backbone_ca"], [[3.0, 0.0, 0.0]])
    np.testing.assert_array_equal(loaded["backbone_c"], [[6.0, 0.0, 0.0]])


def test_backbone_geometry_realistic_pdb_mmcif_hybrid_round_trip_and_identity(
    tmp_path: Path,
) -> None:
    explicit = {
        "backbone_geometry": {
            "mode": "both",
            "max_peptide_bond_distance": 2,
            "torsion_convention": "standard_signed_atan2[-pi,pi)",
            "units": "radians",
            "missing_value_policy": "nan",
        }
    }
    mode = [explicit, "onehot", "bfactor_ca"]
    pdb = ResidueKNNGraph(
        k=2, node_attr_mode=[explicit, "onehot", "bfactor_ca"], use_cache=False
    ).encode(StructureInput(source=str(PDB), chain_id="A"))
    mmcif = ResidueKNNGraph(k=2, node_attr_mode=mode, use_cache=False).encode(
        StructureInput(source=str(MMCIF), chain_id="A")
    )
    repeat = ResidueKNNGraph(k=2, node_attr_mode=mode, use_cache=False).encode(
        StructureInput(source=str(PDB), chain_id="A")
    )

    periodic_keys = [
        "backbone.phi_sin",
        "backbone.phi_cos",
        "backbone.psi_sin",
        "backbone.psi_cos",
        "backbone.omega_sin",
        "backbone.omega_cos",
    ]
    raw_keys = ["backbone.phi", "backbone.psi", "backbone.omega"]
    expected_keys = [
        *[f"aa_onehot.{aa}" for aa in AA_VOCAB],
        "bfactor_ca",
        *periodic_keys,
        *raw_keys,
    ]
    assert pdb.meta["feature_keys"]["node_attr_keys"] == expected_keys
    assert pdb.data["node_attr"].dtype == np.float32
    assert pdb.data["node_attr"].shape == (534, len(expected_keys))
    raw = pdb.data["node_attr"][:, -3:]
    assert np.isnan(raw[0, 0])
    assert np.isnan(raw[-1, 1:]).all()
    assert (np.isfinite(raw).sum(axis=0) > 500).all()
    periodic = pdb.data["node_attr"][:, -9:-3]
    assert np.isnan(periodic[0, 0:2]).all()
    assert np.isnan(periodic[-1, 2:6]).all()
    np.testing.assert_equal(pdb.data["node_attr"], repeat.data["node_attr"])
    np.testing.assert_equal(pdb.data["node_attr"], mmcif.data["node_attr"])

    params = pdb.meta["parameters"]["node_attr_mode"][0]["backbone_geometry"]
    assert params == {
        "max_peptide_bond_distance": 2.0,
        "missing_value_policy": "nan",
        "mode": "both",
        "torsion_convention": "standard_signed_atan2[-pi,pi)",
        "units": "radians",
    }
    assert (
        pdb.meta["provenance"]["config_fingerprint"]
        == mmcif.meta["provenance"]["config_fingerprint"]
    )
    raw_only = ResiduePointCloudCA(
        node_attr_mode=[{"backbone_geometry": {"mode": "raw"}}],
        use_cache=False,
    ).encode(StructureInput(source=str(PDB), chain_id="A"))
    assert raw_only.meta["feature_keys"]["node_attr_keys"] == raw_keys
    assert (
        pdb.meta["provenance"]["config_fingerprint"]
        != raw_only.meta["provenance"]["config_fingerprint"]
    )

    artifact_dir = tmp_path / "backbone-artifact"
    pdb.save(artifact_dir)
    loaded = StructureArtifact.load(artifact_dir, strict=True)
    np.testing.assert_equal(loaded.data["node_attr"], pdb.data["node_attr"])
    assert loaded.meta["feature_keys"]["node_attr_keys"] == expected_keys


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


def test_external_features_align_by_exact_residue_id_and_compose(tmp_path: Path) -> None:
    inp = _write_ca_structure(tmp_path / "protein.pdb", [("ALA", 12.0), ("GLY", 34.0)])
    external = _write_external_features(
        tmp_path / "external.npz",
        values=[[20.0, 21.0], [10.0, 11.0]],
        residue_ids=["A:2::GLY", "A:1::ALA"],
    )

    artifact = ResiduePointCloudCA(
        node_attr_mode=["onehot", {"external": {"path": str(external)}}],
        use_cache=False,
    ).encode(inp)

    assert artifact.data["node_attr"].shape == (2, 23)
    np.testing.assert_array_equal(
        artifact.data["node_attr"][:, -2:],
        np.asarray([[10.0, 11.0], [20.0, 21.0]], dtype=np.float32),
    )
    assert artifact.meta["feature_keys"]["node_attr_keys"][-2:] == [
        "external.custom.score_a",
        "external.custom.score_b",
    ]
    descriptor = artifact.meta["parameters"]["node_attr_mode"][1]["external"]
    assert descriptor["alignment"] == "strict_residue_id"
    assert descriptor["feature_dim"] == 2
    assert artifact.meta["extensions"]["org.eris.external_features"]["invocations"] == [
        {
            "path": str(external.resolve()),
            "content_sha256": descriptor["content_sha256"],
        }
    ]

    artifact_dir = tmp_path / "self-contained-artifact"
    artifact.save(artifact_dir)
    external.unlink()
    reloaded = StructureArtifact.load(artifact_dir, strict=True)
    np.testing.assert_array_equal(reloaded.data["node_attr"], artifact.data["node_attr"])


@pytest.mark.parametrize(
    ("residue_ids", "match"),
    [
        (["A:1::ALA"], r"strict_residue_id.*missing=1"),
        (["A:1::ALA", "A:2::GLY", "A:3::SER"], r"strict_residue_id.*extra=1"),
        (["A:1::ALA", "A:1::ALA"], r"duplicate IDs"),
    ],
    ids=["missing", "extra", "duplicate"],
)
def test_external_features_reject_invalid_residue_identity_set(
    tmp_path: Path,
    residue_ids: list[str],
    match: str,
) -> None:
    inp = _write_ca_structure(tmp_path / "protein.pdb", [("ALA", 12.0), ("GLY", 34.0)])
    values = np.arange(len(residue_ids) * 2, dtype=np.float32).reshape(len(residue_ids), 2)
    external = _write_external_features(
        tmp_path / "external.npz",
        values=values,
        residue_ids=residue_ids,
    )

    with pytest.raises(ExternalFeatureError, match=match):
        ResiduePointCloudCA(
            node_attr_mode=[{"external": {"path": str(external)}}],
            use_cache=False,
        ).encode(inp)


def test_external_rejects_wrong_chain(tmp_path: Path) -> None:
    inp = _write_ca_structure(tmp_path / "protein.pdb", [("ALA", 12.0), ("GLY", 34.0)])
    external = _write_external_features(
        tmp_path / "wrong-chain.npz",
        values=[[1.0, 2.0], [3.0, 4.0]],
        residue_ids=["B:1::ALA", "B:2::GLY"],
        chain_ids=["B"],
    )

    with pytest.raises(ExternalFeatureError, match="External chain-set mismatch"):
        ResiduePointCloudCA(
            node_attr_mode=[{"external": {"path": str(external)}}],
            use_cache=False,
        ).encode(inp)


def test_external_rejects_wrong_model(tmp_path: Path) -> None:
    inp = _write_ca_structure(tmp_path / "protein.pdb", [("ALA", 12.0), ("GLY", 34.0)])
    external = _write_external_features(
        tmp_path / "wrong-model.npz",
        values=[[1.0, 2.0], [3.0, 4.0]],
        residue_ids=["A:1::ALA", "A:2::GLY"],
        model_index=1,
    )

    with pytest.raises(ExternalFeatureError, match="External model-index mismatch"):
        ResiduePointCloudCA(
            node_attr_mode=[{"external": {"path": str(external)}}],
            use_cache=False,
        ).encode(inp)


def test_external_features_align_across_canonical_multichain_selection(tmp_path: Path) -> None:
    structure = tmp_path / "multi.pdb"
    structure.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00           C  ",
                "TER",
                "ATOM      2  CA  GLY B   1       1.000   0.000   0.000  1.00 20.00           C  ",
                "TER",
                "END",
                "",
            ]
        ),
        encoding="ascii",
    )
    external = _write_external_features(
        tmp_path / "multi.npz",
        values=[[3.0, 4.0], [1.0, 2.0]],
        residue_ids=["B:1::GLY", "A:1::ALA"],
        chain_ids=["A", "B"],
    )

    artifact = ResiduePointCloudCA(
        node_attr_mode=[{"external": {"path": str(external)}}],
        use_cache=False,
    ).encode(StructureInput(source=str(structure), chain_id=["B", "A"]))

    assert artifact.meta["input"]["chain_ids"] == ["A", "B"]
    np.testing.assert_array_equal(
        artifact.data["node_attr"], np.asarray([[1.0, 2.0], [3.0, 4.0]], np.float32)
    )


def test_external_legacy_scalar_chain_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "legacy.npz"
    np.savez(
        path,
        values=np.asarray([[1.0]], dtype=np.float32),
        residue_ids=np.asarray(["A:1::ALA"]),
        source_name=np.asarray("legacy"),
        chain_id=np.asarray("A"),
        model_index=np.asarray(0, dtype=np.int64),
    )

    with pytest.raises(ExternalFeatureError, match="missing fields.*chain_ids"):
        ResiduePointCloudConfig(node_attr_mode=[{"external": {"path": str(path)}}])


def test_external_detects_source_mutation_after_config_resolution(tmp_path: Path) -> None:
    inp = _write_ca_structure(tmp_path / "protein.pdb", [("ALA", 12.0), ("GLY", 34.0)])
    external = _write_external_features(
        tmp_path / "mutable.npz",
        values=[[1.0, 2.0], [3.0, 4.0]],
        residue_ids=["A:1::ALA", "A:2::GLY"],
    )
    config = ResiduePointCloudConfig(node_attr_mode=[{"external": {"path": str(external)}}])
    resolved = config.to_dict()["node_attr_mode"][0]["external"]

    _write_external_features(
        external,
        values=[[10.0, 20.0], [30.0, 40.0]],
        residue_ids=["A:1::ALA", "A:2::GLY"],
    )

    with pytest.raises(
        ExternalFeatureError,
        match="changed after configuration was resolved",
    ):
        ResiduePointCloudCA(config=config, use_cache=False).encode(inp)

    assert config.to_dict()["node_attr_mode"][0]["external"] == resolved


def test_external_path_relocation_preserves_scientific_identity(tmp_path: Path) -> None:
    inp = _write_ca_structure(tmp_path / "protein.pdb", [("ALA", 12.0), ("GLY", 34.0)])
    first_path = _write_external_features(
        tmp_path / "first-external.npz",
        values=[[1.0, 2.0], [3.0, 4.0]],
        residue_ids=["A:1::ALA", "A:2::GLY"],
    )
    relocated_path = tmp_path / "relocated" / "external.npz"
    relocated_path.parent.mkdir()
    relocated_path.write_bytes(first_path.read_bytes())

    first = ResiduePointCloudCA(
        node_attr_mode=[{"external": {"path": str(first_path)}}],
        use_cache=False,
    ).encode(inp)
    relocated = ResiduePointCloudCA(
        node_attr_mode=[{"external": {"path": str(relocated_path)}}],
        use_cache=False,
    ).encode(inp)

    assert (
        first.meta["provenance"]["config_fingerprint"]
        == relocated.meta["provenance"]["config_fingerprint"]
    )
    assert (
        first.meta["parameters"]["node_attr_mode"][0]["external"]["content_sha256"]
        == relocated.meta["parameters"]["node_attr_mode"][0]["external"]["content_sha256"]
    )
    first_invocation = first.meta["extensions"]["org.eris.external_features"]["invocations"][0]
    relocated_invocation = relocated.meta["extensions"]["org.eris.external_features"][
        "invocations"
    ][0]
    assert first_invocation["path"] != relocated_invocation["path"]
    np.testing.assert_array_equal(first.data["node_attr"], relocated.data["node_attr"])


def test_high_dimensional_external_hybrid_round_trip_and_identity(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    inp = _write_ca_structure(tmp_path / "protein.pdb", [("ALA", 12.0), ("GLY", 34.0)])
    feature_dim = 512
    feature_keys = [f"feature_{index:03d}" for index in range(feature_dim)]
    external_keys = [f"external.embedding.{key}" for key in feature_keys]
    values = np.arange(2 * feature_dim, dtype=np.float32).reshape(2, feature_dim)
    external = _write_external_features(
        tmp_path / "external.npz",
        values=values,
        residue_ids=["A:1::ALA", "A:2::GLY"],
        source_name="embedding",
        feature_keys=feature_keys,
    )
    cache = DiskCache(tmp_path / "cache")
    node_attr_mode = ["onehot", "bfactor_ca", {"external": {"path": str(external)}}]

    first = ResidueKNNGraph(
        k=1,
        node_attr_mode=node_attr_mode,
        use_cache=True,
        cache=cache,
    ).encode(inp)
    first.finalize()
    artifact_dir = tmp_path / "artifact"
    first.save(artifact_dir)
    loaded = StructureArtifact.load(artifact_dir, strict=True)

    expected_keys = [*[f"aa_onehot.{aa}" for aa in AA_VOCAB], "bfactor_ca", *external_keys]
    assert loaded.data["node_attr"].shape == (2, len(AA_VOCAB) + 1 + feature_dim)
    assert loaded.meta["feature_keys"]["node_attr_keys"] == expected_keys
    np.testing.assert_array_equal(loaded.data["node_attr"][:, -feature_dim:], values)
    torch_output = eris.to_torch(loaded, prefer="graph")
    assert isinstance(torch_output["x"], torch.Tensor)
    np.testing.assert_array_equal(torch_output["x"].cpu().numpy(), loaded.data["node_attr"])

    if find_spec("torch_geometric") is not None:
        pyg = eris.to_pyg(loaded, prefer="graph")
        np.testing.assert_array_equal(pyg.x.cpu().numpy(), loaded.data["node_attr"])

    changed_values = values + np.float32(1000.0)
    _write_external_features(
        external,
        values=changed_values,
        residue_ids=["A:1::ALA", "A:2::GLY"],
        source_name="embedding",
        feature_keys=feature_keys,
    )
    second = ResidueKNNGraph(
        k=1,
        node_attr_mode=node_attr_mode,
        use_cache=True,
        cache=cache,
    ).encode(inp)

    first_descriptor = loaded.meta["parameters"]["node_attr_mode"][2]["external"]
    second_descriptor = second.meta["parameters"]["node_attr_mode"][2]["external"]
    assert first_descriptor["content_sha256"] != second_descriptor["content_sha256"]
    assert (
        loaded.meta["provenance"]["config_fingerprint"]
        != second.meta["provenance"]["config_fingerprint"]
    )
    np.testing.assert_array_equal(second.data["node_attr"][:, -feature_dim:], changed_values)
    assert len(list((cache.cache_dir / "v2").iterdir())) == 2


def test_batch_external_params_by_id_preserve_resume_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_source = _write_ca_structure(tmp_path / "first.pdb", [("ALA", 12.0), ("GLY", 34.0)])
    second_source = _write_ca_structure(tmp_path / "second.pdb", [("ALA", 56.0), ("GLY", 78.0)])
    first_external = _write_external_features(
        tmp_path / "first.npz",
        values=[[1.0, 2.0], [3.0, 4.0]],
        residue_ids=["A:1::ALA", "A:2::GLY"],
    )
    second_external = _write_external_features(
        tmp_path / "second.npz",
        values=[[5.0, 6.0], [7.0, 8.0]],
        residue_ids=["A:1::ALA", "A:2::GLY"],
    )
    rows = [
        StructureManifestRow(id="first", path=first_source.source, chain="A"),
        StructureManifestRow(id="second", path=second_source.source, chain="A"),
    ]
    params_by_id = {
        "first": {"node_attr_mode": [{"external": {"path": str(first_external)}}]},
        "second": {"node_attr_mode": [{"external": {"path": str(second_external)}}]},
    }
    artifact_root = tmp_path / "artifacts"

    results = eris.batch_encode(
        rows,
        representation="residue_point_cloud",
        artifact_root=artifact_root,
        params_by_id=params_by_id,
        use_cache=False,
    )

    assert [result.status for result in results] == ["success", "success"]
    first = StructureArtifact.load(artifact_root / "first" / "residue_point_cloud")
    second = StructureArtifact.load(artifact_root / "second" / "residue_point_cloud")
    np.testing.assert_array_equal(first.data["node_attr"], [[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(second.data["node_attr"], [[5.0, 6.0], [7.0, 8.0]])
    assert (
        first.meta["provenance"]["config_fingerprint"]
        != second.meta["provenance"]["config_fingerprint"]
    )

    def fail_encode(*args: object, **kwargs: object) -> StructureArtifact:
        raise AssertionError("matching per-row external identities must resume without execution")

    monkeypatch.setattr(eris, "encode", fail_encode)
    resumed = eris.batch_encode(
        rows,
        representation="residue_point_cloud",
        artifact_root=artifact_root,
        params_by_id=params_by_id,
        use_cache=False,
        resume=True,
    )
    assert [result.status for result in resumed] == ["success", "success"]


def test_secondary_structure_assigns_frozen_states_with_provider_provenance() -> None:
    artifact = ResiduePointCloudCA(
        node_attr_mode=["secondary_structure"],
        use_cache=False,
    ).encode(StructureInput(source=str(PDB), chain_id="A"))

    assert artifact.meta["feature_keys"]["node_attr_keys"] == [
        "secondary_structure.helix",
        "secondary_structure.sheet",
        "secondary_structure.coil",
        "secondary_structure.unassigned",
    ]
    features = artifact.data["node_attr"]
    np.testing.assert_array_equal(features.sum(axis=1), np.ones(features.shape[0]))
    assert np.all(features.sum(axis=0)[:3] > 0)
    params = artifact.meta["parameters"]["node_attr_mode"][0]["secondary_structure"]
    assert params["provider"] == "biotite"
    assert params["provider_version"] == metadata.version("biotite")
    assert params["algorithm"] == "P-SEA"
    assert params["state_mapping"] == {
        "": "unassigned",
        "a": "helix",
        "b": "sheet",
        "c": "coil",
    }
    assert params["segment_handling"] == "selected_chain_and_peptide_continuity"


def test_secondary_structure_missing_ca_is_unassigned(tmp_path: Path) -> None:
    inp = _write_heavy_atom_structure(
        tmp_path / "missing-ca.pdb",
        [("A", 1, "ALA", {"N": (0.0, 0.0, 0.0), "C": (2.5, 0.0, 0.0), "O": (3.5, 0.0, 0.0)})],
    )

    artifact = ResiduePointCloudCA(
        residue_center="mean",
        node_attr_mode=["secondary_structure"],
        use_cache=False,
    ).encode(inp)

    np.testing.assert_array_equal(artifact.data["node_attr"], [[0.0, 0.0, 0.0, 1.0]])


def test_secondary_structure_uses_geometry_not_header_records(tmp_path: Path) -> None:
    residues = []
    for index in range(6):
        x = index * 3.45
        residues.append(
            (
                "A",
                index + 1,
                "GLY",
                {
                    "N": (x, 0.0, 0.0),
                    "CA": (x + 1.2, 0.5, 0.0),
                    "C": (x + 2.2, 0.0, 0.0),
                    "O": (x + 2.4, -1.0, 0.0),
                },
            )
        )
    plain = _write_heavy_atom_structure(tmp_path / "plain.pdb", residues)
    declared = _write_heavy_atom_structure(
        tmp_path / "declared.pdb",
        residues,
        header=["HELIX    1   1 GLY A    1  GLY A    6  1                                  6"],
    )

    kwargs = {"node_attr_mode": ["secondary_structure"], "use_cache": False}
    plain_artifact = ResiduePointCloudCA(**kwargs).encode(plain)
    declared_artifact = ResiduePointCloudCA(**kwargs).encode(declared)

    np.testing.assert_array_equal(
        plain_artifact.data["node_attr"],
        declared_artifact.data["node_attr"],
    )


def test_secondary_structure_multichain_order_matches_independent_chains(tmp_path: Path) -> None:
    residues = []
    for chain_index, chain_id in enumerate(("A", "B")):
        for index in range(6):
            x = index * 3.45
            y = chain_index * 12.0
            residues.append(
                (
                    chain_id,
                    index + 1,
                    "GLY",
                    {
                        "N": (x, y, 0.0),
                        "CA": (x + 1.2, y + 0.5, 0.0),
                        "C": (x + 2.2, y, 0.0),
                        "O": (x + 2.4, y - 1.0, 0.0),
                    },
                )
            )
    inp = _write_heavy_atom_structure(tmp_path / "two-chains.pdb", residues)

    encoder = ResiduePointCloudCA(node_attr_mode=["secondary_structure"], use_cache=False)
    chain_a = encoder.encode(StructureInput(source=inp.source, chain_id="A"))
    chain_b = encoder.encode(StructureInput(source=inp.source, chain_id="B"))
    combined = encoder.encode(StructureInput(source=inp.source, chain_id=["B", "A"]))

    assert combined.meta["input"]["chain_ids"] == ["A", "B"]
    np.testing.assert_array_equal(
        combined.data["node_attr"],
        np.concatenate([chain_a.data["node_attr"], chain_b.data["node_attr"]]),
    )


def test_sasa_rsa_selected_complex_context_and_tien_normalization(tmp_path: Path) -> None:
    chain_a_atoms = _complete_gly_atoms((0.0, 0.0, 0.0))
    inp = _write_heavy_atom_structure(
        tmp_path / "occlusion.pdb",
        [
            ("A", 1, "GLY", chain_a_atoms),
            ("B", 1, "GLY", _complete_gly_atoms((0.0, 2.8, 0.0))),
        ],
    )
    with_excluded_context = _write_heavy_atom_structure(
        tmp_path / "excluded-context.pdb",
        [
            ("A", 1, "GLY", {**chain_a_atoms, "H": (0.0, 0.5, 0.0)}),
            ("A", 2, "HOH", {"O": (1.0, 1.0, 0.0)}),
            ("A", 3, "LIG", {"C1": (1.0, 1.0, 0.0)}),
            ("A", 4, "ZN", {"ZN": (1.0, 1.0, 0.0)}),
            ("C", 1, "FAD", {"C1": (1.0, 1.0, 0.0)}),
            ("B", 1, "GLY", _complete_gly_atoms((0.0, 2.8, 0.0))),
        ],
    )
    encoder = ResiduePointCloudCA(node_attr_mode=["sasa", "rsa"], use_cache=False)

    isolated = encoder.encode(StructureInput(source=inp.source, chain_id="A"))
    complex_artifact = encoder.encode(StructureInput(source=inp.source, chain_id=["B", "A"]))
    repeated = encoder.encode(StructureInput(source=inp.source, chain_id=["A", "B"]))
    excluded = encoder.encode(
        StructureInput(source=with_excluded_context.source, chain_id=["A", "B", "C"])
    )

    assert isolated.meta["feature_keys"]["node_attr_keys"] == ["sasa", "rsa"]
    assert complex_artifact.data["node_attr"].shape == (2, 2)
    assert np.isfinite(complex_artifact.data["node_attr"]).all()
    assert np.all(complex_artifact.data["node_attr"] >= 0.0)
    assert complex_artifact.data["node_attr"][0, 0] < isolated.data["node_attr"][0, 0]
    assert isolated.data["node_attr"][0, 1] > 1.0
    np.testing.assert_array_equal(complex_artifact.data["node_attr"], repeated.data["node_attr"])
    np.testing.assert_array_equal(complex_artifact.data["node_attr"], excluded.data["node_attr"])
    np.testing.assert_allclose(
        complex_artifact.data["node_attr"][:, 1],
        complex_artifact.data["node_attr"][:, 0] / 104.0,
        rtol=1e-6,
    )
    params = complex_artifact.meta["parameters"]["node_attr_mode"]
    assert params[0]["sasa"]["sphere_point_count"] == 1000
    assert params[0]["sasa"]["vdw_radii"] == "ProtOr"
    assert params[1]["rsa"]["maximum_asa_table"]["GLY"] == 104.0
    assert params[1]["rsa"]["clipping"] == "none"


def test_sasa_incomplete_canonical_and_noncanonical_rsa_policies(tmp_path: Path) -> None:
    inp = _write_heavy_atom_structure(
        tmp_path / "missing-and-noncanonical.pdb",
        [
            (
                "A",
                1,
                "ALA",
                {
                    "N": (0.0, 0.0, 0.0),
                    "CA": (1.4, 0.0, 0.0),
                    "C": (2.4, 1.0, 0.0),
                    "O": (2.2, 2.2, 0.0),
                },
            ),
            (
                "A",
                2,
                "MSE",
                {
                    "N": (3.4, 1.0, 0.0),
                    "CA": (4.4, 2.0, 0.0),
                    "C": (5.7, 1.4, 0.0),
                    "O": (6.0, 0.2, 0.0),
                    "CB": (4.2, 3.5, 0.0),
                },
            ),
        ],
    )

    artifact = ResiduePointCloudCA(
        node_attr_mode=["flags", "sasa", "rsa"],
        use_cache=False,
    ).encode(inp)

    np.testing.assert_array_equal(artifact.data["resname"], ["ALA", "MSE"])
    assert artifact.data["node_attr"][1, 2] == 1.0
    assert np.isnan(artifact.data["node_attr"][0, -2:]).all()
    assert np.isfinite(artifact.data["node_attr"][1, -2])
    assert np.isnan(artifact.data["node_attr"][1, -1])


def test_sasa_fails_for_unsupported_required_protor_radius(tmp_path: Path) -> None:
    mse_atoms = {
        "N": (0.0, 0.0, 0.0),
        "CA": (1.4, 0.0, 0.0),
        "C": (2.4, 1.0, 0.0),
        "O": (2.2, 2.2, 0.0),
        "CB": (1.2, -1.5, 0.0),
        "CG": (2.4, -2.3, 0.0),
        "QQ": (2.0, -4.1, 0.0),
        "CE": (3.7, -5.0, 0.0),
    }
    inp = _write_heavy_atom_structure(
        tmp_path / "unsupported-radius.pdb",
        [
            ("A", 1, "GLY", _complete_gly_atoms((-4.0, 0.0, 0.0))),
            ("A", 2, "MSE", mse_atoms),
        ],
    )

    with pytest.raises(ValueError, match="ProtOr radius is unavailable.*MSE:QQ"):
        ResiduePointCloudCA(node_attr_mode=["sasa"], use_cache=False).encode(inp)


def test_secondary_sasa_rsa_hybrid_graph_round_trip(tmp_path: Path) -> None:
    inp = _write_heavy_atom_structure(
        tmp_path / "hybrid.pdb",
        [
            ("A", 1, "GLY", _complete_gly_atoms((0.0, 0.0, 0.0))),
            ("A", 2, "GLY", _complete_gly_atoms((3.45, 0.0, 0.0))),
            ("B", 1, "GLY", _complete_gly_atoms((0.0, 4.0, 0.0))),
            ("B", 2, "GLY", _complete_gly_atoms((3.45, 4.0, 0.0))),
        ],
    )
    artifact = ResidueKNNGraph(
        k=2,
        node_attr_mode=[
            "onehot",
            "backbone_geometry",
            "secondary_structure",
            "sasa",
            "rsa",
            "interface",
        ],
        edge_attr_mode=["dist", "relative_direction", "interface_contact"],
        use_cache=False,
    ).encode(inp)
    target = tmp_path / "artifact"
    artifact.save(target)
    loaded = StructureArtifact.load(target, strict=True)

    assert loaded.meta["feature_keys"]["node_attr_keys"][-7:] == [
        "interface.is_interface",
        "secondary_structure.helix",
        "secondary_structure.sheet",
        "secondary_structure.coil",
        "secondary_structure.unassigned",
        "sasa",
        "rsa",
    ]
    np.testing.assert_array_equal(loaded.data["node_attr"], artifact.data["node_attr"])
    np.testing.assert_array_equal(loaded.data["edge_index"], artifact.data["edge_index"])


def test_sidechain_definition_table_covers_frozen_depths_and_terminal_choices() -> None:
    assert {name: len(definitions) for name, definitions in SIDECHAIN_CHI_DEFINITIONS.items()} == {
        "ARG": 4,
        "ASN": 2,
        "ASP": 2,
        "CYS": 1,
        "GLN": 3,
        "GLU": 3,
        "HIS": 2,
        "ILE": 2,
        "LEU": 2,
        "LYS": 4,
        "MET": 3,
        "PHE": 2,
        "PRO": 2,
        "SER": 1,
        "THR": 1,
        "TRP": 2,
        "TYR": 2,
        "VAL": 1,
    }
    assert "ALA" not in SIDECHAIN_CHI_DEFINITIONS
    assert "GLY" not in SIDECHAIN_CHI_DEFINITIONS
    for residue_name in ("ASN", "ASP"):
        assert SIDECHAIN_CHI_DEFINITIONS[residue_name][1][-1] == "OD1"
    for residue_name in ("LEU", "PHE", "TRP", "TYR"):
        assert SIDECHAIN_CHI_DEFINITIONS[residue_name][1][-1] == "CD1"
    assert SIDECHAIN_CHI_DEFINITIONS["HIS"][1][-1] == "ND1"
    assert SIDECHAIN_CHI_DEFINITIONS["ILE"][0][-1] == "CG1"
    assert SIDECHAIN_CHI_DEFINITIONS["VAL"][0][-1] == "CG1"


def test_sidechain_torsions_known_angle_depths_and_undefined_policies() -> None:
    missing_lys = _chi_atoms("LYS")
    del missing_lys["CE"]
    degenerate_ser = _chi_atoms("SER")
    degenerate_ser["OG"] = degenerate_ser["CB"]
    nonfinite_ser = _chi_atoms("SER")
    nonfinite_ser["OG"] = (np.nan, 1.0, 1.0)
    coordinates, residue_index, atom_name, resname = _flatten_test_atoms(
        [
            ("ALA", {name: _CHI_TEST_COORDINATES[name] for name in ("N", "CA", "CB")}),
            ("GLY", {name: _CHI_TEST_COORDINATES[name] for name in ("N", "CA")}),
            ("SER", _chi_atoms("SER")),
            ("ASN", _chi_atoms("ASN")),
            ("GLN", _chi_atoms("GLN")),
            ("LYS", _chi_atoms("LYS")),
            ("MSE", _chi_atoms("MET")),
            ("LYS", missing_lys),
            ("SER", degenerate_ser),
            ("SER", nonfinite_ser),
        ]
    )

    angles = sidechain_torsions(coordinates, residue_index, atom_name, resname)

    assert angles.shape == (10, 4)
    np.testing.assert_array_equal(
        np.isfinite(angles).sum(axis=1),
        [0, 0, 1, 2, 3, 4, 0, 2, 0, 0],
    )
    assert angles[2, 0] == pytest.approx(np.pi / 2.0, abs=1e-7)
    assert np.isnan(angles[0]).all()
    assert np.isnan(angles[1]).all()
    assert np.isnan(angles[6]).all()
    assert np.isnan(angles[7, 2:]).all()


def test_sidechain_symmetric_terminal_convention_uses_exact_frozen_atom() -> None:
    atoms = _chi_atoms("ASP")
    atoms["OD2"] = (2.0, -1.0, 2.0)
    coordinates, residue_index, atom_name, resname = _flatten_test_atoms([("ASP", atoms)])

    angles = sidechain_torsions(coordinates, residue_index, atom_name, resname)
    expected = signed_dihedral(*(np.asarray(atoms[name]) for name in ("CA", "CB", "CG", "OD1")))
    alternate = signed_dihedral(*(np.asarray(atoms[name]) for name in ("CA", "CB", "CG", "OD2")))

    assert angles[0, 1] == pytest.approx(expected, abs=1e-7)
    assert angles[0, 1] != pytest.approx(alternate, abs=1e-7)


def test_sidechain_geometry_modes_have_fixed_width_keys_and_order() -> None:
    coordinates, residue_index, atom_name, resname = _flatten_test_atoms(
        [("SER", _chi_atoms("SER"))]
    )
    common = {
        "resname": resname,
        "polymer_heavy_atom_coordinates": coordinates,
        "polymer_heavy_atom_residue_index": residue_index,
        "polymer_heavy_atom_name": atom_name,
    }
    periodic_keys = [
        f"sidechain.chi{chi}_{component}" for chi in range(1, 5) for component in ("sin", "cos")
    ]
    raw_keys = [f"sidechain.chi{chi}" for chi in range(1, 5)]

    sincos, sincos_keys = build_node_attr(
        _aa_indices("S"),
        mode=["sidechain_geometry"],
        return_keys=True,
        **common,
    )
    raw, returned_raw_keys = build_node_attr(
        _aa_indices("S"),
        mode=[{"sidechain_geometry": {"mode": "raw"}}],
        return_keys=True,
        **common,
    )
    both, both_keys = build_node_attr(
        _aa_indices("S"),
        mode=[{"sidechain_geometry": {"mode": "both"}}],
        return_keys=True,
        **common,
    )

    assert sincos.shape == (1, 8)
    assert raw.shape == (1, 4)
    assert both.shape == (1, 12)
    assert sincos_keys == periodic_keys
    assert returned_raw_keys == raw_keys
    assert both_keys == [*periodic_keys, *raw_keys]
    np.testing.assert_allclose(
        sincos[:, :2],
        np.column_stack((np.sin(raw[:, 0]), np.cos(raw[:, 0]))),
        atol=1e-7,
    )
    assert np.isnan(sincos[:, 2:]).all()
    np.testing.assert_allclose(both[:, :8], sincos, equal_nan=True)
    np.testing.assert_allclose(both[:, 8:], raw, equal_nan=True)


def test_sidechain_torsions_are_translation_and_rotation_invariant() -> None:
    coordinates, residue_index, atom_name, resname = _flatten_test_atoms(
        [("LYS", _chi_atoms("LYS"))]
    )
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    translation = np.asarray([12.5, -7.25, 3.0], dtype=np.float32)

    expected = sidechain_torsions(coordinates, residue_index, atom_name, resname)
    translated = sidechain_torsions(
        coordinates + translation,
        residue_index,
        atom_name,
        resname,
    )
    rotated = sidechain_torsions(
        coordinates @ rotation.T,
        residue_index,
        atom_name,
        resname,
    )

    np.testing.assert_allclose(translated, expected, atol=1e-6, equal_nan=True)
    np.testing.assert_allclose(rotated, expected, atol=1e-6, equal_nan=True)


def test_disulfide_geometry_panel_retains_cysteine_sidechain_atoms(tmp_path: Path) -> None:
    inp = _write_heavy_atom_structure(
        tmp_path / "cysteine-pair.pdb",
        [
            (
                "A",
                1,
                "CYS",
                {
                    "N": (0.0, 1.0, 0.0),
                    "CA": (0.0, 0.0, 0.0),
                    "C": (1.0, -1.0, 0.0),
                    "O": (2.0, -1.0, 0.0),
                    "CB": (0.0, 0.0, 1.0),
                    "SG": (1.0, 0.0, 1.0),
                },
            ),
            (
                "A",
                2,
                "CYS",
                {
                    "N": (5.0, 1.0, 0.0),
                    "CA": (4.0, 0.0, 0.0),
                    "C": (5.0, -1.0, 0.0),
                    "O": (6.0, -1.0, 0.0),
                    "CB": (4.0, 0.0, 1.0),
                    "SG": (3.0, 0.0, 1.0),
                },
            ),
        ],
    )
    loaded = load_and_extract_residue(inp)
    sulfur_mask = loaded["polymer_heavy_atom_name"] == "SG"
    sulfur_coordinates = loaded["polymer_heavy_atom_coordinates"][sulfur_mask]
    assert sulfur_coordinates.shape == (2, 3)
    assert np.linalg.norm(sulfur_coordinates[0] - sulfur_coordinates[1]) == pytest.approx(2.0)

    artifact = ResiduePointCloudCA(
        node_attr_mode=[{"sidechain_geometry": {"mode": "raw"}}],
        use_cache=False,
    ).encode(inp)
    assert artifact.data["residue_ids"].tolist() == ["A:1::CYS", "A:2::CYS"]
    assert artifact.meta["feature_keys"]["node_attr_keys"] == [
        "sidechain.chi1",
        "sidechain.chi2",
        "sidechain.chi3",
        "sidechain.chi4",
    ]
    assert np.isfinite(artifact.data["node_attr"][:, 0]).all()
    assert np.isnan(artifact.data["node_attr"][:, 1:]).all()


def test_sidechain_geometry_hybrid_composition_and_artifact_round_trip(tmp_path: Path) -> None:
    inp = _write_heavy_atom_structure(
        tmp_path / "sidechain-hybrid.pdb",
        [
            (
                "A",
                1,
                "SER",
                {
                    "N": (1.0, 0.0, 0.0),
                    "CA": (0.0, 0.0, 0.0),
                    "C": (-1.0, 1.0, 0.0),
                    "O": (-2.0, 1.0, 0.0),
                    "CB": (0.0, 0.0, 1.0),
                    "OG": (0.0, 1.0, 1.0),
                },
            )
        ],
    )
    artifact = ResiduePointCloudCA(
        node_attr_mode=[
            "rsa",
            "secondary_structure",
            "sidechain_geometry",
            "backbone_geometry",
            "physchem",
            "onehot",
        ],
        use_cache=False,
    ).encode(inp)
    repeated = ResiduePointCloudCA(
        node_attr_mode=[
            "onehot",
            "physchem",
            "backbone_geometry",
            "sidechain_geometry",
            "secondary_structure",
            "rsa",
        ],
        use_cache=False,
    ).encode(inp)
    target = tmp_path / "sidechain-artifact"
    artifact.save(target)
    loaded = StructureArtifact.load(target, strict=True)

    sidechain_keys = [
        f"sidechain.chi{chi}_{component}" for chi in range(1, 5) for component in ("sin", "cos")
    ]
    keys = loaded.meta["feature_keys"]["node_attr_keys"]
    assert keys[: len(AA_VOCAB)] == [f"aa_onehot.{aa}" for aa in AA_VOCAB]
    assert keys[len(AA_VOCAB) : len(AA_VOCAB) + len(PHYSchem_KEYS)] == PHYSchem_KEYS
    sidechain_start = keys.index("sidechain.chi1_sin")
    assert keys[sidechain_start : sidechain_start + 8] == sidechain_keys
    assert keys[-5:] == [
        "secondary_structure.helix",
        "secondary_structure.sheet",
        "secondary_structure.coil",
        "secondary_structure.unassigned",
        "rsa",
    ]
    np.testing.assert_array_equal(artifact.data["node_attr"], repeated.data["node_attr"])
    np.testing.assert_array_equal(loaded.data["node_attr"], artifact.data["node_attr"])
    assert (
        loaded.meta["feature_keys"]["node_attr_keys"]
        == artifact.meta["feature_keys"]["node_attr_keys"]
    )
    sidechain_params = next(
        token["sidechain_geometry"]
        for token in artifact.meta["parameters"]["node_attr_mode"]
        if isinstance(token, dict) and "sidechain_geometry" in token
    )
    assert sidechain_params == {
        "atom_definition_convention": "standard_residue_specific_chi1_through_chi4",
        "atom_definition_table": "eris.frozen_sidechain_chi_atoms.v1",
        "missing_value_policy": "nan",
        "mode": "sincos",
        "torsion_convention": "standard_signed_atan2[-pi,pi)",
        "units": "radians",
    }


def test_v01_high_content_feature_export_pooling_and_rigid_transform_panel(
    tmp_path: Path,
) -> None:
    transformed_source = tmp_path / "1AQ2-rigid-transform.pdb"
    _rigid_transform_pdb(PDB, transformed_source)
    selected = load_and_extract_residue(StructureInput(source=str(PDB)))
    external_keys = [f"embedding_{index:02d}" for index in range(32)]
    external_path = _write_external_features(
        tmp_path / "external-high-dimensional.npz",
        values=(
            np.arange(
                len(selected["residue_ids"]) * len(external_keys),
                dtype=np.float32,
            ).reshape(len(selected["residue_ids"]), len(external_keys))
            / 100.0
        ),
        residue_ids=selected["residue_ids"].tolist(),
        feature_keys=external_keys,
        chain_ids=list(selected.chain_ids),
        model_index=selected.model_index,
    )
    node_mode = [
        "rsa",
        "sasa",
        "bfactor_ca",
        "secondary_structure",
        {"sidechain_geometry": {"mode": "both"}},
        {"backbone_geometry": {"mode": "both"}},
        {"external": {"path": str(external_path)}},
        *_aaindex_mode(["KYTJ820101", "FAUJ880103"]),
        "physchem",
        "onehot",
        "interface",
    ]
    edge_mode = [
        {"interface_contact": {"cutoff": 5.0}},
        "relative_direction",
        "inter_chain",
        "same_chain",
        "seq",
        "rbf",
        "dist",
    ]

    def encode(path: Path):
        return ResidueKNNGraph(
            k=3,
            node_attr_mode=node_mode,
            edge_attr_mode=edge_mode,
            use_cache=False,
        ).encode(StructureInput(source=str(path)))

    original = encode(PDB)
    transformed = encode(transformed_source)
    periodic_backbone_keys = [
        "backbone.phi_sin",
        "backbone.phi_cos",
        "backbone.psi_sin",
        "backbone.psi_cos",
        "backbone.omega_sin",
        "backbone.omega_cos",
    ]
    raw_backbone_keys = ["backbone.phi", "backbone.psi", "backbone.omega"]
    periodic_sidechain_keys = [
        f"sidechain.chi{chi}_{component}" for chi in range(1, 5) for component in ("sin", "cos")
    ]
    raw_sidechain_keys = [f"sidechain.chi{chi}" for chi in range(1, 5)]
    expected_node_keys = [
        *[f"aa_onehot.{name}" for name in AA_VOCAB],
        *PHYSchem_KEYS,
        "aaindex.KYTJ820101",
        "aaindex.FAUJ880103",
        "bfactor_ca",
        *periodic_backbone_keys,
        *raw_backbone_keys,
        *periodic_sidechain_keys,
        *raw_sidechain_keys,
        *[f"external.custom.{key}" for key in external_keys],
        "interface.is_interface",
        "secondary_structure.helix",
        "secondary_structure.sheet",
        "secondary_structure.coil",
        "secondary_structure.unassigned",
        "sasa",
        "rsa",
    ]
    expected_edge_keys = [
        "dist",
        *[f"rbf.{index}" for index in range(16)],
        "seq_sep",
        "edge.same_chain",
        "edge.inter_chain",
        "edge.relative_direction_x",
        "edge.relative_direction_y",
        "edge.relative_direction_z",
        "edge.interface_contact",
    ]
    assert original.meta["feature_keys"]["node_attr_keys"] == expected_node_keys
    assert original.meta["feature_keys"]["edge_attr_keys"] == expected_edge_keys
    assert original.data["node_attr"].shape == (
        len(selected["residue_ids"]),
        len(expected_node_keys),
    )
    assert original.data["edge_attr"].shape == (original.data["edge_index"].shape[1], 24)
    assert original.data["node_attr"].dtype == np.float32
    assert original.data["edge_attr"].dtype == np.float32
    assert original.meta["input"]["chain_ids"] == ["A"]
    assert (
        original.meta["provenance"]["config_fingerprint"]
        == transformed.meta["provenance"]["config_fingerprint"]
    )
    assert original.meta["input"]["source_sha256"] != transformed.meta["input"]["source_sha256"]
    assert not np.allclose(original.data["pos"], transformed.data["pos"])
    sasa_index = expected_node_keys.index("sasa")
    rsa_index = expected_node_keys.index("rsa")
    invariant_indices = [
        index for index in range(len(expected_node_keys)) if index not in {sasa_index, rsa_index}
    ]
    np.testing.assert_allclose(
        original.data["node_attr"][:, invariant_indices],
        transformed.data["node_attr"][:, invariant_indices],
        atol=1e-5,
        equal_nan=True,
    )
    valid_sasa = np.isfinite(original.data["node_attr"][:, sasa_index]) & np.isfinite(
        transformed.data["node_attr"][:, sasa_index]
    )
    original_sasa = original.data["node_attr"][valid_sasa, sasa_index]
    transformed_sasa = transformed.data["node_attr"][valid_sasa, sasa_index]
    assert abs(original_sasa.sum() - transformed_sasa.sum()) / original_sasa.sum() < 0.002
    valid_rsa = np.isfinite(original.data["node_attr"][:, rsa_index]) & np.isfinite(
        transformed.data["node_attr"][:, rsa_index]
    )
    rsa_delta = np.abs(
        original.data["node_attr"][valid_rsa, rsa_index]
        - transformed.data["node_attr"][valid_rsa, rsa_index]
    )
    assert np.mean(rsa_delta) < 0.003
    np.testing.assert_array_equal(original.data["edge_index"], transformed.data["edge_index"])
    np.testing.assert_allclose(original.data["edge_dist"], transformed.data["edge_dist"], atol=1e-5)
    np.testing.assert_allclose(original.data["edge_attr"], transformed.data["edge_attr"], atol=1e-5)
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        transformed.data["pos"],
        original.data["pos"] @ rotation.T,
        atol=2e-4,
    )

    artifact_path = tmp_path / "high-content-graph"
    original.save(artifact_path)
    loaded = StructureArtifact.load(artifact_path, strict=True)
    assert loaded.meta["schema_version"] == 2
    assert loaded.meta["feature_keys"] == original.meta["feature_keys"]
    assert loaded.meta["provenance"] == original.meta["provenance"]
    assert loaded.data["chain_id"].tolist() == original.data["chain_id"].tolist()
    for name, values in original.data.items():
        assert loaded.data[name].dtype == values.dtype
        assert loaded.data[name].shape == values.shape
        np.testing.assert_equal(loaded.data[name], values)

    torch = pytest.importorskip("torch")
    torch_graph = eris.to_torch(loaded, prefer="graph")
    for exported_name, artifact_name in (
        ("x", "node_attr"),
        ("edge_attr", "edge_attr"),
        ("edge_index", "edge_index"),
    ):
        np.testing.assert_equal(
            torch_graph[exported_name].cpu().numpy(),
            loaded.data[artifact_name],
        )
    assert torch_graph["x"].dtype == torch.float32
    if find_spec("torch_geometric") is not None:
        pyg_graph = eris.to_pyg(loaded, prefer="graph")
        np.testing.assert_equal(pyg_graph.x.cpu().numpy(), loaded.data["node_attr"])
        np.testing.assert_equal(pyg_graph.edge_attr.cpu().numpy(), loaded.data["edge_attr"])

    meta_before_pool = deepcopy(loaded.meta)
    data_before_pool = {name: values.copy() for name, values in loaded.data.items()}
    pooled = eris.pool(loaded, mode="mean_std")
    transformed_pool = eris.pool(transformed, mode="mean_std")
    expected_pooled_keys = [
        *[f"pool.mean.{key}" for key in expected_node_keys],
        *[f"pool.std.{key}" for key in expected_node_keys],
    ]
    assert pooled.data["pooled_attr"].shape == (2 * len(expected_node_keys),)
    assert pooled.data["pooled_attr"].dtype == np.float64
    assert pooled.meta["feature_keys"]["pooled_attr_keys"] == expected_pooled_keys
    invariant_pool_indices = [
        *invariant_indices,
        *(index + len(expected_node_keys) for index in invariant_indices),
    ]
    np.testing.assert_allclose(
        pooled.data["pooled_attr"][invariant_pool_indices],
        transformed_pool.data["pooled_attr"][invariant_pool_indices],
        atol=1e-5,
        equal_nan=True,
    )
    assert loaded.meta == meta_before_pool
    for name, values in data_before_pool.items():
        np.testing.assert_equal(loaded.data[name], values)

    pooled_path = tmp_path / "high-content-pooled"
    pooled.save(pooled_path)
    loaded_pool = StructureArtifact.load(pooled_path, strict=True)
    assert loaded_pool.meta["schema_version"] == 2
    assert loaded_pool.data["pooled_attr"].dtype == np.float64
    assert loaded_pool.meta["feature_keys"]["pooled_attr_keys"] == expected_pooled_keys
    np.testing.assert_equal(loaded_pool.data["pooled_attr"], pooled.data["pooled_attr"])
