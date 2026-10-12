from __future__ import annotations

from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest

import eris
from eris.core.artifact_schema import compute_config_fingerprint
from eris.core.artifacts import StructureArtifact
from eris.core.cache import DiskCache, compute_cache_key
from eris.core.hashing import sha256_file
from eris.core.types import StructureInput
from eris.encoders.graphs.residue_contact_graph import ResidueContactGraph
from eris.encoders.graphs.residue_knn_graph import ResidueKNNGraph
from eris.features.graph_stats import compute_graph_stats


def _atom_line(
    serial: int,
    atom_name: str,
    residue_name: str,
    residue_number: int,
    x: float,
    *,
    chain_id: str = "A",
    insertion_code: str = " ",
) -> str:
    return (
        f"ATOM  {serial:5d} {atom_name:>4s} {residue_name:>3s} {chain_id}"
        f"{residue_number:4d}{insertion_code}   {x:8.3f}{0.0:8.3f}{0.0:8.3f}"
        f"{1.0:6.2f}{10.0:6.2f}           C  "
    )


def _write_structure(
    path: Path,
    residues: list[tuple[int, str, float, str, str, str]],
) -> StructureInput:
    lines: list[str] = []
    previous_chain: str | None = None
    for serial, (number, insertion_code, x, residue_name, atom_name, chain_id) in enumerate(
        residues,
        start=1,
    ):
        if previous_chain is not None and chain_id != previous_chain:
            lines.append("TER")
        lines.append(
            _atom_line(
                serial,
                atom_name,
                residue_name,
                number,
                x,
                chain_id=chain_id,
                insertion_code=insertion_code,
            )
        )
        previous_chain = chain_id
    lines.extend(("TER", "END"))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return StructureInput(source=str(path), chain_id="A")


def _backbone_atom_line(
    serial: int,
    atom_name: str,
    residue_name: str,
    residue_number: int,
    coordinate: np.ndarray,
    *,
    chain_id: str,
) -> str:
    x, y, z = coordinate.tolist()
    return (
        f"ATOM  {serial:5d} {atom_name:>4s} {residue_name:>3s} {chain_id}"
        f"{residue_number:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
        f"{1.0:6.2f}{10.0:6.2f}           {atom_name[0]:>2s}  "
    )


def _write_backbone_graph_structure(
    path: Path,
    residues: list[tuple[str, int, str, dict[str, np.ndarray]]],
) -> StructureInput:
    lines: list[str] = []
    serial = 1
    previous_chain: str | None = None
    for chain_id, residue_number, residue_name, atoms in residues:
        if previous_chain is not None and chain_id != previous_chain:
            lines.append("TER")
        for atom_name in ("N", "CA", "C"):
            coordinate = atoms.get(atom_name)
            if coordinate is None:
                continue
            lines.append(
                _backbone_atom_line(
                    serial,
                    atom_name,
                    residue_name,
                    residue_number,
                    coordinate,
                    chain_id=chain_id,
                )
            )
            serial += 1
        previous_chain = chain_id
    path.write_text("\n".join([*lines, "TER", "END", ""]), encoding="ascii")
    return StructureInput(source=str(path))


def _transform_residues(
    residues: list[tuple[str, int, str, dict[str, np.ndarray]]],
    rotation: np.ndarray,
    translation: np.ndarray,
) -> list[tuple[str, int, str, dict[str, np.ndarray]]]:
    return [
        (
            chain_id,
            residue_number,
            residue_name,
            {atom_name: coordinate @ rotation.T + translation for atom_name, coordinate in atoms.items()},
        )
        for chain_id, residue_number, residue_name, atoms in residues
    ]


GRAPH_ENCODERS = (ResidueKNNGraph, ResidueContactGraph)


def _encode(encoder_type: type, structure_input: StructureInput):
    kwargs: dict[str, object] = {"use_cache": False, "local_cutoff": 1}
    if encoder_type is ResidueKNNGraph:
        kwargs.update(k=32, cutoff=None, symmetric=True)
    else:
        kwargs.update(cutoff=100.0)
    return encoder_type(**kwargs).encode(structure_input)


def _edge_feature(artifact, source_id: str, target_id: str, feature_key: str) -> float:
    source = artifact.data["residue_ids"].tolist().index(source_id)
    target = artifact.data["residue_ids"].tolist().index(target_id)
    pairs = [tuple(pair) for pair in artifact.data["edge_index"].T.tolist()]
    row = pairs.index((source, target))
    column = artifact.meta["feature_keys"]["edge_attr_keys"].index(feature_key)
    return float(artifact.data["edge_attr"][row, column])


def _edge_features(
    artifact,
    source_id: str,
    target_id: str,
    feature_keys: list[str],
) -> np.ndarray:
    source = artifact.data["residue_ids"].tolist().index(source_id)
    target = artifact.data["residue_ids"].tolist().index(target_id)
    pairs = [tuple(pair) for pair in artifact.data["edge_index"].T.tolist()]
    row = pairs.index((source, target))
    keys = artifact.meta["feature_keys"]["edge_attr_keys"]
    columns = [keys.index(key) for key in feature_keys]
    return artifact.data["edge_attr"][row, columns]


def _edge_rows(artifact) -> dict[tuple[int, int], tuple[float, np.ndarray]]:
    return {
        tuple(pair): (float(artifact.data["edge_dist"][row]), artifact.data["edge_attr"][row])
        for row, pair in enumerate(artifact.data["edge_index"].T.tolist())
    }


def test_graph_stats_accept_finite_edge_distances() -> None:
    edge_index = np.asarray([[0], [1]], dtype=np.int64)

    stats = compute_graph_stats(edge_index, np.asarray([1.25]), n_nodes=2)

    assert stats["edge_dist_mean"] == pytest.approx(1.25)


@pytest.mark.parametrize("invalid_distance", [np.nan, np.inf, -np.inf])
def test_graph_stats_reject_nonfinite_edge_distances(invalid_distance: float) -> None:
    edge_index = np.asarray([[0], [1]], dtype=np.int64)

    with pytest.raises(ValueError, match="finite"):
        compute_graph_stats(edge_index, np.asarray([invalid_distance]), n_nodes=2)


def test_graph_builders_enforce_knn_and_contact_neighborhoods(tmp_path: Path) -> None:
    structure_input = _write_structure(
        tmp_path / "neighborhoods.pdb",
        [
            (1, " ", 0.0, "ALA", "CA", "A"),
            (2, " ", 1.0, "GLY", "CA", "A"),
            (3, " ", 10.0, "SER", "CA", "A"),
        ],
    )
    knn = ResidueKNNGraph(k=1, cutoff=None, symmetric=True, use_cache=False).encode(structure_input)
    contact = ResidueContactGraph(cutoff=2.0, use_cache=False).encode(structure_input)

    knn_pairs = {tuple(pair) for pair in knn.data["edge_index"].T.tolist()}
    contact_pairs = {tuple(pair) for pair in contact.data["edge_index"].T.tolist()}
    assert knn_pairs == {(0, 1), (1, 0), (1, 2), (2, 1)}
    assert contact_pairs == {(0, 1), (1, 0)}
    np.testing.assert_array_equal(contact.data["edge_dist"], [1.0, 1.0])


@pytest.mark.parametrize("encoder_type", GRAPH_ENCODERS)
def test_residue_graph_sequence_features_preserve_numeric_gaps_and_locality(
    encoder_type: type,
    tmp_path: Path,
) -> None:
    structure_input = _write_structure(
        tmp_path / "numeric-gap.pdb",
        [
            (16, " ", 0.0, "ALA", "CA", "A"),
            (17, " ", 1.0, "GLY", "CA", "A"),
            (20, " ", 2.0, "SER", "CA", "A"),
        ],
    )
    artifact = _encode(encoder_type, structure_input)

    assert artifact.data["residue_ids"].tolist() == ["A:16::ALA", "A:17::GLY", "A:20::SER"]
    assert _edge_feature(artifact, "A:16::ALA", "A:17::GLY", "seq_sep") == 1.0
    assert _edge_feature(artifact, "A:17::GLY", "A:20::SER", "seq_sep") == 3.0
    assert _edge_feature(artifact, "A:17::GLY", "A:20::SER", "flag.is_local_le_1") == 0.0


@pytest.mark.parametrize("encoder_type", GRAPH_ENCODERS)
def test_missing_coordinate_skip_does_not_collapse_sequence_gap(
    encoder_type: type,
    tmp_path: Path,
) -> None:
    structure_input = _write_structure(
        tmp_path / "missing-coordinate.pdb",
        [
            (17, " ", 0.0, "ALA", "CA", "A"),
            (18, " ", 1.0, "GLY", "N", "A"),
            (20, " ", 2.0, "SER", "CA", "A"),
        ],
    )
    artifact = _encode(encoder_type, structure_input)

    assert artifact.data["residue_ids"].tolist() == ["A:17::ALA", "A:20::SER"]
    assert artifact.data["omitted_residue_ids"].tolist() == ["A:18::GLY"]
    assert _edge_feature(artifact, "A:17::ALA", "A:20::SER", "seq_sep") == 3.0
    assert _edge_feature(artifact, "A:17::ALA", "A:20::SER", "flag.is_local_le_1") == 0.0


@pytest.mark.parametrize("encoder_type", GRAPH_ENCODERS)
def test_insertion_codes_preserve_source_order_adjacency(
    encoder_type: type,
    tmp_path: Path,
) -> None:
    structure_input = _write_structure(
        tmp_path / "insertions.pdb",
        [
            (20, " ", 0.0, "ALA", "CA", "A"),
            (20, "A", 1.0, "GLY", "CA", "A"),
            (20, "B", 2.0, "SER", "CA", "A"),
            (21, " ", 3.0, "THR", "CA", "A"),
        ],
    )
    artifact = _encode(encoder_type, structure_input)

    residue_ids = artifact.data["residue_ids"].tolist()
    assert residue_ids == ["A:20::ALA", "A:20:A:GLY", "A:20:B:SER", "A:21::THR"]
    for source_id, target_id in zip(residue_ids, residue_ids[1:]):
        assert _edge_feature(artifact, source_id, target_id, "seq_sep") == 1.0
        assert _edge_feature(artifact, source_id, target_id, "flag.is_local_le_1") == 1.0


@pytest.mark.parametrize("encoder_type", GRAPH_ENCODERS)
def test_residue_graphs_select_one_chain_without_cross_chain_edges(
    encoder_type: type,
    tmp_path: Path,
) -> None:
    structure_input = _write_structure(
        tmp_path / "two-chains.pdb",
        [
            (1, " ", 0.0, "ALA", "CA", "A"),
            (2, " ", 1.0, "GLY", "CA", "A"),
            (1, " ", 2.0, "SER", "CA", "B"),
            (2, " ", 3.0, "THR", "CA", "B"),
        ],
    )
    artifact = _encode(encoder_type, structure_input)

    assert artifact.data["residue_ids"].tolist() == ["A:1::ALA", "A:2::GLY"]


@pytest.mark.parametrize("encoder_type", GRAPH_ENCODERS)
def test_residue_graphs_include_cross_chain_edges_with_frozen_sequence_sentinel(
    encoder_type: type,
    tmp_path: Path,
) -> None:
    explicit_a = _write_structure(
        tmp_path / "two-chains-all.pdb",
        [
            (1, " ", 0.0, "ALA", "CA", "A"),
            (2, " ", 1.0, "GLY", "CA", "A"),
            (1, " ", 1.5, "SER", "CA", "B"),
            (2, " ", 2.5, "THR", "CA", "B"),
        ],
    )
    structure_input = StructureInput(source=explicit_a.source)

    artifact = _encode(encoder_type, structure_input)

    assert artifact.meta["input"]["chain_ids"] == ["A", "B"]
    assert artifact.data["chain_id"].tolist() == ["A", "A", "B", "B"]
    assert artifact.data["residue_ids"].tolist() == [
        "A:1::ALA",
        "A:2::GLY",
        "B:1::SER",
        "B:2::THR",
    ]
    assert _edge_feature(artifact, "A:2::GLY", "B:1::SER", "seq_sep") == -1.0
    assert _edge_feature(artifact, "A:2::GLY", "B:1::SER", "flag.is_local_le_1") == 0.0


def _orientation_residues() -> list[tuple[str, int, str, dict[str, np.ndarray]]]:
    return [
        (
            "A",
            1,
            "ALA",
            {
                "N": np.asarray([0.0, 1.0, 0.0]),
                "CA": np.asarray([0.0, 0.0, 0.0]),
                "C": np.asarray([1.0, 0.0, 0.0]),
            },
        ),
        (
            "A",
            2,
            "GLY",
            {
                "N": np.asarray([1.0, 1.0, 1.0]),
                "CA": np.asarray([1.0, 1.0, 0.0]),
                "C": np.asarray([1.0, 2.0, 0.0]),
            },
        ),
        (
            "B",
            1,
            "SER",
            {
                "N": np.asarray([1.5, 0.5, 1.0]),
                "CA": np.asarray([0.5, 0.5, 1.0]),
                "C": np.asarray([0.5, 0.5, 2.0]),
            },
        ),
    ]


def _interface_residues() -> list[tuple[str, int, str, dict[str, np.ndarray]]]:
    def backbone(ca_x: float) -> dict[str, np.ndarray]:
        return {
            "N": np.asarray([ca_x - 1.0, 0.0, 0.0]),
            "CA": np.asarray([ca_x, 0.0, 0.0]),
            "C": np.asarray([ca_x + 1.0, 0.0, 0.0]),
        }

    return [
        ("A", 1, "ALA", backbone(0.0)),
        ("A", 2, "GLY", backbone(30.0)),
        ("B", 1, "SER", backbone(7.0)),
        ("C", 1, "THR", backbone(60.0)),
    ]


def _encode_interface_graph(
    encoder_type: type,
    structure_input: StructureInput,
    *,
    interface_cutoff: float,
):
    kwargs: dict[str, object] = {
        "node_attr_mode": [{"interface": {"cutoff": interface_cutoff}}],
        "edge_attr_mode": [{"interface_contact": {"cutoff": interface_cutoff}}],
        "use_cache": False,
    }
    if encoder_type is ResidueKNNGraph:
        kwargs.update(k=8, cutoff=None, symmetric=True)
    else:
        kwargs.update(cutoff=7.0)
    return encoder_type(**kwargs).encode(structure_input)


@pytest.mark.parametrize("encoder_type", GRAPH_ENCODERS)
def test_interface_features_use_heavy_atoms_without_changing_topology(
    encoder_type: type,
    tmp_path: Path,
) -> None:
    structure_input = _write_backbone_graph_structure(
        tmp_path / "interface.pdb",
        _interface_residues(),
    )
    baseline_kwargs: dict[str, object] = {"use_cache": False}
    if encoder_type is ResidueKNNGraph:
        baseline_kwargs.update(k=8, cutoff=None, symmetric=True)
    else:
        baseline_kwargs.update(cutoff=7.0)
    baseline = encoder_type(**baseline_kwargs).encode(structure_input)
    artifact = _encode_interface_graph(
        encoder_type,
        structure_input,
        interface_cutoff=5.0,
    )

    np.testing.assert_array_equal(artifact.data["edge_index"], baseline.data["edge_index"])
    assert artifact.meta["feature_keys"]["node_attr_keys"] == ["interface.is_interface"]
    assert artifact.meta["feature_keys"]["edge_attr_keys"] == ["edge.interface_contact"]
    assert artifact.data["interface_residue_ids"].tolist() == ["A:1::ALA", "B:1::SER"]
    np.testing.assert_array_equal(artifact.data["node_attr"][:, 0], [1.0, 0.0, 1.0, 0.0])
    assert (
        _edge_feature(
            artifact,
            "A:1::ALA",
            "B:1::SER",
            "edge.interface_contact",
        )
        == 1.0
    )
    for source, target in artifact.data["edge_index"].T.tolist():
        source_id = artifact.data["residue_ids"][source]
        target_id = artifact.data["residue_ids"][target]
        if {source_id, target_id} != {"A:1::ALA", "B:1::SER"}:
            assert (
                _edge_feature(
                    artifact,
                    str(source_id),
                    str(target_id),
                    "edge.interface_contact",
                )
                == 0.0
            )

    params = artifact.meta["parameters"]
    assert params["node_attr_mode"][0]["interface"]["cutoff"] == 5.0
    assert params["edge_attr_mode"][0]["interface_contact"]["cutoff"] == 5.0


@pytest.mark.parametrize("encoder_type", GRAPH_ENCODERS)
def test_interface_cutoff_chain_filtering_round_trip_and_fingerprint(
    encoder_type: type,
    tmp_path: Path,
) -> None:
    structure_input = _write_backbone_graph_structure(
        tmp_path / "interface-selection.pdb",
        _interface_residues(),
    )
    contacting = _encode_interface_graph(encoder_type, structure_input, interface_cutoff=5.0)
    below_contact = _encode_interface_graph(encoder_type, structure_input, interface_cutoff=4.9)
    filtered = _encode_interface_graph(
        encoder_type,
        StructureInput(source=structure_input.source, chain_id=("A", "C")),
        interface_cutoff=5.0,
    )

    assert contacting.data["interface_residue_ids"].tolist() == ["A:1::ALA", "B:1::SER"]
    assert below_contact.data["interface_residue_ids"].tolist() == []
    assert (
        _edge_feature(
            below_contact,
            "A:1::ALA",
            "B:1::SER",
            "edge.interface_contact",
        )
        == 0.0
    )
    assert filtered.meta["input"]["chain_ids"] == ["A", "C"]
    assert filtered.data["interface_residue_ids"].tolist() == []
    assert (
        contacting.meta["provenance"]["config_fingerprint"]
        != below_contact.meta["provenance"]["config_fingerprint"]
    )

    output = tmp_path / f"{encoder_type.__name__}-interface"
    contacting.save(output)
    loaded = StructureArtifact.load(output, strict=True)
    assert loaded.meta == contacting.meta
    for name in contacting.data:
        np.testing.assert_equal(loaded.data[name], contacting.data[name])


def _encode_orientation_graph(encoder_type: type, structure_input: StructureInput):
    edge_mode = [
        "relative_direction",
        "inter_chain",
        "rbf",
        "dist",
        "same_chain",
        "seq",
        "local",
    ]
    if encoder_type is ResidueKNNGraph:
        return encoder_type(
            k=2,
            cutoff=None,
            symmetric=True,
            edge_attr_mode=edge_mode,
            use_cache=False,
        ).encode(structure_input)
    return encoder_type(cutoff=3.0, edge_attr_mode=edge_mode, use_cache=False).encode(structure_input)


@pytest.mark.parametrize("encoder_type", GRAPH_ENCODERS)
def test_orientation_edges_are_invariant_directed_and_multichain(
    encoder_type: type,
    tmp_path: Path,
) -> None:
    residues = _orientation_residues()
    original_input = _write_backbone_graph_structure(tmp_path / "orientation.pdb", residues)
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    translation = np.asarray([8.0, -5.0, 3.0], dtype=np.float64)
    translated_input = _write_backbone_graph_structure(
        tmp_path / "orientation-translated.pdb",
        _transform_residues(residues, np.eye(3), translation),
    )
    rotated_input = _write_backbone_graph_structure(
        tmp_path / "orientation-rotated.pdb",
        _transform_residues(residues, rotation, np.zeros(3)),
    )

    original = _encode_orientation_graph(encoder_type, original_input)
    transformed_artifacts = [
        _encode_orientation_graph(encoder_type, translated_input),
        _encode_orientation_graph(encoder_type, rotated_input),
    ]
    expected_keys = [
        "dist",
        *[f"rbf.{index}" for index in range(16)],
        "seq_sep",
        "flag.is_local_le_4",
        "edge.same_chain",
        "edge.inter_chain",
        "edge.relative_direction_x",
        "edge.relative_direction_y",
        "edge.relative_direction_z",
    ]
    assert original.meta["feature_keys"]["edge_attr_keys"] == expected_keys
    original_rows = _edge_rows(original)
    for transformed in transformed_artifacts:
        transformed_rows = _edge_rows(transformed)
        assert original_rows.keys() == transformed_rows.keys()
        for pair, (distance, edge_attr) in original_rows.items():
            transformed_distance, transformed_edge_attr = transformed_rows[pair]
            assert transformed_distance == pytest.approx(distance, abs=1e-6)
            np.testing.assert_allclose(transformed_edge_attr, edge_attr, atol=1e-6)

    direction_keys = expected_keys[-3:]
    forward = _edge_features(original, "A:1::ALA", "A:2::GLY", direction_keys)
    reverse = _edge_features(original, "A:2::GLY", "A:1::ALA", direction_keys)
    root_half = np.float32(1.0 / np.sqrt(2.0))
    np.testing.assert_allclose(forward, [root_half, root_half, 0.0], atol=1e-6)
    np.testing.assert_allclose(reverse, [-root_half, 0.0, -root_half], atol=1e-6)
    assert not np.allclose(reverse, -forward)

    assert _edge_feature(original, "A:1::ALA", "A:2::GLY", "edge.same_chain") == 1.0
    assert _edge_feature(original, "A:1::ALA", "A:2::GLY", "edge.inter_chain") == 0.0
    assert _edge_feature(original, "A:1::ALA", "B:1::SER", "seq_sep") == -1.0
    assert _edge_feature(original, "A:1::ALA", "B:1::SER", "flag.is_local_le_4") == 0.0
    assert _edge_feature(original, "A:1::ALA", "B:1::SER", "edge.same_chain") == 0.0
    assert _edge_feature(original, "A:1::ALA", "B:1::SER", "edge.inter_chain") == 1.0
    assert np.isfinite(_edge_features(original, "A:1::ALA", "B:1::SER", direction_keys)).all()

    parameters = next(
        token["relative_direction"]
        for token in original.meta["parameters"]["edge_attr_mode"]
        if isinstance(token, dict) and "relative_direction" in token
    )
    assert parameters == {
        "direction_convention": "source_frame_transpose_unit_ca_displacement",
        "frame_atoms": ["N", "CA", "C"],
        "frame_definition": "x=C-CA;y=orthogonalized(N-CA);z=x_cross_y",
        "frame_epsilon": 1e-6,
        "frame_handedness": "right",
        "frame_matrix_layout": "columns_xyz_global",
        "frame_origin": "CA",
        "missing_frame_policy": "nan",
    }


def test_multichain_edge_feature_composition_is_deterministic_and_round_trips(
    tmp_path: Path,
) -> None:
    residues = _orientation_residues()
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    translation = np.asarray([9.0, -6.0, 2.0], dtype=np.float64)

    def encode(path: Path):
        return ResidueKNNGraph(
            k=2,
            cutoff=None,
            symmetric=True,
            node_attr_mode=["interface"],
            edge_attr_mode=[
                {"interface_contact": {"cutoff": 5.0}},
                "relative_direction",
                "inter_chain",
                "same_chain",
                "seq",
                "rbf",
                "dist",
            ],
            use_cache=False,
        ).encode(StructureInput(source=str(path)))

    original = encode(_write_backbone_graph_structure(tmp_path / "edge-original.pdb", residues).source)
    transformed = encode(
        _write_backbone_graph_structure(
            tmp_path / "edge-transformed.pdb",
            _transform_residues(residues, rotation, translation),
        ).source
    )
    expected_keys = [
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

    assert original.meta["feature_keys"]["edge_attr_keys"] == expected_keys
    assert original.data["edge_attr"].shape == (original.data["edge_index"].shape[1], 24)
    assert original.data["edge_attr"].dtype == np.float32
    assert original.data["chain_id"].tolist() == ["A", "A", "B"]
    assert original.data["interface_residue_ids"].tolist() == [
        "A:1::ALA",
        "A:2::GLY",
        "B:1::SER",
    ]
    np.testing.assert_array_equal(original.data["edge_index"], transformed.data["edge_index"])
    np.testing.assert_array_equal(
        original.data["interface_residue_ids"],
        transformed.data["interface_residue_ids"],
    )
    np.testing.assert_array_equal(original.data["node_attr"], transformed.data["node_attr"])
    np.testing.assert_allclose(original.data["edge_dist"], transformed.data["edge_dist"], atol=1e-6)
    np.testing.assert_allclose(
        original.data["edge_attr"],
        transformed.data["edge_attr"],
        atol=1e-6,
    )

    key_index = {key: index for index, key in enumerate(expected_keys)}
    same_chain = original.data["edge_attr"][:, key_index["edge.same_chain"]]
    inter_chain = original.data["edge_attr"][:, key_index["edge.inter_chain"]]
    interface_contact = original.data["edge_attr"][:, key_index["edge.interface_contact"]]
    np.testing.assert_array_equal(same_chain + inter_chain, np.ones_like(same_chain))
    assert np.all(interface_contact <= inter_chain)
    assert np.any(interface_contact == 1.0)

    target = tmp_path / "edge-composition"
    original.save(target)
    loaded = StructureArtifact.load(target, strict=True)
    assert loaded.meta["feature_keys"]["edge_attr_keys"] == expected_keys
    for name, values in original.data.items():
        np.testing.assert_equal(loaded.data[name], values)


def test_relative_direction_missing_frame_nan_round_trip_and_exports(tmp_path: Path) -> None:
    residues = _orientation_residues()[:2]
    del residues[0][3]["N"]
    structure_input = _write_backbone_graph_structure(tmp_path / "missing-frame.pdb", residues)
    artifact = ResidueKNNGraph(
        k=1,
        cutoff=None,
        symmetric=True,
        include_self_loops=True,
        edge_attr_mode=["same_chain", "inter_chain", "relative_direction"],
        use_cache=False,
    ).encode(structure_input)
    direction_keys = [
        "edge.relative_direction_x",
        "edge.relative_direction_y",
        "edge.relative_direction_z",
    ]

    assert np.isnan(_edge_features(artifact, "A:1::ALA", "A:2::GLY", direction_keys)).all()
    assert np.isfinite(_edge_features(artifact, "A:2::GLY", "A:1::ALA", direction_keys)).all()
    assert np.isnan(_edge_features(artifact, "A:2::GLY", "A:2::GLY", direction_keys)).all()
    relations = artifact.data["edge_attr"][:, :2]
    np.testing.assert_array_equal(relations[:, 0] + relations[:, 1], 1.0)

    artifact_dir = tmp_path / "orientation-artifact"
    artifact.save(artifact_dir)
    loaded = StructureArtifact.load(artifact_dir, strict=True)
    np.testing.assert_equal(loaded.data["edge_attr"], artifact.data["edge_attr"])
    assert loaded.meta["feature_keys"]["edge_attr_keys"] == [
        "edge.same_chain",
        "edge.inter_chain",
        *direction_keys,
    ]

    torch = pytest.importorskip("torch")
    torch_graph = eris.to_torch(loaded)
    assert isinstance(torch_graph["edge_attr"], torch.Tensor)
    np.testing.assert_equal(torch_graph["edge_attr"].cpu().numpy(), artifact.data["edge_attr"])
    if find_spec("torch_geometric") is not None:
        pyg = eris.to_pyg(loaded)
        np.testing.assert_equal(pyg.edge_attr.cpu().numpy(), artifact.data["edge_attr"])


@pytest.mark.parametrize("encoder_type", GRAPH_ENCODERS)
def test_sequence_edge_features_remain_aligned_and_symmetric(
    encoder_type: type,
    tmp_path: Path,
) -> None:
    structure_input = _write_structure(
        tmp_path / "symmetric.pdb",
        [(17, " ", 0.0, "ALA", "CA", "A"), (20, " ", 1.0, "GLY", "CA", "A")],
    )
    artifact = _encode(encoder_type, structure_input)
    edge_index = artifact.data["edge_index"]
    pairs = [tuple(pair) for pair in edge_index.T.tolist()]
    edge_keys = artifact.meta["feature_keys"]["edge_attr_keys"]
    sequence_column = edge_keys.index("seq_sep")
    local_column = edge_keys.index("flag.is_local_le_1")

    assert artifact.data["edge_attr"].shape[0] == edge_index.shape[1]
    for row, (source, target) in enumerate(pairs):
        reverse_row = pairs.index((target, source))
        assert artifact.data["edge_attr"][row, sequence_column] == 3.0
        assert artifact.data["edge_attr"][row, local_column] == 0.0
        np.testing.assert_array_equal(
            artifact.data["edge_attr"][row], artifact.data["edge_attr"][reverse_row]
        )


@pytest.mark.parametrize("encoder_type", GRAPH_ENCODERS)
def test_corrected_graph_cache_version_cannot_reuse_previous_version(
    encoder_type: type,
    tmp_path: Path,
) -> None:
    structure_input = _write_structure(
        tmp_path / "cache.pdb",
        [(17, " ", 0.0, "ALA", "CA", "A"), (20, " ", 1.0, "GLY", "CA", "A")],
    )
    cache = DiskCache(tmp_path / "cache")
    current = _encode(encoder_type, structure_input)
    stale_meta = deepcopy(current.meta)
    stale_meta["encoder"]["version"] = "0.1.3"
    stale = StructureArtifact(data=dict(current.data), meta=stale_meta)
    stale.validate(strict=True)

    kwargs: dict[str, object] = {"use_cache": True, "cache": cache, "local_cutoff": 1}
    if encoder_type is ResidueKNNGraph:
        kwargs.update(k=32, cutoff=None, symmetric=True)
    else:
        kwargs.update(cutoff=100.0)
    encoder = encoder_type(**kwargs)
    parameters = encoder.config.to_dict()
    common = {
        "representation_name": encoder.name,
        "encoder_implementation": encoder.encoder_implementation,
        "config_fingerprint": compute_config_fingerprint(
            encoder.name, parameters, chain_ids=structure_input.chain_id
        ),
        "source_sha256": sha256_file(Path(structure_input.source)),
        "parser_hint": ".pdb",
        "model_index": structure_input.model_index,
        "chain_ids": structure_input.chain_id,
    }
    old_key = compute_cache_key(encoder_version="0.1.3", **common)
    current_key = compute_cache_key(encoder_version=encoder.version, **common)
    cache.set(old_key, stale)

    assert encoder.version == "0.1.4"
    assert old_key != current_key
    assert cache.get(current_key) is None
    result = encoder.encode(structure_input)
    assert result.meta["encoder"]["version"] == "0.1.4"
    assert cache.get(current_key) is not None
