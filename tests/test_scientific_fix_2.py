from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from eris.core.artifact_schema_v2 import compute_config_fingerprint
from eris.core.artifacts import StructureArtifact
from eris.core.cache import DiskCache, compute_cache_key_v2
from eris.core.hashing import sha256_file
from eris.core.types import StructureInput
from eris.encoders.graphs.residue_contact_graph import ResidueContactGraph
from eris.encoders.graphs.residue_knn_graph import ResidueKNNGraph


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
    stale_meta["encoder"]["version"] = "0.1.1"
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
        "config_fingerprint": compute_config_fingerprint(encoder.name, parameters),
        "source_sha256": sha256_file(Path(structure_input.source)),
        "parser_hint": ".pdb",
        "model_index": structure_input.model_index,
        "chain_id": structure_input.chain_id,
    }
    old_key = compute_cache_key_v2(encoder_version="0.1.1", **common)
    current_key = compute_cache_key_v2(encoder_version=encoder.version, **common)
    cache.set_v2(old_key, stale)

    assert encoder.version == "0.1.2"
    assert old_key != current_key
    assert cache.get_v2(current_key) is None
    result = encoder.encode(structure_input)
    assert result.meta["encoder"]["version"] == "0.1.2"
    assert cache.get_v2(current_key) is not None
