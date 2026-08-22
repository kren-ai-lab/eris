from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eris.core.types import StructureInput
from eris.encoders.clouds.atom_point_cloud import AtomPointCloud
from eris.encoders.graphs.atom_knn_graph import AtomKNNGraph

_PDB_ATOM_SELECTION = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N
ATOM      2  CA AALA A   1       1.000   0.000   0.000  0.40 11.00           C
ATOM      3  CA BALA A   1       2.000   0.000   0.000  0.60 12.00           C
ATOM      4  C   ALA A   1       3.000   0.000   0.000  1.00 13.00           C
ATOM      5  H   ALA A   1       4.000   0.000   0.000  1.00 14.00           H
HETATM    6  C1  LIG A   2       5.000   0.000   0.000  1.00 15.00           C
HETATM    7  H1  LIG A   2       6.000   0.000   0.000  1.00 16.00           H
HETATM    8  O   HOH A   3       7.000   0.000   0.000  1.00 17.00           O
HETATM    9 MG    MG A   4       8.000   0.000   0.000  1.00 18.00          MG
TER
END
"""

FILTER_DEFAULTS = {
    "include_hetero": False,
    "include_h": False,
    "drop_water": True,
    "keep_ions": True,
    "altloc": "first",
    "amino_acids_only": False,
}


@pytest.fixture
def atom_selection_pdb(tmp_path: Path) -> Path:
    path = tmp_path / "atom_selection.pdb"
    path.write_text(_PDB_ATOM_SELECTION, encoding="utf-8")
    return path


def _point_cloud(**kwargs) -> AtomPointCloud:
    return AtomPointCloud(use_cache=False, center=False, **kwargs)


def _atom_graph(**kwargs) -> AtomKNNGraph:
    return AtomKNNGraph(
        use_cache=False,
        center=False,
        k=2,
        cutoff=None,
        **kwargs,
    )


def _encode(encoder, path: Path):
    return encoder.encode(StructureInput(source=str(path), chain_id="A"))


def _resnames(artifact) -> list[str]:
    return artifact.data["resname"].tolist()


def _atom_names(artifact) -> list[str]:
    return artifact.data["atom_name"].tolist()


def _identities(artifact) -> list[tuple[str, str, float]]:
    return [
        (resname, atom_name, float(position[0]))
        for resname, atom_name, position in zip(
            artifact.data["resname"],
            artifact.data["atom_name"],
            artifact.data["pos"],
            strict=True,
        )
    ]


@pytest.mark.parametrize("encoder_factory", [_point_cloud, _atom_graph])
def test_atom_encoder_defaults_match_existing_extractor_defaults(encoder_factory):
    encoder = encoder_factory()

    assert {name: getattr(encoder, name) for name in FILTER_DEFAULTS} == FILTER_DEFAULTS


def test_default_selection_excludes_hetero_water_and_hydrogen(atom_selection_pdb):
    artifact = _encode(_point_cloud(), atom_selection_pdb)

    assert _resnames(artifact) == ["ALA", "ALA", "ALA", "ALA"]
    assert _atom_names(artifact) == ["N", "CA", "CA", "C"]
    np.testing.assert_array_equal(artifact.data["pos"][:, 0], [0.0, 1.0, 2.0, 3.0])


def test_include_hetero_controls_ligand_and_ion(atom_selection_pdb):
    excluded = _encode(_point_cloud(include_hetero=False), atom_selection_pdb)
    included = _encode(_point_cloud(include_hetero=True), atom_selection_pdb)

    assert len(excluded.data["pos"]) == 4
    assert len(included.data["pos"]) == 6
    assert set(_resnames(included)) == {"ALA", "LIG", "MG"}
    assert "HOH" not in _resnames(included)


def test_include_h_controls_hydrogen_by_element(atom_selection_pdb):
    excluded = _encode(_point_cloud(include_h=False), atom_selection_pdb)
    included = _encode(_point_cloud(include_h=True), atom_selection_pdb)

    assert len(excluded.data["pos"]) == 4
    assert len(included.data["pos"]) == 5
    assert "H" not in _atom_names(excluded)
    assert "H" in _atom_names(included)


def test_drop_water_requires_hetero_inclusion_to_observe_water(atom_selection_pdb):
    dropped = _encode(
        _point_cloud(include_hetero=True, drop_water=True),
        atom_selection_pdb,
    )
    retained = _encode(
        _point_cloud(include_hetero=True, drop_water=False),
        atom_selection_pdb,
    )

    assert len(dropped.data["pos"]) == 6
    assert len(retained.data["pos"]) == 7
    assert "HOH" not in _resnames(dropped)
    assert "HOH" in _resnames(retained)


def test_keep_ions_requires_hetero_inclusion_to_observe_ion(atom_selection_pdb):
    hetero_excluded = _encode(
        _point_cloud(include_hetero=False, keep_ions=True),
        atom_selection_pdb,
    )
    retained = _encode(
        _point_cloud(include_hetero=True, keep_ions=True),
        atom_selection_pdb,
    )
    dropped = _encode(
        _point_cloud(include_hetero=True, keep_ions=False),
        atom_selection_pdb,
    )

    assert len(hetero_excluded.data["pos"]) == 4
    assert "MG" not in _resnames(hetero_excluded)
    assert len(retained.data["pos"]) == 6
    assert len(dropped.data["pos"]) == 5
    assert "MG" in _resnames(retained)
    assert "MG" not in _resnames(dropped)


def test_amino_acids_only_runs_before_other_entity_filters(atom_selection_pdb):
    all_entities = _encode(
        _point_cloud(
            include_hetero=True,
            include_h=True,
            drop_water=False,
            amino_acids_only=False,
        ),
        atom_selection_pdb,
    )
    amino_acids = _encode(
        _point_cloud(
            include_hetero=True,
            include_h=True,
            drop_water=False,
            amino_acids_only=True,
        ),
        atom_selection_pdb,
    )

    assert len(all_entities.data["pos"]) == 9
    assert len(amino_acids.data["pos"]) == 5
    assert set(_resnames(amino_acids)) == {"ALA"}


def test_altloc_policy_is_propagated_without_changing_current_semantics(atom_selection_pdb):
    first = _encode(_point_cloud(altloc="first"), atom_selection_pdb)
    max_occupancy = _encode(_point_cloud(altloc="max_occupancy"), atom_selection_pdb)

    first_ca_positions = first.data["pos"][
        np.asarray(_atom_names(first)) == "CA",
        0,
    ]
    max_ca_positions = max_occupancy.data["pos"][
        np.asarray(_atom_names(max_occupancy)) == "CA",
        0,
    ]

    np.testing.assert_array_equal(first_ca_positions, [1.0, 2.0])
    np.testing.assert_array_equal(max_ca_positions, [2.0])
    assert len(first.data["pos"]) == 4
    assert len(max_occupancy.data["pos"]) == 3


def test_atom_point_cloud_and_graph_share_selection(atom_selection_pdb):
    filters = {
        "include_hetero": True,
        "include_h": True,
        "drop_water": False,
        "keep_ions": True,
        "altloc": "max_occupancy",
        "amino_acids_only": False,
    }

    cloud = _encode(_point_cloud(**filters), atom_selection_pdb)
    graph = _encode(_atom_graph(**filters), atom_selection_pdb)

    assert _identities(cloud) == _identities(graph)


def test_selection_filter_changes_fingerprint():
    excluded = _point_cloud(include_hetero=False)
    included = _point_cloud(include_hetero=True)

    assert excluded.fingerprint() != included.fingerprint()


@pytest.mark.parametrize("encoder_factory", [_point_cloud, _atom_graph])
def test_atom_selection_filters_are_recorded_in_artifact_params(
    encoder_factory,
    atom_selection_pdb,
):
    filters = {
        "include_hetero": True,
        "include_h": True,
        "drop_water": False,
        "keep_ions": False,
        "altloc": "max_occupancy",
        "amino_acids_only": False,
    }
    artifact = _encode(encoder_factory(**filters), atom_selection_pdb)

    assert {name: artifact.meta["parameters"][name] for name in filters} == filters
    assert all(name not in artifact.meta for name in filters)
