from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eris.core.artifact_schema import compute_config_fingerprint, validate_schema
from eris.core.hashing import sha256_file
from eris.core.io import load_and_extract_residue
from eris.core.types import StructureInput
from eris.encoders.clouds.atom_point_cloud import AtomPointCloud
from eris.encoders.clouds.residue_point_cloud import ResiduePointCloudCA
from eris.encoders.geometry.contact_map_bins_ca import ContactMapBinsCA
from eris.encoders.geometry.contact_map_ca import ContactMapCA
from eris.encoders.geometry.distogram_bins_ca import DistogramBinsCA
from eris.encoders.geometry.distogram_ca import DistogramCA
from eris.encoders.graphs.atom_knn_graph import AtomKNNGraph
from eris.encoders.graphs.residue_contact_graph import ResidueContactGraph
from eris.encoders.graphs.residue_knn_graph import ResidueKNNGraph
from eris.encoders.grids.voxel_grid_ca import VoxelGridCA
from eris.features.node.constants import AA_VOCAB
from tests.helpers import MMCIF, MULTIMER_PDB, PDB, atom_line, write_pdb

ENCODER_TYPES = (
    AtomPointCloud,
    ResiduePointCloudCA,
    AtomKNNGraph,
    ResidueKNNGraph,
    ResidueContactGraph,
    DistogramCA,
    DistogramBinsCA,
    ContactMapCA,
    ContactMapBinsCA,
    VoxelGridCA,
)


def _write_linear_ca_structure(path: Path) -> StructureInput:
    residues = (("ALA", 0.0), ("GLY", 3.0), ("SER", 10.0))
    lines = [atom_line(i, "CA", name, i, x) for i, (name, x) in enumerate(residues, start=1)]
    return StructureInput(source=str(write_pdb(path, lines)), chain_id="A")


def _expected_counts(data: dict[str, np.ndarray], representation_type: str):
    if representation_type == "cloud":
        counts = {"n_nodes": int(data["pos"].shape[0])}
    elif representation_type == "graph":
        edge_index = data["edge_index"]
        counts = {
            "n_nodes": int(data["pos"].shape[0]),
            "n_edges": int(edge_index.shape[1]),
            "n_self_loops": int(np.count_nonzero(edge_index[0] == edge_index[1])),
        }
    elif representation_type == "matrix":
        primary = next(name for name in ("dist", "dist_bins", "contact", "contact_bins") if name in data)
        counts = {"n_nodes": int(data[primary].shape[0])}
    else:
        counts = {}
    if "omitted_residue_ids" in data:
        counts["n_omitted_residues"] = int(data["omitted_residue_ids"].shape[0])
    return counts


@pytest.mark.parametrize(("source", "expected"), [(PDB, "pdb"), (MMCIF, "mmcif")])
def test_loader_threads_gemmi_confirmed_input_format(source: Path, expected: str):
    loaded = load_and_extract_residue(StructureInput(source=str(source)))

    assert loaded.input_format == expected

    artifact = ResiduePointCloudCA(use_cache=False).encode(StructureInput(source=str(source)))
    artifact.validate()
    assert artifact.meta["input"]["format"] == expected


def test_distance_and_contact_maps_preserve_metric_and_cutoff(tmp_path: Path) -> None:
    inp = _write_linear_ca_structure(tmp_path / "linear.pdb")
    distogram = DistogramCA(include_self=True, use_cache=False).encode(inp)
    contact_map = ContactMapCA(cutoff=4.0, include_self=False, use_cache=False).encode(inp)

    np.testing.assert_array_equal(
        distogram.data["dist"],
        np.asarray([[0.0, 3.0, 10.0], [3.0, 0.0, 7.0], [10.0, 7.0, 0.0]], np.float32),
    )
    np.testing.assert_array_equal(
        contact_map.data["contact"],
        np.asarray([[0, 1, 0], [1, 0, 0], [0, 0, 0]], np.uint8),
    )


def test_distance_contact_and_distogram_matrices_are_rigid_transform_invariant(
    tmp_path: Path,
) -> None:
    coordinates = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [5.0, 0.0, 1.0],
            [2.0, 4.0, 2.0],
        ],
        dtype=np.float64,
    )
    rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    translation = np.asarray([13.0, -8.0, 4.0], dtype=np.float64)

    def write_structure(path: Path, values: np.ndarray) -> StructureInput:
        names = ("ALA", "GLY", "SER", "THR")
        lines = [
            atom_line(i, "CA", name, i, tuple(xyz))
            for i, (name, xyz) in enumerate(zip(names, values.tolist(), strict=True), 1)
        ]
        return StructureInput(source=str(write_pdb(path, lines)), chain_id="A")

    original_input = write_structure(tmp_path / "matrix-original.pdb", coordinates)
    transformed_input = write_structure(
        tmp_path / "matrix-transformed.pdb",
        coordinates @ rotation.T + translation,
    )
    encoders = (
        DistogramCA(include_self=True, use_cache=False),
        DistogramBinsCA(include_self=True, use_cache=False),
        ContactMapCA(cutoff=4.0, include_self=False, use_cache=False),
        ContactMapBinsCA(include_self=False, use_cache=False),
    )

    for encoder in encoders:
        original = encoder.encode(original_input)
        transformed = encoder.encode(transformed_input)
        assert original.data.keys() == transformed.data.keys()
        for name, values in original.data.items():
            if name in {"pos", "centroid"}:
                continue
            if np.issubdtype(values.dtype, np.floating):
                np.testing.assert_allclose(values, transformed.data[name], atol=1e-6)
            else:
                np.testing.assert_array_equal(values, transformed.data[name])


@pytest.mark.parametrize(
    ("source", "expected_chains"),
    [(PDB, ["A"]), (MULTIMER_PDB, ["A", "B"])],
    ids=["experimental-monomer", "experimental-multimer"],
)
def test_residue_cloud_features_on_monomer_and_multimer(source: Path, expected_chains: list[str]) -> None:
    artifact = ResiduePointCloudCA(
        node_attr_mode=["onehot", "bfactor_ca"],
        use_cache=False,
    ).encode(StructureInput(source=str(source)))

    assert artifact.meta["input"]["chain_ids"] == expected_chains
    assert list(dict.fromkeys(artifact.data["chain_id"].tolist())) == expected_chains
    assert artifact.data["node_attr"].dtype == np.float32
    assert artifact.data["node_attr"].shape == (
        artifact.data["pos"].shape[0],
        len(AA_VOCAB) + 1,
    )
    assert artifact.meta["feature_keys"]["node_attr_keys"] == [
        *[f"aa_onehot.{name}" for name in AA_VOCAB],
        "bfactor_ca",
    ]
    np.testing.assert_array_equal(
        artifact.data["node_attr"][:, -1],
        artifact.data["bfactor_ca"],
    )


@pytest.mark.parametrize("encoder_type", ENCODER_TYPES, ids=lambda value: value.name)
def test_all_explicit_encoders_return_complete_schema(encoder_type):
    encoder = encoder_type(use_cache=False)
    artifact = encoder.encode(StructureInput(source=str(PDB)))
    metadata = artifact.meta

    assert validate_schema(metadata, artifact.data, strict=True) == metadata
    assert metadata["representation_name"] == encoder.name
    assert metadata["representation_family"] == "explicit"
    assert metadata["representation_type"] == encoder.representation.representation_type
    assert metadata["granularity"] == encoder.config.granularity
    assert metadata["coordinate_source"] == encoder.config.coordinate_source
    assert metadata["encoder"] == {
        "implementation": encoder.encoder_implementation,
        "version": encoder.version,
    }
    assert metadata["parameters"] == encoder.config.to_dict()
    assert metadata["input"] == {
        "source": str(PDB),
        "source_sha256": sha256_file(PDB),
        "format": "pdb",
        "model_index": 0,
        "chain_ids": ["A"],
    }
    assert metadata["provenance"]["config_fingerprint"] == compute_config_fingerprint(
        encoder.name,
        encoder.config.to_dict(),
        chain_ids=["A"],
    )
    assert metadata["counts"] == _expected_counts(
        artifact.data,
        metadata["representation_type"],
    )


@pytest.mark.parametrize(
    ("num_bins", "add_no_contact_class", "fits"),
    [(256, False, True), (256, True, False)],
)
def test_binned_class_ids_must_fit_the_output_dtype(num_bins, add_no_contact_class, fits):
    representation = "residue_distogram_bins"
    import eris

    params = {
        "num_bins": num_bins,
        "dmin": 0.0,
        "dmax": 20.0,
        "dtype": "uint8",
        "add_no_contact_class": add_no_contact_class,
    }
    if fits:
        artifact = eris.encode(PDB, representation=representation, use_cache=False, **params)
        assert artifact.data["dist_bins"].dtype == np.uint8
    else:
        with pytest.raises(ValueError, match="cannot represent"):
            eris.encode(PDB, representation=representation, use_cache=False, **params)
