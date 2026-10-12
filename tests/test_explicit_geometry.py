from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from eris.core.artifact_schema import compute_config_fingerprint, validate_schema
from eris.core.hashing import sha256_file
from eris.core.io import load_and_extract_residue
from eris.core.representations.output_declarations import (
    ATOM_GRAPH_OUTPUTS,
    ATOM_POINT_CLOUD_OUTPUTS,
    RESIDUE_CONTACT_MAP_BINS_OUTPUTS,
    RESIDUE_CONTACT_MAP_OUTPUTS,
    RESIDUE_DISTOGRAM_BINS_OUTPUTS,
    RESIDUE_DISTOGRAM_OUTPUTS,
    RESIDUE_GRAPH_OUTPUTS,
    RESIDUE_POINT_CLOUD_OUTPUTS,
    VOXEL_OUTPUTS,
)
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

PDB = Path("tests/data/1AQ2.pdb")
MMCIF = Path("tests/data/1AQ2.cif")
MULTIMER_PDB = Path("tests/data/3EC1.pdb")

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

EXPECTED_DECLARATIONS = {
    "atom_point_cloud": ATOM_POINT_CLOUD_OUTPUTS,
    "residue_point_cloud": RESIDUE_POINT_CLOUD_OUTPUTS,
    "atom_knn_graph": ATOM_GRAPH_OUTPUTS,
    "residue_knn_graph": RESIDUE_GRAPH_OUTPUTS,
    "residue_contact_graph": RESIDUE_GRAPH_OUTPUTS,
    "residue_distogram": RESIDUE_DISTOGRAM_OUTPUTS,
    "residue_distogram_bins": RESIDUE_DISTOGRAM_BINS_OUTPUTS,
    "residue_contact_map": RESIDUE_CONTACT_MAP_OUTPUTS,
    "residue_contact_map_bins": RESIDUE_CONTACT_MAP_BINS_OUTPUTS,
    "residue_voxel_grid": VOXEL_OUTPUTS,
}

ATOM_AUX_BASELINE = (
    ("chain_id", (4114,), "<U1"),
    ("elem_index", (4114,), "int16"),
    ("atom_role", (4114,), "int8"),
    ("bfactor", (4114,), "float32"),
    ("occ", (4114,), "float32"),
    ("residue_ids", (4114,), "<U10"),
    ("resname", (4114,), "<U3"),
    ("atom_name", (4114,), "<U3"),
    ("seq_pos_residue", (4114,), "int32"),
    ("aa_index_residue", (4114,), "int16"),
)
RESIDUE_AUX_BASELINE = (
    ("residue_ids", (534,), "<U10"),
    ("chain_id", (534,), "<U1"),
    ("omitted_residue_ids", (0,), "<U1"),
    ("resname", (534,), "<U3"),
    ("aa_index", (534,), "int16"),
    ("plddt", (534,), "float32"),
    ("bfactor_ca", (534,), "float32"),
)
RESIDUE_PREFIX_BASELINE = (
    ("pos", (534, 3), "float32"),
    ("centroid", (1, 3), "float32"),
)

ARRAY_BASELINES = {
    "atom_point_cloud": (
        ("pos", (4114, 3), "float32"),
        ("centroid", (1, 3), "float32"),
        ("node_attr", (4114, 25), "float32"),
        *ATOM_AUX_BASELINE,
    ),
    "residue_point_cloud": (
        *RESIDUE_PREFIX_BASELINE,
        ("node_attr", (534, 37), "float32"),
        *RESIDUE_AUX_BASELINE,
    ),
    "atom_knn_graph": (
        ("pos", (4114, 3), "float32"),
        ("centroid", (1, 3), "float32"),
        ("node_attr", (4114, 25), "float32"),
        ("edge_index", (2, 75900), "int64"),
        ("edge_dist", (75900,), "float32"),
        ("edge_attr", (75900, 19), "float32"),
        ("graph_attr", (13,), "float32"),
        *ATOM_AUX_BASELINE,
    ),
    "residue_knn_graph": (
        *RESIDUE_PREFIX_BASELINE,
        ("node_attr", (534, 37), "float32"),
        ("edge_index", (2, 9646), "int64"),
        ("edge_dist", (9646,), "float32"),
        ("edge_attr", (9646, 19), "float32"),
        ("graph_attr", (13,), "float32"),
        *RESIDUE_AUX_BASELINE,
    ),
    "residue_contact_graph": (
        *RESIDUE_PREFIX_BASELINE,
        ("node_attr", (534, 37), "float32"),
        ("edge_index", (2, 5396), "int64"),
        ("edge_dist", (5396,), "float32"),
        ("edge_attr", (5396, 19), "float32"),
        ("graph_attr", (13,), "float32"),
        *RESIDUE_AUX_BASELINE,
    ),
    "residue_distogram": (
        ("dist", (534, 534), "float32"),
        *RESIDUE_PREFIX_BASELINE,
        *RESIDUE_AUX_BASELINE,
    ),
    "residue_distogram_bins": (
        ("dist_bins", (534, 534), "uint8"),
        ("bin_edges", (37,), "float32"),
        *RESIDUE_PREFIX_BASELINE,
        *RESIDUE_AUX_BASELINE,
    ),
    "residue_contact_map": (
        ("contact", (534, 534), "uint8"),
        *RESIDUE_PREFIX_BASELINE,
        *RESIDUE_AUX_BASELINE,
    ),
    "residue_contact_map_bins": (
        ("contact_bins", (534, 534), "uint8"),
        ("bin_edges", (37,), "float32"),
        *RESIDUE_PREFIX_BASELINE,
        *RESIDUE_AUX_BASELINE,
    ),
    "residue_voxel_grid": (
        ("grid", (24, 64, 64, 64), "float32"),
        ("grid_size", (1,), "int32"),
        ("voxel_size", (1,), "float32"),
        ("origin", (3,), "float32"),
        ("centroid", (1, 3), "float32"),
        *RESIDUE_AUX_BASELINE,
    ),
}

CLOUD_DIGESTS = {
    "atom_point_cloud": {
        "pos": "4fbaf06af17251ada366958fc3bc8218bb2d68785b6ccd5b0eb76cb392721c2b",
        # Protects raw atomic B-factors: no implicit division or clipping.
        "node_attr": "80820a0ff12567827f0afe82392a0327df67ef33a186ce5f09a3bc45b6b6854b",
    },
    "residue_point_cloud": {
        "pos": "54a3d3c574f404a5cbb6d034def52abf5b64d1096121df673501ed715f3474d3",
        "node_attr": "e4f890bb892fefd20b759387b2e4fa6cefcf6906178bb5f76f7887fa68035964",
    },
}


def _write_linear_ca_structure(path: Path) -> StructureInput:
    lines = [
        f"ATOM  {serial:5d}  CA  {resname:>3s} A{serial:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{10.0:6.2f}           C  "
        for serial, (resname, x) in enumerate(
            (("ALA", 0.0), ("GLY", 3.0), ("SER", 10.0)),
            start=1,
        )
    ]
    path.write_text("\n".join([*lines, "TER", "END", ""]), encoding="ascii")
    return StructureInput(source=str(path), chain_id="A")


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


def _compact_array_contract(data: dict[str, np.ndarray]):
    return tuple((name, tuple(array.shape), str(array.dtype)) for name, array in data.items())


def _compact_declarations(declarations):
    return tuple(
        (
            declaration["name"],
            declaration["role"],
            declaration.get("granularity"),
            declaration.get("layout"),
        )
        for declaration in declarations
    )


def test_shared_output_declarations_have_frozen_semantics():
    atom_cloud = _compact_declarations(ATOM_POINT_CLOUD_OUTPUTS)
    residue_cloud = _compact_declarations(RESIDUE_POINT_CLOUD_OUTPUTS)
    atom_graph = _compact_declarations(ATOM_GRAPH_OUTPUTS)
    residue_graph = _compact_declarations(RESIDUE_GRAPH_OUTPUTS)
    voxel = _compact_declarations(VOXEL_OUTPUTS)

    assert atom_cloud == (
        ("pos", "coordinates", "atom", "N3"),
        ("centroid", "centroid", "structure", None),
        ("node_attr", "node_features", "atom", "NF"),
        ("chain_id", "chain_id", "atom", "N"),
        ("elem_index", "element_index", "atom", "N"),
        ("atom_role", "atom_role", "atom", "N"),
        ("bfactor", "bfactor", "atom", "N"),
        ("occ", "occupancy", "atom", "N"),
        ("residue_ids", "residue_ids", "atom", "N"),
        ("resname", "residue_names", "atom", "N"),
        ("atom_name", "atom_names", "atom", "N"),
        ("seq_pos_residue", "residue_sequence_index", "atom", "N"),
        ("aa_index_residue", "amino_acid_index", "atom", "N"),
    )
    assert residue_cloud == (
        ("pos", "coordinates", "residue", "N3"),
        ("centroid", "centroid", "structure", None),
        ("node_attr", "node_features", "residue", "NF"),
        ("residue_ids", "node_ids", "residue", "N"),
        ("chain_id", "chain_id", "residue", "N"),
        ("omitted_residue_ids", "omitted_residue_ids", "residue", "N"),
        ("resname", "residue_names", "residue", "N"),
        ("aa_index", "amino_acid_index", "residue", "N"),
        ("plddt", "confidence", "residue", "N"),
        ("bfactor_ca", "bfactor", "residue", "N"),
    )
    graph_outputs = (
        ("edge_index", "edge_index", None, "2E"),
        ("edge_dist", "edge_distance", None, "E"),
        ("edge_attr", "edge_features", None, "EF"),
        ("graph_attr", "graph_features", "structure", "F"),
    )
    assert atom_graph == atom_cloud[:3] + graph_outputs + atom_cloud[3:]
    assert residue_graph == residue_cloud[:3] + graph_outputs + residue_cloud[3:]

    residue_aux = residue_cloud[:2] + residue_cloud[3:]
    assert _compact_declarations(RESIDUE_DISTOGRAM_OUTPUTS) == (
        ("dist", "matrix", "residue", "NN"),
        *residue_aux,
    )
    assert _compact_declarations(RESIDUE_DISTOGRAM_BINS_OUTPUTS) == (
        ("dist_bins", "matrix", "residue", "NN"),
        ("bin_edges", "bin_edges", None, None),
        *residue_aux,
    )
    assert _compact_declarations(RESIDUE_CONTACT_MAP_OUTPUTS) == (
        ("contact", "matrix", "residue", "NN"),
        *residue_aux,
    )
    assert _compact_declarations(RESIDUE_CONTACT_MAP_BINS_OUTPUTS) == (
        ("contact_bins", "matrix", "residue", "NN"),
        ("bin_edges", "bin_edges", None, None),
        *residue_aux,
    )
    assert voxel == (
        ("grid", "grid", None, "CXYZ"),
        ("grid_size", "grid_size", None, None),
        ("voxel_size", "voxel_size", None, None),
        ("origin", "origin", None, None),
        ("centroid", "centroid", "structure", None),
        ("residue_ids", "residue_ids", "residue", "N"),
        ("chain_id", "chain_id", "residue", "N"),
        ("omitted_residue_ids", "omitted_residue_ids", "residue", "N"),
        ("resname", "residue_names", "residue", "N"),
        ("aa_index", "amino_acid_index", "residue", "N"),
        ("plddt", "confidence", "residue", "N"),
        ("bfactor_ca", "bfactor", "residue", "N"),
    )


@pytest.mark.parametrize(("source", "expected"), [(PDB, "pdb"), (MMCIF, "mmcif")])
def test_loader_threads_gemmi_confirmed_input_format(source: Path, expected: str):
    loaded = load_and_extract_residue(StructureInput(source=str(source)))

    assert loaded.input_format == expected

    artifact = ResiduePointCloudCA(use_cache=False).encode(StructureInput(source=str(source)))
    artifact.validate(strict=True)
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
        lines = []
        for serial, (resname, (x, y, z)) in enumerate(zip(names, values, strict=True), 1):
            lines.append(
                f"ATOM  {serial:5d}  CA  {resname:>3s} A{serial:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{20.0:6.2f}           C  "
            )
        path.write_text("\n".join([*lines, "TER", "END", ""]), encoding="ascii")
        return StructureInput(source=str(path), chain_id="A")

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
def test_experimental_structure_validation_panel(source: Path, expected_chains: list[str]) -> None:
    artifact = ResiduePointCloudCA(
        node_attr_mode=["onehot", "bfactor_ca"],
        use_cache=False,
    ).encode(StructureInput(source=str(source)))

    assert artifact.meta["schema_version"] == 1
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
    assert metadata["schema_version"] == 1
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
    assert [output["name"] for output in metadata["outputs"]] == [
        declaration["name"] for declaration in EXPECTED_DECLARATIONS[encoder.name]
    ]
    expected_feature_fields = {
        "cloud": {"node_attr_keys"},
        "graph": {"node_attr_keys", "edge_attr_keys", "graph_attr_keys"},
        "matrix": set(),
        "grid": {"grid_channels"},
    }
    assert set(metadata["feature_keys"]) == expected_feature_fields[metadata["representation_type"]]
    assert metadata["counts"] == _expected_counts(
        artifact.data,
        metadata["representation_type"],
    )
    assert _compact_array_contract(artifact.data) == ARRAY_BASELINES[encoder.name]
    for name, expected_digest in CLOUD_DIGESTS.get(encoder.name, {}).items():
        assert hashlib.sha256(artifact.data[name].tobytes()).hexdigest() == expected_digest
    assert {
        "encoder_name",
        "encoder_version",
        "encoder_fingerprint",
        "type",
        "level",
        "params",
        "matrix_key",
        "matrix_attr_keys",
        "grid_key",
        "grid_layout",
    }.isdisjoint(metadata)


@pytest.mark.parametrize("representation", ["residue_distogram_bins", "residue_contact_map_bins"])
@pytest.mark.parametrize(
    ("num_bins", "add_no_contact_class", "fits"),
    [(256, False, True), (255, True, True), (257, False, False), (256, True, False)],
)
def test_binned_class_ids_must_fit_the_output_dtype(representation, num_bins, add_no_contact_class, fits):
    import eris

    params = {
        "num_bins": num_bins,
        "dmin": 0.0,
        "dmax": 20.0,
        "dtype": "uint8",
        "add_no_contact_class": add_no_contact_class,
    }
    if fits:
        artifact = eris.encode(
            "tests/data/1AQ2.pdb", representation=representation, use_cache=False, **params
        )
        key = {"residue_distogram_bins": "dist_bins", "residue_contact_map_bins": "contact_bins"}
        assert artifact.data[key[representation]].dtype == np.uint8
    else:
        with pytest.raises(ValueError, match="cannot represent"):
            eris.encode("tests/data/1AQ2.pdb", representation=representation, use_cache=False, **params)
