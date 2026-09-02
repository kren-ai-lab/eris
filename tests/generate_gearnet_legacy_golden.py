"""Explicitly regenerate the frozen Python-3.10 TorchDrug GearNet reference.

Run only in the documented legacy CPU/fp32 environment, for example after entering
the MSVC x64 developer environment on Windows:

    python tests/generate_gearnet_legacy_golden.py \
        --checkpoint C:\\path\\to\\mc_gearnet_edge.pth --overwrite

To regenerate only the supplemental SpatialEdge reference, use:

    python tests/generate_gearnet_legacy_golden.py --spatial-reference --overwrite

Regeneration changes the scientific parity baseline. It is intentionally never run
by the normal test suite, never downloads a checkpoint, and never overwrites an
existing fixture without ``--overwrite``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from eris.core.learned.backends.gearnet.backend import GearNetEdgeBackend
from eris.core.learned.backends.gearnet.config import GearNetEdgeConfig
from eris.core.learned.backends.gearnet.dependencies import probe_gearnet_availability
from eris.core.learned.backends.gearnet.metadata import (
    GEARNET_EDGE_ARCHITECTURE,
    GEARNET_EDGE_BACKEND_ID,
    GEARNET_EDGE_CHECKPOINT_IDENTIFIER,
    GEARNET_EDGE_CHECKPOINT_DESCRIPTOR,
    GEARNET_EDGE_ENCODER_VERSION,
    GEARNET_EDGE_EDGE_INPUT_DIM,
    GEARNET_EDGE_HIDDEN_DIMS,
    GEARNET_EDGE_IMPLEMENTATION,
    GEARNET_EDGE_NODE_INPUT_DIM,
    GEARNET_EDGE_NUM_ANGLE_BIN,
    GEARNET_EDGE_NUM_RELATION,
    GEARNET_EDGE_OUTPUT_DIM,
    GEARNET_EDGE_REPRESENTATION_NAME,
)
from eris.core.learned.backends.gearnet.preprocessing import (
    BiologicalIdentityTable,
    BiologicalResidueIdentity,
    TransportNodeRecord,
    construct_gearnet_input,
)
from eris.core.learned.checkpoints import verify_checkpoint
from eris.core.learned.execution import ResolvedExecution
from eris.core.types import StructureInput


DATA_DIR = Path(__file__).with_name("data")
INPUT_PATH = DATA_DIR / "gearnet_legacy_reference.pdb"
GRAPH_PATH = DATA_DIR / "gearnet_legacy_reference_graph.npz"
EMBEDDING_PATH = DATA_DIR / "gearnet_legacy_reference_embeddings.npz"
METADATA_PATH = DATA_DIR / "gearnet_legacy_reference.json"
OUTPUT_PATHS = (GRAPH_PATH, EMBEDDING_PATH, METADATA_PATH)
SPATIAL_INPUT_PATH = DATA_DIR / "gearnet_legacy_spatial_reference.pdb"
SPATIAL_GRAPH_PATH = DATA_DIR / "gearnet_legacy_spatial_reference_graph.npz"
SPATIAL_METADATA_PATH = DATA_DIR / "gearnet_legacy_spatial_reference.json"
SPATIAL_PROBE_PATH = DATA_DIR / "gearnet_legacy_spatial_max_neighbors_probe.npz"
SPATIAL_PROBE_METADATA_PATH = DATA_DIR / "gearnet_legacy_spatial_max_neighbors_probe.json"
SPATIAL_OUTPUT_PATHS = (
    SPATIAL_GRAPH_PATH,
    SPATIAL_METADATA_PATH,
    SPATIAL_PROBE_PATH,
    SPATIAL_PROBE_METADATA_PATH,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _as_numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def _require_legacy_environment() -> None:
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("Golden reference generation requires the frozen Python 3.10 runtime")
    if shutil.which("ninja") is None:
        raise RuntimeError(
            "Ninja executable is not on PATH; activate the legacy environment before generation"
        )
    if sys.platform == "win32" and shutil.which("cl") is None:
        raise RuntimeError(
            "MSVC x64 cl.exe is not on PATH; start an x64 Visual Studio developer environment"
        )
    availability = probe_gearnet_availability()
    if not availability.available:
        raise RuntimeError(f"Frozen GearNet dependency validation failed: {availability.reason}")


def _validate_targets(paths: tuple[Path, ...], *, overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise RuntimeError(
            f"Refusing to overwrite existing golden fixture(s): {names}; use --overwrite"
        )


def _tie_metadata(node_position: np.ndarray) -> dict[str, object]:
    pair_distances = [
        float(np.linalg.norm(node_position[left] - node_position[right]))
        for left in range(node_position.shape[0])
        for right in range(left + 1, node_position.shape[0])
    ]
    rounded = np.round(np.asarray(pair_distances, dtype=np.float64), decimals=5)
    return {
        "primary_fixture_contains_distance_ties": bool(np.unique(rounded).size < rounded.size),
        "tie_comparison_decimals": 5,
        "ordering_authority": "legacy TorchDrug graph and line-graph array order",
    }


def regenerate(*, checkpoint_path: Path, overwrite: bool) -> None:
    """Create graph, embedding, and metadata fixtures from the real legacy backend."""

    _require_legacy_environment()
    _validate_targets(OUTPUT_PATHS, overwrite=overwrite)
    checkpoint = verify_checkpoint(checkpoint_path, GEARNET_EDGE_CHECKPOINT_DESCRIPTOR)
    backend = GearNetEdgeBackend()
    execution = ResolvedExecution(device="cpu", precision="fp32")
    config = GearNetEdgeConfig(
        representation_name=GEARNET_EDGE_REPRESENTATION_NAME,
        output_mode="both",
    )
    preprocessed = backend.preprocess(
        StructureInput(source=str(INPUT_PATH), chain_id="A", model_index=0),
        config=config,
    )
    prepared = preprocessed.prepared
    model = backend.load(checkpoint, execution=execution)
    first = backend.infer(model, prepared, config=config, execution=execution)
    second = backend.infer(model, prepared, config=config, execution=execution)
    if not np.array_equal(first.node_embedding, second.node_embedding) or not np.array_equal(
        first.global_embedding, second.global_embedding
    ):
        raise RuntimeError("Legacy CPU/fp32 inference was not exactly deterministic")

    graph = prepared.graph
    line_graph = model.spatial_line_graph(graph)
    edge_list = _as_numpy(graph.edge_list)
    line_edge_list = _as_numpy(line_graph.edge_list)
    node_position = _as_numpy(graph.node_position)
    graph_arrays = {
        "node_ids": prepared.node_ids.copy(),
        "carrier": prepared.carrier.copy(),
        "node_feature": _as_numpy(graph.node_feature),
        "node_position": node_position,
        "edge_source": edge_list[:, 0],
        "edge_destination": edge_list[:, 1],
        "edge_relation": edge_list[:, 2],
        "edge_feature": _as_numpy(graph.edge_feature),
        "line_edge_source": line_edge_list[:, 0],
        "line_edge_destination": line_edge_list[:, 1],
        "line_angle_relation": line_edge_list[:, 2],
    }
    embedding_arrays = {
        "node_ids": first.node_ids.copy(),
        "node_embedding": first.node_embedding.copy(),
        "global_embedding": first.global_embedding.copy(),
    }
    np.savez(GRAPH_PATH, **graph_arrays)
    np.savez(EMBEDDING_PATH, **embedding_arrays)
    optional_versions = {
        name: _distribution_version(name)
        for name in (
            "torch",
            "torchdrug",
            "torch_scatter",
            "torch_cluster",
            "torch_sparse",
            "torch_geometric",
            "rdkit-pypi",
            "numpy",
            "ninja",
        )
    }
    metadata_payload = {
        "fixture_schema_version": 1,
        "input": {
            "filename": INPUT_PATH.name,
            "sha256": _sha256(INPUT_PATH),
            "chain_id": "A",
            "model_index": 0,
        },
        "checkpoint": {
            "filename": GEARNET_EDGE_CHECKPOINT_IDENTIFIER,
            "sha256": checkpoint.sha256,
        },
        "legacy_environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "optional_dependencies": optional_versions,
        },
        "backend": {
            "representation_name": GEARNET_EDGE_REPRESENTATION_NAME,
            "backend_id": GEARNET_EDGE_BACKEND_ID,
            "implementation": GEARNET_EDGE_IMPLEMENTATION,
            "encoder_version": GEARNET_EDGE_ENCODER_VERSION,
            "execution": {"device": "cpu", "precision": "fp32", "eval": True},
        },
        "model": {
            "architecture": GEARNET_EDGE_ARCHITECTURE,
            "input_dim": GEARNET_EDGE_NODE_INPUT_DIM,
            "hidden_dims": list(GEARNET_EDGE_HIDDEN_DIMS),
            "num_relation": GEARNET_EDGE_NUM_RELATION,
            "edge_input_dim": GEARNET_EDGE_EDGE_INPUT_DIM,
            "num_angle_bin": GEARNET_EDGE_NUM_ANGLE_BIN,
            "output_dim": GEARNET_EDGE_OUTPUT_DIM,
        },
        "arrays": {
            "graph": GRAPH_PATH.name,
            "embeddings": EMBEDDING_PATH.name,
        },
        "determinism": {
            "same_process_repeated_inference": "exact_array_equal",
        },
        "tie_behavior": _tie_metadata(node_position),
        "reference_generation_note": (
            "Generated from the frozen Python-3.10 TorchDrug GearNet-Edge CPU/fp32 "
            "backend; regenerating changes the scientific parity baseline."
        ),
    }
    METADATA_PATH.write_text(
        json.dumps(metadata_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _graph_arrays(prepared: Any, line_graph: Any) -> dict[str, np.ndarray]:
    graph = prepared.graph
    edge_list = _as_numpy(graph.edge_list)
    line_edge_list = _as_numpy(line_graph.edge_list)
    return {
        "node_ids": prepared.node_ids.copy(),
        "carrier": prepared.carrier.copy(),
        "node_feature": _as_numpy(graph.node_feature),
        "node_position": _as_numpy(graph.node_position),
        "edge_source": edge_list[:, 0],
        "edge_destination": edge_list[:, 1],
        "edge_relation": edge_list[:, 2],
        "edge_feature": _as_numpy(graph.edge_feature),
        "line_edge_source": line_edge_list[:, 0],
        "line_edge_destination": line_edge_list[:, 1],
        "line_angle_relation": line_edge_list[:, 2],
    }


def _line_graph(graph: Any) -> Any:
    from torchdrug.layers.geometry import SpatialLineGraph

    return SpatialLineGraph(num_angle_bin=GEARNET_EDGE_NUM_ANGLE_BIN)(graph)


def _assert_exact_arrays(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> None:
    if first.keys() != second.keys() or any(
        not np.array_equal(first[name], second[name]) for name in first
    ):
        raise RuntimeError("Legacy TorchDrug graph construction was not exactly deterministic")


def _relation_counts(edge_relation: np.ndarray) -> dict[str, int]:
    return {
        str(relation): int(np.count_nonzero(edge_relation == relation))
        for relation in range(GEARNET_EDGE_NUM_RELATION)
    }


def _spatial_observations(graph_arrays: dict[str, np.ndarray]) -> dict[str, object]:
    source = graph_arrays["edge_source"]
    destination = graph_arrays["edge_destination"]
    relation = graph_arrays["edge_relation"]
    spatial = relation == 5
    spatial_pairs = list(zip(source[spatial].tolist(), destination[spatial].tolist(), strict=True))
    positions = graph_arrays["node_position"]
    spatial_pair_set = set(spatial_pairs)
    knn_pair_set = set(
        zip(source[relation == 6].tolist(), destination[relation == 6].tolist(), strict=True)
    )
    target_zero_sources = [left for left, right in spatial_pairs if right == 0]
    tie_sources = [6, 7]
    tie_distances = [
        float(np.linalg.norm(positions[source_index] - positions[0]))
        for source_index in tie_sources
    ]
    if not all((source_index, 0) in spatial_pair_set for source_index in tie_sources):
        raise RuntimeError("Spatial reference did not retain the required equal-distance neighbors")
    if (9, 0) in spatial_pair_set or (0, 9) in spatial_pair_set:
        raise RuntimeError("Spatial reference did not establish the radius=10.0 exclusion boundary")
    if (4, 0) in spatial_pair_set or (0, 4) in spatial_pair_set:
        raise RuntimeError("Spatial reference did not establish min sequence-distance exclusion")
    if (5, 0) not in spatial_pair_set or (0, 5) not in spatial_pair_set:
        raise RuntimeError("Spatial reference did not establish min sequence-distance inclusion")
    if target_zero_sources != [5, 6, 7, 8]:
        raise RuntimeError("Unexpected legacy SpatialEdge traversal order for target node 0")
    overlap = sorted(spatial_pair_set.intersection(knn_pair_set))
    if not overlap:
        raise RuntimeError("Spatial reference did not retain cross-relation duplicate pairs")
    line_source = graph_arrays["line_edge_source"]
    line_destination = graph_arrays["line_edge_destination"]
    spatial_edge_indexes = np.flatnonzero(spatial)
    line_involving_spatial = np.isin(line_source, spatial_edge_indexes) | np.isin(
        line_destination, spatial_edge_indexes
    )
    if not bool(line_involving_spatial.any()):
        raise RuntimeError("Spatial reference did not produce line-graph spatial relationships")
    angle_relations = graph_arrays["line_angle_relation"][line_involving_spatial]
    if np.unique(angle_relations).size < 2:
        raise RuntimeError("Spatial reference did not produce nontrivial angle bins")
    first_spatial = int(np.flatnonzero(spatial)[0])
    return {
        "source_destination_convention": "edge_source sends to edge_destination",
        "radius_comparison": "strict_less_than",
        "radius_boundary_pair": {"source": 9, "destination": 0, "distance": 10.0},
        "min_sequence_distance": {
            "excluded_pair": {"source": 4, "destination": 0, "sequence_distance": 4},
            "included_pair": {"source": 5, "destination": 0, "sequence_distance": 5},
        },
        "target_zero_spatial_sources_in_legacy_order": target_zero_sources,
        "equal_distance_tie": {
            "destination": 0,
            "sources_in_legacy_order": tie_sources,
            "distance": tie_distances[0],
        },
        "cross_relation_duplicate_pairs": [list(pair) for pair in overlap],
        "spatial_relation_slice": {"start": first_spatial, "count": int(spatial.sum())},
        "line_graph_spatial_edge_count": int(line_involving_spatial.sum()),
        "line_graph_spatial_angle_bins": sorted(int(value) for value in np.unique(angle_relations)),
    }


def _max_neighbor_probe() -> dict[str, np.ndarray]:
    node_count = 40
    names = ("ALA",) * node_count
    coordinates = [
        (0.0, 0.0, 0.0),
        *((100.0 + index, 0.0, 0.0) for index in range(1, 5)),
        *((5.0, 0.0, 0.0) for _ in range(5, node_count)),
    ]
    table = BiologicalIdentityTable(
        records=tuple(
            BiologicalResidueIdentity(
                index=index,
                chain_id="A",
                residue_number=index + 1,
                insertion_code="",
                residue_name=name,
                node_id=f"A:{index + 1}::{name}",
            )
            for index, name in enumerate(names)
        )
    )
    records = tuple(
        TransportNodeRecord(
            transport_index=index,
            biological_index=index,
            residue_name=name,
            coordinate=coordinates[index],
        )
        for index, name in enumerate(names)
    )
    prepared = construct_gearnet_input(records, table)
    graph = prepared.graph
    edge_list = _as_numpy(graph.edge_list)
    relation_five = edge_list[:, 2] == 5
    arrays = {
        "node_position": _as_numpy(graph.node_position),
        "spatial_edge_source": edge_list[relation_five, 0],
        "spatial_edge_destination": edge_list[relation_five, 1],
    }
    return arrays


def regenerate_spatial_reference(*, overwrite: bool) -> None:
    """Create supplemental SpatialEdge and max-neighbor legacy parity fixtures."""

    _require_legacy_environment()
    _validate_targets(SPATIAL_OUTPUT_PATHS, overwrite=overwrite)
    backend = GearNetEdgeBackend()
    config = GearNetEdgeConfig(
        representation_name=GEARNET_EDGE_REPRESENTATION_NAME,
        output_mode="both",
    )
    first = backend.preprocess(
        StructureInput(source=str(SPATIAL_INPUT_PATH), chain_id="A", model_index=0),
        config=config,
    ).prepared
    second = backend.preprocess(
        StructureInput(source=str(SPATIAL_INPUT_PATH), chain_id="A", model_index=0),
        config=config,
    ).prepared
    first_arrays = _graph_arrays(first, _line_graph(first.graph))
    second_arrays = _graph_arrays(second, _line_graph(second.graph))
    _assert_exact_arrays(first_arrays, second_arrays)
    counts = _relation_counts(first_arrays["edge_relation"])
    if counts["5"] == 0:
        raise RuntimeError("Supplemental spatial reference has zero SpatialEdge entries")
    observations = _spatial_observations(first_arrays)
    np.savez(SPATIAL_GRAPH_PATH, **first_arrays)

    first_probe = _max_neighbor_probe()
    second_probe = _max_neighbor_probe()
    _assert_exact_arrays(first_probe, second_probe)
    target_zero_sources = first_probe["spatial_edge_source"][
        first_probe["spatial_edge_destination"] == 0
    ]
    if target_zero_sources.shape != (32,):
        raise RuntimeError("Legacy max-neighbor probe did not retain exactly 32 spatial neighbors")
    np.savez(SPATIAL_PROBE_PATH, **first_probe)

    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "optional_dependencies": {
            name: _distribution_version(name)
            for name in (
                "torch",
                "torchdrug",
                "torch_scatter",
                "torch_cluster",
                "torch_sparse",
                "torch_geometric",
                "rdkit-pypi",
                "numpy",
                "ninja",
            )
        },
    }
    SPATIAL_METADATA_PATH.write_text(
        json.dumps(
            {
                "fixture_schema_version": 1,
                "purpose": "legacy spatial parity reference",
                "input": {
                    "filename": SPATIAL_INPUT_PATH.name,
                    "sha256": _sha256(SPATIAL_INPUT_PATH),
                    "chain_id": "A",
                    "model_index": 0,
                },
                "graph_parameters": {
                    "spatial_radius": 10.0,
                    "spatial_min_sequence_distance": 5,
                    "spatial_max_num_neighbors": 32,
                    "num_relation": GEARNET_EDGE_NUM_RELATION,
                    "edge_feature_width": GEARNET_EDGE_EDGE_INPUT_DIM,
                    "num_angle_bin": GEARNET_EDGE_NUM_ANGLE_BIN,
                },
                "relation_counts": counts,
                "legacy_environment": environment,
                "arrays": {"graph": SPATIAL_GRAPH_PATH.name},
                "observations": observations,
                "reference_generation_note": (
                    "Generated by the frozen Python-3.10 TorchDrug graph construction; "
                    "regenerating changes the structural parity baseline."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    SPATIAL_PROBE_METADATA_PATH.write_text(
        json.dumps(
            {
                "fixture_schema_version": 1,
                "purpose": "legacy spatial max-neighbor parity probe",
                "node_count": 40,
                "target_node": 0,
                "candidate_spatial_neighbors_before_truncation": 35,
                "max_num_neighbors": 32,
                "retained_spatial_neighbors": int(target_zero_sources.shape[0]),
                "retained_sources_in_legacy_order": target_zero_sources.tolist(),
                "legacy_environment": environment,
                "arrays": {"probe": SPATIAL_PROBE_PATH.name},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--spatial-reference",
        action="store_true",
        help="regenerate only the supplemental SpatialEdge reference fixtures",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly replace existing scientific golden fixtures",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    if arguments.spatial_reference:
        regenerate_spatial_reference(overwrite=arguments.overwrite)
    elif arguments.checkpoint is None:
        raise SystemExit("--checkpoint is required unless --spatial-reference is selected")
    else:
        regenerate(checkpoint_path=arguments.checkpoint, overwrite=arguments.overwrite)
