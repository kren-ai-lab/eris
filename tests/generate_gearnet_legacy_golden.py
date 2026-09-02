"""Explicitly regenerate the frozen Python-3.10 TorchDrug GearNet reference.

Run only in the documented legacy CPU/fp32 environment, for example after entering
the MSVC x64 developer environment on Windows:

    python tests/generate_gearnet_legacy_golden.py \
        --checkpoint C:\\path\\to\\mc_gearnet_edge.pth --overwrite

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
from eris.core.learned.checkpoints import verify_checkpoint
from eris.core.learned.execution import ResolvedExecution
from eris.core.types import StructureInput


DATA_DIR = Path(__file__).with_name("data")
INPUT_PATH = DATA_DIR / "gearnet_legacy_reference.pdb"
GRAPH_PATH = DATA_DIR / "gearnet_legacy_reference_graph.npz"
EMBEDDING_PATH = DATA_DIR / "gearnet_legacy_reference_embeddings.npz"
METADATA_PATH = DATA_DIR / "gearnet_legacy_reference.json"
OUTPUT_PATHS = (GRAPH_PATH, EMBEDDING_PATH, METADATA_PATH)


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


def _validate_output_targets(*, overwrite: bool) -> None:
    existing = [path for path in OUTPUT_PATHS if path.exists()]
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
    _validate_output_targets(overwrite=overwrite)
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly replace existing scientific golden fixtures",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    regenerate(checkpoint_path=arguments.checkpoint, overwrite=arguments.overwrite)
