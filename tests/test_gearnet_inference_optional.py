from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchdrug")

from eris.core.learned.backends.gearnet.backend import GearNetEdgeBackend  # noqa: E402
from eris.core.learned.backends.gearnet.model import (  # noqa: E402
    GEARNET_EDGE_CHECKPOINT_IDENTIFIER,
    GEARNET_EDGE_CHECKPOINT_SHA256,
)
from eris.core.learned.checkpoints import CheckpointDescriptor, verify_checkpoint  # noqa: E402
from eris.core.learned.config import StructuralEmbedderConfig  # noqa: E402
from eris.core.learned.execution import ResolvedExecution  # noqa: E402
from eris.core.types import StructureInput  # noqa: E402

RESIDUES = (
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
)


def _atom_line(index: int, residue_name: str) -> str:
    x = 3.8 * (index - 1)
    y = 1.25 * ((index - 1) % 3)
    z = 0.75 * ((index - 1) % 2)
    return (
        f"ATOM  {index:5d}  CA  {residue_name:>3s} A{index:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{10.0:6.2f}           C  "
    )


def _checkpoint() -> Path:
    configured = os.environ.get("ERIS_GEARNET_CHECKPOINT")
    if not configured:
        pytest.skip("ERIS_GEARNET_CHECKPOINT is not configured")
    path = Path(configured)
    if not path.is_file():
        pytest.fail(f"ERIS_GEARNET_CHECKPOINT is not a file: {path}")
    return path


def test_real_cpu_fp32_inference_modes_and_repeatability(tmp_path):
    source = tmp_path / "canonical-12.pdb"
    source.write_text(
        "\n".join(
            [
                *(_atom_line(index, name) for index, name in enumerate(RESIDUES, start=1)),
                "TER",
                "END",
            ]
        )
        + "\n",
        encoding="ascii",
    )
    descriptor = CheckpointDescriptor(
        identifier=GEARNET_EDGE_CHECKPOINT_IDENTIFIER,
        sha256=GEARNET_EDGE_CHECKPOINT_SHA256,
        revision="Zenodo record v2",
        source="https://zenodo.org/records/7723075",
    )
    checkpoint = verify_checkpoint(_checkpoint(), descriptor)
    backend = GearNetEdgeBackend()
    execution = ResolvedExecution(device="cpu", precision="fp32")
    preprocessed = backend.preprocess(
        StructureInput(source=str(source), chain_id="A", model_index=0),
        config=StructuralEmbedderConfig(
            representation_name="test_gearnet_edge",
            output_mode="both",
        ),
    )
    model = backend.load(checkpoint, execution=execution)

    results = {}
    for mode in ("node", "global", "both"):
        results[mode] = backend.infer(
            model,
            preprocessed.prepared,
            config=StructuralEmbedderConfig(
                representation_name="test_gearnet_edge",
                output_mode=mode,
            ),
            execution=execution,
        )

    expected_node_ids = np.asarray(
        [f"A:{index}::{name}" for index, name in enumerate(RESIDUES, start=1)],
        dtype=np.str_,
    )
    assert results["node"].node_embedding.shape == (12, 3072)
    np.testing.assert_array_equal(results["node"].node_ids, expected_node_ids)
    assert results["node"].global_embedding is None
    assert results["global"].node_embedding is None
    assert results["global"].node_ids is None
    assert results["global"].global_embedding.shape == (3072,)
    assert results["both"].node_embedding.shape == (12, 3072)
    assert results["both"].global_embedding.shape == (3072,)
    np.testing.assert_array_equal(results["both"].node_ids, expected_node_ids)

    repeated = backend.infer(
        model,
        preprocessed.prepared,
        config=StructuralEmbedderConfig(
            representation_name="test_gearnet_edge",
            output_mode="both",
        ),
        execution=execution,
    )
    assert np.array_equal(results["both"].node_embedding, repeated.node_embedding)
    assert np.array_equal(results["both"].global_embedding, repeated.global_embedding)
