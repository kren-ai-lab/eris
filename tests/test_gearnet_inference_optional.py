from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchdrug")

import eris.core.learned.runner as runner_module  # noqa: E402
from eris.core.artifacts import StructureArtifact  # noqa: E402
from eris.core.cache import DiskCache  # noqa: E402
from eris.core.learned.backends.gearnet.backend import GearNetEdgeBackend  # noqa: E402
from eris.core.learned.backends.gearnet.config import GearNetEdgeConfig  # noqa: E402
from eris.core.learned.backends.gearnet.metadata import (  # noqa: E402
    GEARNET_EDGE_CHECKPOINT_DESCRIPTOR,
    GEARNET_EDGE_ENCODER_VERSION,
    GEARNET_EDGE_IMPLEMENTATION,
    GEARNET_EDGE_MODEL_DESCRIPTOR,
    GEARNET_EDGE_REPRESENTATION_NAME,
)
from eris.core.learned.backends.gearnet.preprocessing import (  # noqa: E402
    GearNetPreprocessingError,
)
from eris.core.learned.checkpoint_cache import CheckpointCache  # noqa: E402
from eris.core.learned.checkpoints import verify_checkpoint  # noqa: E402
from eris.core.learned.config import StructuralEmbedderConfig  # noqa: E402
from eris.core.learned.execution import (  # noqa: E402
    ResolvedExecution,
    StructuralExecutionConfig,
)
from eris.core.learned.registry import learned_registry  # noqa: E402
from eris.core.learned.runner import StructuralEmbedderRunner  # noqa: E402
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
    checkpoint = verify_checkpoint(_checkpoint(), GEARNET_EDGE_CHECKPOINT_DESCRIPTOR)
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


def test_real_runner_builds_strict_gearnet_artifact_and_reuses_generation_provenance(
    tmp_path,
    monkeypatch,
):
    source_a = tmp_path / "locator-a" / "canonical-12.pdb"
    source_b = tmp_path / "locator-b" / "canonical-12.pdb"
    source_a.parent.mkdir()
    source_b.parent.mkdir()
    structure_bytes = (
        "\n".join(
            [
                *(_atom_line(index, name) for index, name in enumerate(RESIDUES, start=1)),
                "TER",
                "END",
            ]
        )
        + "\n"
    ).encode("ascii")
    source_a.write_bytes(structure_bytes)
    source_b.write_bytes(structure_bytes)

    calls = {
        "availability": 0,
        "factory": 0,
        "execution": 0,
        "checkpoint": 0,
        "preprocess": 0,
        "load": 0,
        "infer": 0,
    }
    cache_hit_only = False
    real_availability = learned_registry.availability
    real_create = learned_registry.create
    real_resolve_execution = runner_module.resolve_execution
    real_resolve_checkpoint = StructuralEmbedderRunner._resolve_checkpoint
    real_preprocess = GearNetEdgeBackend.preprocess
    real_load = GearNetEdgeBackend.load
    real_infer = GearNetEdgeBackend.infer

    def count(name, callback):
        def wrapped(*args, **kwargs):
            if cache_hit_only:
                raise AssertionError(f"artifact cache hit must not execute {name}")
            calls[name] += 1
            return callback(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(learned_registry, "availability", count("availability", real_availability))
    monkeypatch.setattr(learned_registry, "create", count("factory", real_create))
    monkeypatch.setattr(
        runner_module,
        "resolve_execution",
        count("execution", real_resolve_execution),
    )
    monkeypatch.setattr(
        StructuralEmbedderRunner,
        "_resolve_checkpoint",
        count("checkpoint", real_resolve_checkpoint),
    )
    monkeypatch.setattr(
        GearNetEdgeBackend,
        "preprocess",
        count("preprocess", real_preprocess),
    )
    monkeypatch.setattr(GearNetEdgeBackend, "load", count("load", real_load))
    monkeypatch.setattr(GearNetEdgeBackend, "infer", count("infer", real_infer))
    runner = StructuralEmbedderRunner(
        artifact_cache=DiskCache(tmp_path / "artifact-cache"),
        checkpoint_cache=CheckpointCache(tmp_path / "checkpoint-cache"),
    )
    config = GearNetEdgeConfig(
        representation_name=GEARNET_EDGE_REPRESENTATION_NAME,
        output_mode="both",
    )
    execution = StructuralExecutionConfig(device="cpu", precision="fp32")

    first = runner.run(
        StructureInput(source=str(source_a), chain_id="A", model_index=0),
        config=config,
        model=GEARNET_EDGE_MODEL_DESCRIPTOR,
        checkpoint_path=_checkpoint(),
        execution=execution,
    )

    first.validate(strict=True)
    assert learned_registry.names() == ["gearnet_edge"]
    assert calls == {name: 1 for name in calls}
    assert first.meta["schema_version"] == 2
    assert first.meta["representation_name"] == "gearnet_edge"
    assert first.meta["representation_family"] == "learned"
    assert first.meta["representation_type"] == "embedding"
    assert first.meta["granularity"] == "mixed"
    assert first.meta["coordinate_source"] == "model_specific"
    assert first.meta["encoder"] == {
        "implementation": GEARNET_EDGE_IMPLEMENTATION,
        "version": GEARNET_EDGE_ENCODER_VERSION,
    }
    assert first.meta["model"] == GEARNET_EDGE_MODEL_DESCRIPTOR.to_dict()
    assert first.meta["parameters"] == config.to_dict()
    assert first.meta["feature_keys"] == {}
    assert first.meta["shapes"] == {
        "node_embedding": [12, 3072],
        "global_embedding": [3072],
        "node_ids": [12],
    }
    assert first.meta["dtypes"] == {
        "node_embedding": "float32",
        "global_embedding": "float32",
        "node_ids": str(first.data["node_ids"].dtype),
    }
    assert first.meta["counts"] == {"n_nodes": 12}
    assert first.meta["provenance"]["environment"]["execution"] == {
        "requested": {"device": "cpu", "precision": "fp32"},
        "resolved": {"device": "cpu", "precision": "fp32"},
    }
    assert first.data["node_ids"].tolist() == [
        f"A:{index}::{name}" for index, name in enumerate(RESIDUES, start=1)
    ]
    assert first.data["node_embedding"].dtype == np.float32
    assert first.data["global_embedding"].dtype == np.float32
    assert first.data["node_ids"].dtype.kind == "U"
    assert np.isfinite(first.data["node_embedding"]).all()
    assert np.isfinite(first.data["global_embedding"]).all()
    assert len(set(first.data["node_ids"].tolist())) == 12
    assert all(first.data["node_ids"])
    assert "valid_mask" not in first.data
    serialized = json.dumps(first.meta, sort_keys=True)
    assert _checkpoint().parent.name not in serialized
    assert "checkpoint-cache" not in serialized
    assert "eris-gearnet-" not in serialized
    assert "torch_ext" not in serialized

    saved = tmp_path / "saved-artifact"
    first.save(saved)
    reloaded = StructureArtifact.load(saved, validate=True, strict=True)
    assert reloaded.meta == first.meta
    for name in first.data:
        np.testing.assert_array_equal(reloaded.data[name], first.data[name])

    cache_hit_only = True
    second = runner.run(
        StructureInput(source=str(source_b), chain_id="A", model_index=0),
        config=config,
        model=GEARNET_EDGE_MODEL_DESCRIPTOR,
        execution=StructuralExecutionConfig(device="auto", precision="fp32"),
    )

    second.validate(strict=True)
    assert calls == {name: 1 for name in calls}
    assert second.meta["input"]["source"] == str(source_b)
    assert second.meta["provenance"] == first.meta["provenance"]
    assert second.meta["model"] == first.meta["model"]
    for name in first.data:
        np.testing.assert_array_equal(second.data[name], first.data[name])

    cache_hit_only = False
    independent = StructuralEmbedderRunner(
        artifact_cache=DiskCache(tmp_path / "independent-artifact-cache"),
        checkpoint_cache=CheckpointCache(tmp_path / "independent-checkpoint-cache"),
    ).run(
        StructureInput(source=str(source_a), chain_id="A", model_index=0),
        config=config,
        model=GEARNET_EDGE_MODEL_DESCRIPTOR,
        checkpoint_path=_checkpoint(),
        execution=execution,
    )
    assert calls == {name: 2 for name in calls}
    for name in ("node_embedding", "global_embedding", "node_ids"):
        assert np.array_equal(independent.data[name], first.data[name])


def test_real_runner_missing_ca_fails_before_load_and_leaves_no_artifact(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "missing-ca.pdb"
    source.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N  \n"
        "TER\nEND\n",
        encoding="ascii",
    )
    artifact_cache_dir = tmp_path / "artifact-cache"
    runner = StructuralEmbedderRunner(
        artifact_cache=DiskCache(artifact_cache_dir),
        checkpoint_cache=CheckpointCache(tmp_path / "checkpoint-cache"),
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("missing-CA input must fail before model load or inference")

    monkeypatch.setattr(GearNetEdgeBackend, "load", unexpected)
    monkeypatch.setattr(GearNetEdgeBackend, "infer", unexpected)

    with pytest.raises(GearNetPreprocessingError, match="missing CA"):
        runner.run(
            StructureInput(source=str(source), chain_id="A", model_index=0),
            config=GearNetEdgeConfig(
                representation_name=GEARNET_EDGE_REPRESENTATION_NAME,
                output_mode="both",
            ),
            model=GEARNET_EDGE_MODEL_DESCRIPTOR,
            checkpoint_path=_checkpoint(),
            execution=StructuralExecutionConfig(device="cpu", precision="fp32"),
        )

    artifact_entries = list((artifact_cache_dir / "v2").glob("*"))
    assert artifact_entries == []
