from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import numpy as np
import pytest

from eris.core.artifact_schema_v2 import compute_config_fingerprint
from eris.core.learned import (
    CheckpointDescriptor,
    ExecutionCapabilities,
    ResolvedExecution,
    StructuralEmbedderBackend,
    StructuralEmbedderConfig,
    StructuralEmbeddingResult,
    StructuralExecutionConfig,
    VerifiedCheckpoint,
    resolve_execution,
    validate_embedding_result,
)
from eris.exceptions import ConfigError, ExecutionError


class FakeBackend:
    backend_id = "fake"

    def __init__(self) -> None:
        self.loaded_checkpoint: VerifiedCheckpoint | None = None
        self.loaded_execution: ResolvedExecution | None = None
        self.inferred_executions: list[ResolvedExecution] = []

    def is_available(self) -> bool:
        return True

    def execution_capabilities(self) -> ExecutionCapabilities:
        return ExecutionCapabilities(
            devices=("cpu",),
            precisions=("fp32",),
            preferred_device="cpu",
        )

    def preprocess(
        self,
        structure: tuple[str, ...],
        *,
        config: StructuralEmbedderConfig,
    ) -> tuple[str, ...]:
        del config
        return tuple(structure)

    def load(
        self,
        checkpoint: VerifiedCheckpoint,
        *,
        execution: ResolvedExecution,
    ) -> str:
        self.loaded_checkpoint = checkpoint
        self.loaded_execution = execution
        return "fake-model"

    def infer(
        self,
        model: str,
        prepared: tuple[str, ...],
        *,
        config: StructuralEmbedderConfig,
        execution: ResolvedExecution,
    ) -> StructuralEmbeddingResult:
        assert model == "fake-model"
        self.inferred_executions.append(execution)
        node_embedding = np.arange(len(prepared) * 2, dtype=np.float32).reshape(-1, 2)
        global_embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        node_ids = np.asarray(prepared, dtype=np.str_)
        valid_mask = np.ones(len(prepared), dtype=np.bool_)

        return StructuralEmbeddingResult(
            node_embedding=node_embedding if config.output_mode != "global" else None,
            global_embedding=global_embedding if config.output_mode != "node" else None,
            node_ids=node_ids if config.output_mode != "global" else None,
            valid_mask=valid_mask if config.output_mode != "global" else None,
        )


def _node_result(**overrides: object) -> StructuralEmbeddingResult:
    values = {
        "node_embedding": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "node_ids": np.array(["A:1", "A:2"], dtype=np.str_),
    }
    values.update(overrides)
    return StructuralEmbeddingResult(**values)


def _global_result(**overrides: object) -> StructuralEmbeddingResult:
    values = {"global_embedding": np.array([0.2, 0.3], dtype=np.float32)}
    values.update(overrides)
    return StructuralEmbeddingResult(**values)


def _both_result(**overrides: object) -> StructuralEmbeddingResult:
    values = {
        "node_embedding": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        "global_embedding": np.array([0.2, 0.3], dtype=np.float32),
        "node_ids": np.array(["A:1", "A:2"], dtype=np.str_),
    }
    values.update(overrides)
    return StructuralEmbeddingResult(**values)


def test_structural_embedder_config_is_frozen_and_defaults_to_both():
    config = StructuralEmbedderConfig(representation_name="synthetic_embedding")

    assert config.output_mode == "both"
    with pytest.raises(FrozenInstanceError):
        config.output_mode = "node"


@pytest.mark.parametrize("output_mode", ["node", "global", "both"])
def test_structural_embedder_config_accepts_canonical_output_modes(output_mode):
    config = StructuralEmbedderConfig(
        representation_name="synthetic_embedding",
        output_mode=output_mode,
    )

    assert config.output_mode == output_mode


@pytest.mark.parametrize("representation_name", ["", "   ", None, 1])
def test_structural_embedder_config_rejects_invalid_representation_names(representation_name):
    with pytest.raises(ConfigError, match="representation_name"):
        StructuralEmbedderConfig(representation_name=representation_name)


@pytest.mark.parametrize("output_mode", ["NODE", "local", "", None, []])
def test_structural_embedder_config_rejects_unknown_output_modes(output_mode):
    with pytest.raises(ConfigError, match="output_mode"):
        StructuralEmbedderConfig(
            representation_name="synthetic_embedding",
            output_mode=output_mode,
        )


def test_structural_embedder_config_is_deterministic_and_json_compatible():
    config = StructuralEmbedderConfig(
        representation_name=" synthetic_embedding ",
        output_mode="node",
    )

    assert config.to_dict() == {
        "representation_name": "synthetic_embedding",
        "output_mode": "node",
    }
    assert config.to_dict() == config.to_dict()
    assert {field.name for field in fields(config)} == {"representation_name", "output_mode"}


def test_structural_embedder_config_uses_existing_fingerprint_contract():
    config = StructuralEmbedderConfig(
        representation_name="synthetic_embedding",
        output_mode="both",
    )

    left = compute_config_fingerprint(config.representation_name, config.to_dict())
    right = compute_config_fingerprint(config.representation_name, config.to_dict())

    assert left == right
    assert len(left) == 64


@pytest.mark.parametrize(
    ("result", "output_mode"),
    [
        (_node_result(), "node"),
        (_global_result(), "global"),
        (_both_result(), "both"),
        (
            _both_result(valid_mask=np.array([True, False], dtype=np.bool_)),
            "both",
        ),
    ],
)
def test_valid_embedding_results(result, output_mode):
    validate_embedding_result(result, output_mode=output_mode)


@pytest.mark.parametrize(
    ("result", "match"),
    [
        (_node_result(node_embedding=np.array([0.1, 0.2], dtype=np.float32)), "two-dimensional"),
        (_node_result(node_embedding=np.array([[1, 2]], dtype=np.int64)), "floating dtype"),
        (_node_result(node_embedding=np.empty((0, 2), dtype=np.float32)), "non-zero"),
        (_node_result(node_embedding=np.empty((2, 0), dtype=np.float32)), "non-zero"),
        (_node_result(node_embedding=np.array([[np.nan, 0.2]], dtype=np.float32)), "finite"),
        (_node_result(node_embedding=np.array([[np.inf, 0.2]], dtype=np.float32)), "finite"),
        (
            _global_result(global_embedding=np.array([[0.1, 0.2]], dtype=np.float32)),
            "one-dimensional",
        ),
        (_global_result(global_embedding=np.array([1, 2], dtype=np.int64)), "floating dtype"),
        (_global_result(global_embedding=np.array([], dtype=np.float32)), "non-zero"),
        (_global_result(global_embedding=np.array([np.nan], dtype=np.float32)), "finite"),
        (_global_result(global_embedding=np.array([np.inf], dtype=np.float32)), "finite"),
        (_node_result(node_ids=None), "requires node_ids"),
        (_node_result(node_ids=np.array(["A:1"], dtype=np.str_)), "must align"),
        (
            _node_result(node_ids=np.array([["A:1"], ["A:2"]], dtype=np.str_)),
            "one-dimensional",
        ),
        (_node_result(node_ids=np.array(["A:1", ""], dtype=np.str_)), "non-empty"),
        (_node_result(node_ids=np.array(["A:1", "A:1"], dtype=np.str_)), "unique"),
        (
            _node_result(valid_mask=np.array([1, 0], dtype=np.int8)),
            "bool dtype",
        ),
        (
            _node_result(valid_mask=np.array([[True], [False]], dtype=np.bool_)),
            "one-dimensional",
        ),
        (
            _node_result(valid_mask=np.array([True], dtype=np.bool_)),
            "must align",
        ),
    ],
)
def test_invalid_embedding_result_values_are_rejected(result, match):
    mode = "global" if result.node_embedding is None and result.node_ids is None else "node"

    with pytest.raises(ExecutionError, match=match):
        validate_embedding_result(result, output_mode=mode)


@pytest.mark.parametrize(
    ("result", "output_mode", "match"),
    [
        (_global_result(), "node", "requires node_embedding"),
        (_node_result(), "global", "requires global_embedding"),
        (_node_result(), "both", "requires node_embedding, node_ids, and global_embedding"),
        (_both_result(), "node", "forbids global_embedding"),
        (_both_result(), "global", "forbids node_embedding"),
    ],
)
def test_output_mode_requires_exact_primary_outputs(result, output_mode, match):
    with pytest.raises(ExecutionError, match=match):
        validate_embedding_result(result, output_mode=output_mode)


def test_result_requires_numpy_arrays_and_unicode_node_ids():
    with pytest.raises(ExecutionError, match="NumPy array"):
        validate_embedding_result(
            StructuralEmbeddingResult(node_embedding=[[0.1]], node_ids=np.array(["A:1"])),
            output_mode="node",
        )

    with pytest.raises(ExecutionError, match="Unicode"):
        validate_embedding_result(
            _node_result(node_ids=np.array(["A:1", "A:2"], dtype=object)),
            output_mode="node",
        )


def test_fake_backend_structurally_implements_protocol_without_inheritance():
    backend: StructuralEmbedderBackend[tuple[str, ...], tuple[str, ...], str] = FakeBackend()

    assert isinstance(backend, StructuralEmbedderBackend)
    assert backend.backend_id == "fake"
    assert backend.is_available() is True


@pytest.mark.parametrize("output_mode", ["node", "global", "both"])
def test_fake_backend_outputs_are_deterministic_and_valid(output_mode):
    backend = FakeBackend()
    checkpoint = VerifiedCheckpoint(
        path=Path("fake-checkpoint.bin"),
        descriptor=CheckpointDescriptor(
            identifier="fake-checkpoint",
            sha256="a" * 64,
        ),
        sha256="a" * 64,
    )
    config = StructuralEmbedderConfig(
        representation_name="synthetic_embedding",
        output_mode=output_mode,
    )
    prepared = backend.preprocess(("A:1", "A:2"), config=config)
    execution = resolve_execution(
        StructuralExecutionConfig(),
        backend.execution_capabilities(),
    )
    model = backend.load(checkpoint, execution=execution)

    assert backend.loaded_checkpoint is checkpoint
    assert backend.loaded_execution is execution

    left = backend.infer(model, prepared, config=config, execution=execution)
    right = backend.infer(model, prepared, config=config, execution=execution)

    assert backend.inferred_executions == [execution, execution]

    validate_embedding_result(left, output_mode=config.output_mode)
    validate_embedding_result(right, output_mode=config.output_mode)
    for name in ("node_embedding", "global_embedding", "node_ids", "valid_mask"):
        left_value = getattr(left, name)
        right_value = getattr(right, name)
        if left_value is None:
            assert right_value is None
        else:
            np.testing.assert_array_equal(left_value, right_value)


def test_importing_learned_contract_does_not_load_optional_frameworks():
    script = """
import sys
import eris.core.learned

blocked = {'torch', 'torchdrug', 'torch_geometric', 'transformers'}
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise SystemExit(f'optional frameworks loaded: {loaded}')
"""

    subprocess.run([sys.executable, "-c", script], check=True)


def test_learned_contract_is_not_exposed_from_top_level_eris():
    import eris

    assert not hasattr(eris, "StructuralEmbedderBackend")
    assert not hasattr(eris, "StructuralEmbedderConfig")
