from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import eris.core.learned.runner as runner_module
from eris.core.artifacts import StructureArtifact
from eris.core.cache import DiskCache
from eris.core.hashing import sha256_file
from eris.core.io import load_and_extract_residue
from eris.core.learned import (
    BackendAvailability,
    CheckpointCache,
    CheckpointDescriptor,
    EmbedderRegistry,
    ExecutionCapabilities,
    ResolvedExecution,
    StructuralEmbedderConfig,
    StructuralEmbedderRunner,
    StructuralEmbeddingResult,
    StructuralExecutionConfig,
    StructuralModelDescriptor,
    StructuralPreprocessingResult,
    learned_registry,
)
from eris.core.types import StructureInput
from eris.exceptions import ConfigError, ExecutionError

PDB_BYTES = b"""\
ATOM      1  CA  ALA A   1      11.000  12.000  13.000  1.00 10.00           C
ATOM      2  CA  GLY A   2      14.000  15.000  16.000  1.00 20.00           C
ATOM      3  CA  SER A   3      17.000  18.000  19.000  1.00 30.00           C
TER
END
"""
CHECKPOINT_BYTES = b"eris fake learned checkpoint\n"
REPRESENTATION_NAME = "fake_structure_embedding"
ENCODER_VERSION = "test-adapter-v1"


@dataclass(frozen=True, kw_only=True)
class FakePrepared:
    residue_ids: np.ndarray


@dataclass(frozen=True, kw_only=True)
class FakeModel:
    token: bytes
    execution: ResolvedExecution


@dataclass
class FakeState:
    available: bool = True
    probe_calls: int = 0
    factory_calls: int = 0
    instance_availability_calls: int = 0
    capabilities_calls: int = 0
    preprocess_calls: int = 0
    load_calls: int = 0
    infer_calls: int = 0
    loaded_checkpoint_path: Path | None = None
    load_execution: ResolvedExecution | None = None
    infer_execution: ResolvedExecution | None = None


class FakeBackend:
    backend_id = "fake"

    def __init__(
        self,
        state: FakeState,
        capabilities: ExecutionCapabilities,
        *,
        confirmed_format: str = "pdb",
    ) -> None:
        self.state = state
        self.capabilities = capabilities
        self.confirmed_format = confirmed_format

    def is_available(self) -> bool:
        self.state.instance_availability_calls += 1
        return self.state.available

    def execution_capabilities(self) -> ExecutionCapabilities:
        self.state.capabilities_calls += 1
        return self.capabilities

    def preprocess(
        self,
        structure: StructureInput,
        *,
        config: StructuralEmbedderConfig,
    ) -> StructuralPreprocessingResult[FakePrepared]:
        del config
        self.state.preprocess_calls += 1
        loaded = load_and_extract_residue(structure)
        input_format = (
            loaded.input_format if self.confirmed_format == "pdb" else self.confirmed_format
        )
        return StructuralPreprocessingResult(
            prepared=FakePrepared(residue_ids=loaded["residue_ids"]),
            input_format=input_format,
        )

    def load(
        self,
        checkpoint,
        *,
        execution: ResolvedExecution,
    ) -> FakeModel:
        self.state.load_calls += 1
        self.state.loaded_checkpoint_path = checkpoint.path
        self.state.load_execution = execution
        return FakeModel(token=checkpoint.path.read_bytes(), execution=execution)

    def infer(
        self,
        model: FakeModel,
        prepared: FakePrepared,
        *,
        config: StructuralEmbedderConfig,
        execution: ResolvedExecution,
    ) -> StructuralEmbeddingResult:
        self.state.infer_calls += 1
        self.state.infer_execution = execution
        assert model.token == CHECKPOINT_BYTES
        assert model.execution is execution

        scale = 1.0 if execution.precision == "fp32" else 2.0
        node_embedding = (
            np.arange(prepared.residue_ids.shape[0] * 3, dtype=np.float32).reshape(-1, 3) * scale
        )
        global_embedding = node_embedding.mean(axis=0, dtype=np.float32)
        valid_mask = np.ones(prepared.residue_ids.shape[0], dtype=np.bool_)
        return StructuralEmbeddingResult(
            node_embedding=node_embedding if config.output_mode != "global" else None,
            global_embedding=global_embedding if config.output_mode != "node" else None,
            node_ids=prepared.residue_ids if config.output_mode != "global" else None,
            valid_mask=valid_mask if config.output_mode != "global" else None,
        )


class CountingArtifactCache(DiskCache):
    def __init__(self, cache_dir: Path) -> None:
        super().__init__(cache_dir)
        self.get_calls = 0
        self.set_calls = 0
        self.keys: list[str] = []

    def get_v2(self, key: str):
        self.get_calls += 1
        self.keys.append(key)
        return super().get_v2(key)

    def set_v2(self, key: str, artifact: StructureArtifact) -> None:
        self.set_calls += 1
        super().set_v2(key, artifact)


class CountingCheckpointCache(CheckpointCache):
    def __init__(self, cache_dir: Path) -> None:
        super().__init__(cache_dir)
        self.get_calls = 0
        self.put_calls = 0

    def get(self, descriptor):
        self.get_calls += 1
        return super().get(descriptor)

    def put(self, path, descriptor):
        self.put_calls += 1
        return super().put(path, descriptor)


@dataclass(frozen=True, kw_only=True)
class SpecializedConfig(StructuralEmbedderConfig):
    scale: int = 1


def _write_inputs(tmp_path: Path, *, directory: str = "input") -> tuple[Path, Path]:
    source = tmp_path / directory / "structure.pdb"
    checkpoint = tmp_path / "candidate" / "weights.bin"
    source.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(PDB_BYTES)
    checkpoint.write_bytes(CHECKPOINT_BYTES)
    return source, checkpoint


def _model(checkpoint: Path, *, source: str = "test://mirror-a") -> StructuralModelDescriptor:
    return StructuralModelDescriptor(
        architecture="fake-structural-model",
        checkpoint=CheckpointDescriptor(
            identifier="fake-checkpoint",
            revision="test-revision",
            source=source,
            sha256=sha256_file(checkpoint),
        ),
    )


def _runner_environment(
    tmp_path: Path,
    *,
    devices: tuple[str, ...] = ("cpu", "cuda:0"),
    precisions: tuple[str, ...] = ("fp32", "fp16"),
    preferred_device: str = "cpu",
    confirmed_format: str = "pdb",
    config_type: type[StructuralEmbedderConfig] = StructuralEmbedderConfig,
):
    source, checkpoint = _write_inputs(tmp_path)
    state = FakeState()
    capabilities = ExecutionCapabilities(
        devices=devices,
        precisions=precisions,
        preferred_device=preferred_device,
    )
    registry = EmbedderRegistry()

    def probe() -> BackendAvailability:
        state.probe_calls += 1
        return (
            BackendAvailability(available=True)
            if state.available
            else BackendAvailability(available=False, reason="fake backend unavailable")
        )

    def factory() -> FakeBackend:
        state.factory_calls += 1
        return FakeBackend(
            state,
            capabilities,
            confirmed_format=confirmed_format,
        )

    registry.register(
        representation_name=REPRESENTATION_NAME,
        backend_id="fake",
        implementation=f"{FakeBackend.__module__}.{FakeBackend.__qualname__}",
        encoder_version=ENCODER_VERSION,
        config_type=config_type,
        backend_factory=factory,
        availability_probe=probe,
    )
    artifact_cache = CountingArtifactCache(tmp_path / "artifact-cache")
    checkpoint_cache = CountingCheckpointCache(tmp_path / "checkpoint-cache")
    runner = StructuralEmbedderRunner(
        registry=registry,
        artifact_cache=artifact_cache,
        checkpoint_cache=checkpoint_cache,
    )
    return runner, state, artifact_cache, checkpoint_cache, source, checkpoint


def _config(output_mode: str = "both") -> StructuralEmbedderConfig:
    return StructuralEmbedderConfig(
        representation_name=REPRESENTATION_NAME,
        output_mode=output_mode,
    )


def test_complete_fake_path_round_trip_and_lazy_artifact_hit(tmp_path):
    runner, state, artifact_cache, checkpoint_cache, source, checkpoint = _runner_environment(
        tmp_path
    )
    model = _model(checkpoint)
    structure_input = StructureInput(source=str(source))
    execution = StructuralExecutionConfig(device="cpu", precision="fp32")

    first = runner.run(
        structure_input,
        config=_config(),
        model=model,
        checkpoint_path=checkpoint,
        execution=execution,
    )

    first.validate(strict=True)
    assert list(first.data) == [
        "node_embedding",
        "global_embedding",
        "node_ids",
        "valid_mask",
    ]
    assert first.data["node_ids"].tolist() == ["A:1::ALA", "A:2::GLY", "A:3::SER"]
    assert first.data["node_embedding"].shape[0] == first.data["node_ids"].shape[0]
    assert first.data["valid_mask"].shape[0] == first.data["node_ids"].shape[0]
    assert first.meta["provenance"]["environment"]["execution"] == {
        "requested": {"device": "cpu", "precision": "fp32"},
        "resolved": {"device": "cpu", "precision": "fp32"},
    }
    assert state.load_execution is state.infer_execution
    assert state.probe_calls == state.factory_calls == 1
    assert state.capabilities_calls == state.preprocess_calls == 1
    assert state.load_calls == state.infer_calls == 1
    assert artifact_cache.get_calls == artifact_cache.set_calls == 1
    assert checkpoint_cache.put_calls == 1

    outdir = tmp_path / "saved-artifact"
    first.save(outdir)
    loaded = StructureArtifact.load(outdir)
    assert loaded.meta == first.meta
    for name, expected in first.data.items():
        np.testing.assert_array_equal(loaded.data[name], expected)

    checkpoint_get_calls = checkpoint_cache.get_calls
    checkpoint_put_calls = checkpoint_cache.put_calls
    checkpoint.unlink()
    assert state.loaded_checkpoint_path is not None
    state.loaded_checkpoint_path.unlink()
    state.available = False

    second = runner.run(
        structure_input,
        config=_config(),
        model=model,
        execution=StructuralExecutionConfig(device="cuda:99", precision="fp32"),
    )

    assert second.meta == first.meta
    assert state.probe_calls == state.factory_calls == 1
    assert state.capabilities_calls == state.preprocess_calls == 1
    assert state.load_calls == state.infer_calls == 1
    assert checkpoint_cache.get_calls == checkpoint_get_calls
    assert checkpoint_cache.put_calls == checkpoint_put_calls
    assert artifact_cache.get_calls == 2
    assert artifact_cache.set_calls == 1


def test_precision_and_output_mode_partition_artifact_cache(tmp_path):
    runner, state, artifact_cache, _, source, checkpoint = _runner_environment(tmp_path)
    model = _model(checkpoint)
    structure_input = StructureInput(source=str(source))

    fp32 = runner.run(
        structure_input,
        config=_config("both"),
        model=model,
        checkpoint_path=checkpoint,
        execution=StructuralExecutionConfig(device="cpu", precision="fp32"),
    )
    fp16 = runner.run(
        structure_input,
        config=_config("both"),
        model=model,
        execution=StructuralExecutionConfig(device="cpu", precision="fp16"),
    )
    node = runner.run(
        structure_input,
        config=_config("node"),
        model=model,
        execution=StructuralExecutionConfig(device="cpu", precision="fp32"),
    )

    assert state.infer_calls == 3
    assert len(set(artifact_cache.keys)) == 3
    assert not np.array_equal(fp32.data["node_embedding"], fp16.data["node_embedding"])
    assert fp32.meta["provenance"]["environment"]["execution"]["resolved"]["precision"] == "fp32"
    assert fp16.meta["provenance"]["environment"]["execution"]["resolved"]["precision"] == "fp16"
    assert list(node.data) == ["node_embedding", "node_ids", "valid_mask"]


def test_auto_resolution_records_preferred_device_without_hardware_probe(tmp_path):
    runner, state, _, _, source, checkpoint = _runner_environment(
        tmp_path,
        preferred_device="cuda:0",
    )

    artifact = runner.run(
        StructureInput(source=str(source)),
        config=_config(),
        model=_model(checkpoint),
        checkpoint_path=checkpoint,
    )

    assert artifact.meta["provenance"]["environment"]["execution"] == {
        "requested": {"device": "auto", "precision": "fp32"},
        "resolved": {"device": "cuda:0", "precision": "fp32"},
    }
    assert state.load_execution is state.infer_execution


def test_model_mirror_and_input_relocation_reuse_generation_artifact(tmp_path):
    runner, state, artifact_cache, _, source_a, checkpoint = _runner_environment(tmp_path)
    source_b = tmp_path / "relocated" / "structure.pdb"
    source_b.parent.mkdir()
    source_b.write_bytes(source_a.read_bytes())
    model_a = _model(checkpoint, source="test://mirror-a")
    model_b = _model(checkpoint, source="test://mirror-b")

    first = runner.run(
        StructureInput(source=str(source_a)),
        config=_config(),
        model=model_a,
        checkpoint_path=checkpoint,
    )
    second = runner.run(
        StructureInput(source=str(source_b)),
        config=_config(),
        model=model_b,
    )

    assert state.infer_calls == 1
    assert artifact_cache.keys[0] == artifact_cache.keys[1]
    assert first.meta["model"]["checkpoint"]["source"] == "test://mirror-a"
    assert second.meta["model"]["checkpoint"]["source"] == "test://mirror-a"
    assert second.meta["input"]["source"] == str(source_b)
    assert second.meta["provenance"] == first.meta["provenance"]
    cached = artifact_cache.get_v2(artifact_cache.keys[0])
    assert cached is not None
    assert cached.meta["input"]["source"] == str(source_a)


def test_checkpoint_miss_without_candidate_is_offline_and_actionable(tmp_path):
    runner, state, _, checkpoint_cache, source, checkpoint = _runner_environment(tmp_path)

    with pytest.raises(
        ExecutionError,
        match="fake-checkpoint.*local checkpoint_path is required",
    ):
        runner.run(
            StructureInput(source=str(source)),
            config=_config(),
            model=_model(checkpoint),
            checkpoint_path=None,
        )

    assert state.probe_calls == state.factory_calls == 1
    assert state.preprocess_calls == state.load_calls == state.infer_calls == 0
    assert checkpoint_cache.get_calls == 1


def test_unavailable_backend_fails_on_miss_before_checkpoint_access(tmp_path):
    runner, state, _, checkpoint_cache, source, checkpoint = _runner_environment(tmp_path)
    state.available = False

    with pytest.raises(ExecutionError, match="fake backend unavailable"):
        runner.run(
            StructureInput(source=str(source)),
            config=_config(),
            model=_model(checkpoint),
            checkpoint_path=checkpoint,
        )

    assert state.probe_calls == 1
    assert state.factory_calls == 0
    assert checkpoint_cache.get_calls == checkpoint_cache.put_calls == 0


@pytest.mark.parametrize(
    "mutation",
    ["encoder", "model", "granularity", "outputs", "precision", "missing_execution"],
)
def test_wrong_learned_identity_under_expected_key_is_a_miss(tmp_path, mutation):
    runner, state, artifact_cache, _, source, checkpoint = _runner_environment(tmp_path)
    structure_input = StructureInput(source=str(source))
    model = _model(checkpoint)
    runner.run(
        structure_input,
        config=_config(),
        model=model,
        checkpoint_path=checkpoint,
    )
    entry = artifact_cache.cache_dir / "v2" / artifact_cache.keys[0] / "meta.json"
    metadata = json.loads(entry.read_text(encoding="utf-8"))
    if mutation == "encoder":
        metadata["encoder"]["version"] = "wrong-version"
    elif mutation == "model":
        metadata["model"]["checkpoint"]["identifier"] = "wrong-checkpoint"
    elif mutation == "granularity":
        metadata["granularity"] = "residue"
    elif mutation == "outputs":
        metadata["outputs"] = list(reversed(metadata["outputs"]))
    elif mutation == "precision":
        metadata["provenance"]["environment"]["execution"]["resolved"]["precision"] = "fp16"
    else:
        del metadata["provenance"]["environment"]["execution"]
    entry.write_text(json.dumps(metadata), encoding="utf-8")

    runner.run(
        structure_input,
        config=_config(),
        model=model,
    )

    assert state.infer_calls == 2
    assert artifact_cache.set_calls == 2


def test_source_is_hashed_once_even_when_artifact_cache_is_disabled(tmp_path, monkeypatch):
    runner, state, artifact_cache, _, source, checkpoint = _runner_environment(tmp_path)
    real_sha256_file = runner_module.sha256_file
    calls = 0

    def counting_sha256_file(path):
        nonlocal calls
        calls += 1
        return real_sha256_file(path)

    monkeypatch.setattr(runner_module, "sha256_file", counting_sha256_file)

    runner.run(
        StructureInput(source=str(source)),
        config=_config(),
        model=_model(checkpoint),
        checkpoint_path=checkpoint,
        use_artifact_cache=False,
    )

    assert calls == 1
    assert artifact_cache.get_calls == artifact_cache.set_calls == 0
    assert state.infer_calls == 1


def test_parser_confirmed_format_must_match_parser_hint(tmp_path):
    runner, state, _, _, source, checkpoint = _runner_environment(
        tmp_path,
        confirmed_format="mmcif",
    )

    with pytest.raises(ExecutionError, match="incompatible.*mmcif"):
        runner.run(
            StructureInput(source=str(source)),
            config=_config(),
            model=_model(checkpoint),
            checkpoint_path=checkpoint,
        )

    assert state.preprocess_calls == 1
    assert state.load_calls == state.infer_calls == 0


def test_runner_rejects_config_incompatible_with_registry_type(tmp_path):
    runner, _, _, _, source, checkpoint = _runner_environment(
        tmp_path,
        config_type=SpecializedConfig,
    )

    with pytest.raises(ConfigError, match="SpecializedConfig"):
        runner.run(
            StructureInput(source=str(source)),
            config=_config(),
            model=_model(checkpoint),
            checkpoint_path=checkpoint,
        )


def test_runner_import_is_framework_independent_and_internal_only():
    script = """
import sys
import eris
import eris.core.learned
import eris.core.learned.runner

blocked = {'torch', 'torchdrug', 'torch_geometric', 'transformers'}
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise SystemExit(f'optional frameworks loaded: {loaded}')
assert not hasattr(eris, 'embed')
assert not hasattr(eris, 'StructuralEmbedderRunner')
"""

    subprocess.run([sys.executable, "-c", script], check=True)
    assert learned_registry.names() == ["gearnet_edge"]
