from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from eris.core.artifact_schema_v2 import (
    build_input_metadata,
    compute_config_fingerprint,
)
from eris.core.cache import DiskCache, compute_cache_key_v2
from eris.core.learned.artifacts import build_learned_artifact
from eris.core.learned.backends.gearnet import (
    GEARNET_EDGE_BACKEND_ID,
    GEARNET_EDGE_CHECKPOINT_DESCRIPTOR,
    GEARNET_EDGE_ENCODER_VERSION,
    GEARNET_EDGE_MODEL_DESCRIPTOR,
    GEARNET_EDGE_REPRESENTATION_NAME,
    GearNetEdgeConfig,
    register_gearnet_edge,
)
from eris.core.learned.backends.gearnet.metadata import (
    GEARNET_EDGE_CANONICAL_RESIDUES,
    GEARNET_EDGE_IMPLEMENTATION,
)
from eris.core.learned.checkpoint_cache import CheckpointCache
from eris.core.learned.checkpoints import CheckpointDescriptor, StructuralModelDescriptor
from eris.core.learned.execution import (
    ResolvedExecution,
    StructuralExecutionConfig,
    build_execution_provenance,
    execution_request_cache_identity,
)
from eris.core.learned.registry import BackendAvailability, EmbedderRegistry, learned_registry
from eris.core.learned.runner import (
    StructuralEmbedderRunner,
    _matches_cache_identity,
    _refresh_input_source,
)
from eris.core.learned.types import StructuralEmbeddingResult
from eris.core.types import StructureInput
from eris.exceptions import ExecutionError

EXPECTED_CHECKPOINT_SHA256 = "6dc11e2db6eff3c90f90672af227c647125adb7476de18a2bb860457bdb598ec"


def _config(output_mode: str = "both") -> GearNetEdgeConfig:
    return GearNetEdgeConfig(
        representation_name=GEARNET_EDGE_REPRESENTATION_NAME,
        output_mode=output_mode,
    )


def _result(output_mode: str) -> StructuralEmbeddingResult:
    node_embedding = np.arange(3 * 3072, dtype=np.float32).reshape(3, 3072)
    global_embedding = np.arange(3072, dtype=np.float32)
    node_ids = np.asarray(["A:1::ALA", "A:2::GLY", "A:3::SER"], dtype=np.str_)
    return StructuralEmbeddingResult(
        node_embedding=node_embedding if output_mode != "global" else None,
        global_embedding=global_embedding if output_mode != "node" else None,
        node_ids=node_ids if output_mode != "global" else None,
        valid_mask=None,
    )


def _input_metadata(source: str = "C:/structures/protein.pdb"):
    return build_input_metadata(
        source=source,
        source_sha256="a" * 64,
        input_format="pdb",
        model_index=2,
        chain_id="A",
    )


def _artifact(
    output_mode: str = "both",
    *,
    source: str = "C:/structures/protein.pdb",
    encoder_version: str = GEARNET_EDGE_ENCODER_VERSION,
):
    request = StructuralExecutionConfig(device="cpu", precision="fp32")
    resolved = ResolvedExecution(device="cpu", precision="fp32")
    return build_learned_artifact(
        config=_config(output_mode),
        result=_result(output_mode),
        model=GEARNET_EDGE_MODEL_DESCRIPTOR,
        input_metadata=_input_metadata(source),
        encoder_metadata={
            "implementation": GEARNET_EDGE_IMPLEMENTATION,
            "version": encoder_version,
        },
        provenance_environment={
            "execution": build_execution_provenance(request, resolved),
        },
    )


def _cache_key(
    *,
    config: GearNetEdgeConfig | None = None,
    model: StructuralModelDescriptor = GEARNET_EDGE_MODEL_DESCRIPTOR,
    encoder_version: str = GEARNET_EDGE_ENCODER_VERSION,
    implementation: str = GEARNET_EDGE_IMPLEMENTATION,
    precision: str = "fp32",
    fingerprint: str | None = None,
) -> str:
    config = config or _config()
    return compute_cache_key_v2(
        representation_name=config.representation_name,
        encoder_implementation=implementation,
        encoder_version=encoder_version,
        config_fingerprint=fingerprint
        or compute_config_fingerprint(config.representation_name, config.to_dict()),
        source_sha256="a" * 64,
        parser_hint=".pdb",
        model_index=2,
        chain_id="A",
        model=model.to_cache_identity(),
        execution={"precision": precision},
    )


def test_frozen_gearnet_model_checkpoint_and_adapter_identity():
    assert GEARNET_EDGE_REPRESENTATION_NAME == "gearnet_edge"
    assert GEARNET_EDGE_BACKEND_ID == "gearnet-edge"
    assert GEARNET_EDGE_ENCODER_VERSION == "2"
    assert GEARNET_EDGE_IMPLEMENTATION == (
        "eris.core.learned.backends.gearnet.backend.GearNetEdgeBackend"
    )
    assert GEARNET_EDGE_MODEL_DESCRIPTOR.to_dict() == {
        "architecture": "GearNet-Edge",
        "checkpoint": {
            "identifier": "mc_gearnet_edge.pth",
            "sha256": EXPECTED_CHECKPOINT_SHA256,
            "revision": "Zenodo record v2",
            "source": "https://zenodo.org/records/7723075",
        },
    }
    assert GEARNET_EDGE_CHECKPOINT_DESCRIPTOR is GEARNET_EDGE_MODEL_DESCRIPTOR.checkpoint
    assert "path" not in GEARNET_EDGE_MODEL_DESCRIPTOR.to_dict()["checkpoint"]


def test_gearnet_config_records_exact_frozen_preprocessing_without_locators():
    parameters = _config().to_dict()

    assert parameters["representation_name"] == "gearnet_edge"
    assert parameters["output_mode"] == "both"
    biological = parameters["preprocessing"]["biological_selection"]
    assert biological == {
        "policy": "canonical_20_only",
        "residue_names": list(GEARNET_EDGE_CANONICAL_RESIDUES),
        "coordinate_source": "ca",
        "nonstandard_residues": "exclude",
        "mse": "exclude",
        "hetero": "exclude",
        "water": "exclude",
        "ligands": "exclude",
        "metals": "exclude",
        "ions": "exclude",
        "missing_retained_ca": "fail",
        "altloc": "first",
    }
    assert parameters["preprocessing"]["graph_construction"] == {
        "node_layer": "alpha_carbon",
        "sequential_edge": {"max_distance": 2},
        "spatial_edge": {
            "radius": 10.0,
            "min_distance": 5,
            "max_distance": None,
            "max_num_neighbors": 32,
        },
        "knn_edge": {"k": 10, "min_distance": 5, "max_distance": None},
        "edge_feature": "gearnet",
        "num_relation": 7,
        "node_feature_width": 21,
        "edge_feature_width": 59,
    }
    serialized = json.dumps(parameters)
    assert all(
        forbidden not in serialized
        for forbidden in ("checkpoint_path", "temporary", "torch_ext", "cache_path")
    )


def test_registration_metadata_is_explicit_side_effect_free_and_canonical():
    registry = EmbedderRegistry()

    info = register_gearnet_edge(registry)

    assert learned_registry.names() == ["gearnet_edge"]
    assert registry.names() == ["gearnet_edge"]
    assert info.representation_name == GEARNET_EDGE_REPRESENTATION_NAME
    assert info.backend_id == GEARNET_EDGE_BACKEND_ID
    assert info.implementation == GEARNET_EDGE_IMPLEMENTATION
    assert info.encoder_version == GEARNET_EDGE_ENCODER_VERSION
    assert info.config_type is GearNetEdgeConfig
    assert registry.create_config("gearnet_edge", output_mode="node") == _config("node")

    production_info = learned_registry.get_info("gearnet_edge")
    assert production_info.representation_name == GEARNET_EDGE_REPRESENTATION_NAME
    assert production_info.backend_id == GEARNET_EDGE_BACKEND_ID
    assert production_info.implementation == GEARNET_EDGE_IMPLEMENTATION
    assert production_info.encoder_version == GEARNET_EDGE_ENCODER_VERSION
    assert production_info.config_type is GearNetEdgeConfig
    with pytest.raises(KeyError, match="already registered"):
        register_gearnet_edge(learned_registry)


def test_gearnet_metadata_and_registration_imports_are_framework_lazy():
    script = """
import sys
import eris
from eris.core.learned import learned_registry

assert learned_registry.names() == ['gearnet_edge']
info = learned_registry.get_info('gearnet_edge')
assert info.backend_id == 'gearnet-edge'
assert info.implementation == 'eris.core.learned.backends.gearnet.backend.GearNetEdgeBackend'
assert info.encoder_version == '2'
blocked = {'torch', 'torchdrug', 'torch_scatter', 'torch_cluster', 'rdkit'}
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise SystemExit(f'optional frameworks loaded: {loaded}')
"""

    subprocess.run([sys.executable, "-c", script], check=True)


def test_production_runner_reports_unavailable_before_factory_or_checkpoint(
    tmp_path,
    monkeypatch,
):
    from eris.core.learned.backends.gearnet import dependencies

    source = tmp_path / "input.pdb"
    source.write_text("END\n", encoding="ascii")
    artifact_cache = tmp_path / "artifact-cache"
    checkpoint_cache = tmp_path / "checkpoint-cache"
    monkeypatch.setattr(
        dependencies,
        "probe_gearnet_availability",
        lambda: BackendAvailability(
            available=False,
            reason="Install Eris GearNet support with compatible optional dependencies",
        ),
    )

    runner = StructuralEmbedderRunner(
        artifact_cache=DiskCache(artifact_cache),
        checkpoint_cache=CheckpointCache(checkpoint_cache),
    )
    with pytest.raises(ExecutionError, match="unavailable.*optional dependencies"):
        runner.run(
            StructureInput(source=str(source), chain_id="A", model_index=0),
            config=_config(),
            model=GEARNET_EDGE_MODEL_DESCRIPTOR,
            checkpoint_path=Path("not-accessed.pth"),
            execution=StructuralExecutionConfig(device="cpu", precision="fp32"),
        )

    assert list((artifact_cache / "v2").glob("*")) == []
    assert not checkpoint_cache.exists()


@pytest.mark.parametrize(
    ("output_mode", "expected_order", "granularity", "shapes", "counts"),
    [
        (
            "node",
            ["node_embedding", "node_ids"],
            "residue",
            {"node_embedding": [3, 3072], "node_ids": [3]},
            {"n_nodes": 3},
        ),
        (
            "global",
            ["global_embedding"],
            "structure",
            {"global_embedding": [3072]},
            {},
        ),
        (
            "both",
            ["node_embedding", "global_embedding", "node_ids"],
            "mixed",
            {
                "node_embedding": [3, 3072],
                "global_embedding": [3072],
                "node_ids": [3],
            },
            {"n_nodes": 3},
        ),
    ],
)
def test_gearnet_artifact_metadata_is_complete_by_output_mode(
    output_mode,
    expected_order,
    granularity,
    shapes,
    counts,
):
    artifact = _artifact(output_mode)

    artifact.validate(strict=True)
    assert artifact.meta["schema_version"] == 2
    assert artifact.meta["representation_name"] == "gearnet_edge"
    assert artifact.meta["representation_family"] == "learned"
    assert artifact.meta["representation_type"] == "embedding"
    assert artifact.meta["granularity"] == granularity
    assert artifact.meta["coordinate_source"] == "model_specific"
    assert artifact.meta["encoder"] == {
        "implementation": GEARNET_EDGE_IMPLEMENTATION,
        "version": "2",
    }
    assert artifact.meta["model"] == GEARNET_EDGE_MODEL_DESCRIPTOR.to_dict()
    assert artifact.meta["input"]["model_index"] == 2
    assert artifact.meta["input"]["chain_id"] == "A"
    assert list(artifact.data) == expected_order
    assert [item["name"] for item in artifact.meta["outputs"]] == expected_order
    assert artifact.meta["feature_keys"] == {}
    assert artifact.meta["shapes"] == shapes
    assert artifact.meta["dtypes"] == {
        name: str(artifact.data[name].dtype) for name in expected_order
    }
    assert artifact.meta["counts"] == counts
    assert "valid_mask" not in artifact.data
    assert artifact.meta["provenance"]["environment"]["execution"] == {
        "requested": {"device": "cpu", "precision": "fp32"},
        "resolved": {"device": "cpu", "precision": "fp32"},
    }


def test_gearnet_cache_identity_includes_scientific_contract_dimensions():
    baseline = _cache_key()
    parameters = _config().to_dict()
    changed_parameters = copy.deepcopy(parameters)
    changed_parameters["preprocessing"]["graph_construction"]["knn_edge"]["k"] = 11
    changed_fingerprint = compute_config_fingerprint(
        GEARNET_EDGE_REPRESENTATION_NAME,
        changed_parameters,
    )
    changed_checkpoint = StructuralModelDescriptor(
        architecture="GearNet-Edge",
        checkpoint=CheckpointDescriptor(
            identifier="other-checkpoint.pth",
            sha256="b" * 64,
            revision="other-revision",
            source="https://mirror.invalid/other",
        ),
    )

    assert baseline != _cache_key(fingerprint=changed_fingerprint)
    assert baseline != _cache_key(model=changed_checkpoint)
    assert baseline != _cache_key(encoder_version="1")
    assert baseline != _cache_key(
        implementation="eris.core.learned.backends.gearnet.backend.OtherAdapter"
    )
    assert baseline != _cache_key(precision="fp16")


def test_gearnet_cache_identity_excludes_operational_locators_and_device():
    baseline = _cache_key()
    mirrored_model = StructuralModelDescriptor(
        architecture="GearNet-Edge",
        checkpoint=CheckpointDescriptor(
            identifier="mc_gearnet_edge.pth",
            sha256=EXPECTED_CHECKPOINT_SHA256,
            revision="Zenodo record v2",
            source="https://mirror.example/mc_gearnet_edge.pth",
        ),
    )

    assert GEARNET_EDGE_MODEL_DESCRIPTOR.to_cache_identity() == mirrored_model.to_cache_identity()
    assert baseline == _cache_key(model=mirrored_model)
    assert execution_request_cache_identity(
        StructuralExecutionConfig(device="cpu", precision="fp32")
    ) == execution_request_cache_identity(
        StructuralExecutionConfig(device="auto", precision="fp32")
    )
    assert "source" not in GEARNET_EDGE_MODEL_DESCRIPTOR.to_cache_identity()["checkpoint"]
    assert "path" not in GEARNET_EDGE_MODEL_DESCRIPTOR.to_cache_identity()["checkpoint"]


def test_gearnet_cache_hit_refreshes_only_input_locator_and_preserves_generation_provenance():
    config = _config()
    artifact = _artifact(source="C:/locator-a/protein.pdb")
    registry = EmbedderRegistry()
    info = register_gearnet_edge(registry)
    parameters = config.to_dict()
    fingerprint = compute_config_fingerprint(config.representation_name, parameters)
    relocated = StructureInput(
        source="D:/locator-b/protein.pdb",
        model_index=2,
        chain_id="A",
    )

    assert _matches_cache_identity(
        artifact,
        structure_input=relocated,
        info=info,
        config=config,
        model=GEARNET_EDGE_MODEL_DESCRIPTOR,
        source_sha256="a" * 64,
        parser_hint=".pdb",
        parameters=parameters,
        config_fingerprint=fingerprint,
        execution_request=StructuralExecutionConfig(device="cpu", precision="fp32"),
    )
    refreshed = _refresh_input_source(
        artifact,
        structure_input=relocated,
        source_sha256="a" * 64,
    )

    assert refreshed.meta["input"]["source"] == "D:/locator-b/protein.pdb"
    assert refreshed.meta["input"]["source_sha256"] == artifact.meta["input"]["source_sha256"]
    assert refreshed.meta["provenance"] == artifact.meta["provenance"]
    assert refreshed.meta["model"] == artifact.meta["model"]
    assert refreshed.data is artifact.data


def test_legacy_encoder_version_cannot_satisfy_native_cache_identity():
    config = _config()
    artifact = _artifact(encoder_version="1")
    registry = EmbedderRegistry()
    info = register_gearnet_edge(registry)
    parameters = config.to_dict()
    fingerprint = compute_config_fingerprint(config.representation_name, parameters)

    assert info.encoder_version == "2"
    assert not _matches_cache_identity(
        artifact,
        structure_input=StructureInput(
            source="C:/structures/protein.pdb", model_index=2, chain_id="A"
        ),
        info=info,
        config=config,
        model=GEARNET_EDGE_MODEL_DESCRIPTOR,
        source_sha256="a" * 64,
        parser_hint=".pdb",
        parameters=parameters,
        config_fingerprint=fingerprint,
        execution_request=StructuralExecutionConfig(device="cpu", precision="fp32"),
    )
