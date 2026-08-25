from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from eris.core.artifact_schema_v2 import build_input_metadata, compute_config_fingerprint
from eris.core.artifacts import ArtifactValidationError, StructureArtifact
from eris.core.cache import (
    DiskCache,
    compute_cache_key_v2,
    parser_hint_from_source,
    parser_hint_matches_format,
)
from eris.core.hashing import sha256_file
from eris.core.learned.artifacts import build_learned_artifact
from eris.core.learned.checkpoint_cache import CheckpointCache
from eris.core.learned.checkpoints import StructuralModelDescriptor, VerifiedCheckpoint
from eris.core.learned.config import StructuralEmbedderConfig
from eris.core.learned.execution import (
    ResolvedExecution,
    StructuralExecutionConfig,
    build_execution_provenance,
    execution_cache_identity,
    execution_request_cache_identity,
    resolve_execution,
)
from eris.core.learned.registry import EmbedderInfo, EmbedderRegistry, learned_registry
from eris.core.learned.types import (
    StructuralPreprocessingResult,
    validate_embedding_result,
)
from eris.core.types import StructureInput
from eris.exceptions import ConfigError, ExecutionError

_GRANULARITY_BY_MODE = {
    "node": "residue",
    "global": "structure",
    "both": "mixed",
}

_OUTPUT_ORDERS = {
    "node": (
        ("node_embedding", "node_ids"),
        ("node_embedding", "node_ids", "valid_mask"),
    ),
    "global": (("global_embedding",),),
    "both": (
        ("node_embedding", "global_embedding", "node_ids"),
        ("node_embedding", "global_embedding", "node_ids", "valid_mask"),
    ),
}


def _cached_model_identity(model_metadata: object) -> dict[str, object] | None:
    if not isinstance(model_metadata, dict):
        return None
    architecture = model_metadata.get("architecture")
    checkpoint = model_metadata.get("checkpoint")
    if not isinstance(architecture, str) or not isinstance(checkpoint, dict):
        return None
    identifier = checkpoint.get("identifier")
    sha256 = checkpoint.get("sha256")
    if not isinstance(identifier, str) or not isinstance(sha256, str):
        return None

    checkpoint_identity: dict[str, str] = {"identifier": identifier}
    revision = checkpoint.get("revision")
    if revision is not None:
        if not isinstance(revision, str):
            return None
        checkpoint_identity["revision"] = revision
    checkpoint_identity["sha256"] = sha256
    return {
        "architecture": architecture,
        "checkpoint": checkpoint_identity,
    }


def _cached_execution_matches(
    provenance: object,
    *,
    requested_precision: str,
) -> bool:
    if not isinstance(provenance, dict):
        return False
    environment = provenance.get("environment")
    if not isinstance(environment, dict):
        return False
    execution = environment.get("execution")
    if not isinstance(execution, dict) or set(execution) != {"requested", "resolved"}:
        return False
    requested = execution["requested"]
    resolved = execution["resolved"]
    if not isinstance(requested, dict) or not isinstance(resolved, dict):
        return False
    try:
        cached_request = StructuralExecutionConfig(**requested)
        cached_resolved = ResolvedExecution(**resolved)
    except (ConfigError, TypeError):
        return False
    return cached_request.precision == cached_resolved.precision == requested_precision


def _matches_cache_identity(
    artifact: StructureArtifact,
    *,
    structure_input: StructureInput,
    info: EmbedderInfo,
    config: StructuralEmbedderConfig,
    model: StructuralModelDescriptor,
    source_sha256: str,
    parser_hint: str,
    parameters: dict[str, Any],
    config_fingerprint: str,
    execution_request: StructuralExecutionConfig,
) -> bool:
    try:
        artifact.validate(strict=True)
    except (ArtifactValidationError, TypeError, ValueError):
        return False

    meta = artifact.meta
    encoder = meta.get("encoder")
    input_metadata = meta.get("input")
    provenance = meta.get("provenance")
    outputs = meta.get("outputs")
    if not all(
        isinstance(value, dict) for value in (encoder, input_metadata, provenance)
    ) or not isinstance(outputs, list):
        return False

    output_names = tuple(output.get("name") for output in outputs if isinstance(output, dict))
    if len(output_names) != len(outputs):
        return False
    if output_names not in _OUTPUT_ORDERS[config.output_mode]:
        return False
    if tuple(artifact.data) != output_names:
        return False

    expected_counts = (
        {"n_nodes": int(artifact.data["node_embedding"].shape[0])}
        if config.output_mode in {"node", "both"}
        else {}
    )
    return (
        type(meta.get("schema_version")) is int
        and meta["schema_version"] == 2
        and meta.get("representation_name") == config.representation_name
        and meta.get("representation_family") == "learned"
        and meta.get("representation_type") == "embedding"
        and meta.get("granularity") == _GRANULARITY_BY_MODE[config.output_mode]
        and meta.get("coordinate_source") == "model_specific"
        and meta.get("parameters") == parameters
        and meta.get("feature_keys") == {}
        and meta.get("counts") == expected_counts
        and encoder.get("implementation") == info.implementation
        and encoder.get("version") == info.encoder_version
        and provenance.get("config_fingerprint") == config_fingerprint
        and input_metadata.get("source_sha256") == source_sha256
        and input_metadata.get("model_index") == structure_input.model_index
        and input_metadata.get("chain_id") == structure_input.chain_id
        and isinstance(input_metadata.get("format"), str)
        and parser_hint_matches_format(parser_hint, input_metadata["format"])
        and _cached_model_identity(meta.get("model")) == model.to_cache_identity()
        and _cached_execution_matches(
            provenance,
            requested_precision=execution_request.precision,
        )
    )


def _refresh_input_source(
    artifact: StructureArtifact,
    *,
    structure_input: StructureInput,
    source_sha256: str,
) -> StructureArtifact:
    metadata = deepcopy(artifact.meta)
    metadata["input"] = build_input_metadata(
        source=structure_input.source,
        source_sha256=source_sha256,
        input_format=artifact.meta["input"]["format"],
        model_index=structure_input.model_index,
        chain_id=structure_input.chain_id,
    )
    refreshed = StructureArtifact(data=artifact.data, meta=metadata)
    refreshed.validate(strict=True)
    return refreshed


class StructuralEmbedderRunner:
    """Orchestrate one internal framework-independent learned embedding run."""

    def __init__(
        self,
        *,
        registry: EmbedderRegistry = learned_registry,
        artifact_cache: DiskCache | None = None,
        checkpoint_cache: CheckpointCache | None = None,
    ) -> None:
        self.registry = registry
        self.artifact_cache = artifact_cache
        self.checkpoint_cache = checkpoint_cache

    def _artifact_cache(self) -> DiskCache:
        if self.artifact_cache is None:
            self.artifact_cache = DiskCache()
        return self.artifact_cache

    def _checkpoint_cache(self) -> CheckpointCache:
        if self.checkpoint_cache is None:
            self.checkpoint_cache = CheckpointCache()
        return self.checkpoint_cache

    def _resolve_checkpoint(
        self,
        model: StructuralModelDescriptor,
        checkpoint_path: str | Path | None,
    ) -> VerifiedCheckpoint:
        cache = self._checkpoint_cache()
        checkpoint = cache.get(model.checkpoint)
        if checkpoint is not None:
            return checkpoint
        if checkpoint_path is None:
            raise ExecutionError(
                f"No cached checkpoint found for '{model.checkpoint.identifier}'; "
                "a local checkpoint_path is required"
            )
        return cache.put(checkpoint_path, model.checkpoint)

    def run(
        self,
        structure_input: StructureInput,
        *,
        config: StructuralEmbedderConfig,
        model: StructuralModelDescriptor,
        checkpoint_path: str | Path | None = None,
        execution: StructuralExecutionConfig | None = None,
        use_artifact_cache: bool = True,
    ) -> StructureArtifact:
        if not isinstance(structure_input, StructureInput):
            raise TypeError("structure_input must be a StructureInput")
        if not isinstance(config, StructuralEmbedderConfig):
            raise TypeError("config must be a StructuralEmbedderConfig")
        if not isinstance(model, StructuralModelDescriptor):
            raise TypeError("model must be a StructuralModelDescriptor")
        if checkpoint_path is not None and not isinstance(checkpoint_path, (str, Path)):
            raise TypeError("checkpoint_path must be a string, pathlib.Path, or None")
        if execution is not None and not isinstance(execution, StructuralExecutionConfig):
            raise TypeError("execution must be a StructuralExecutionConfig or None")
        if type(use_artifact_cache) is not bool:
            raise TypeError("use_artifact_cache must be a bool")

        info = self.registry.get_info(config.representation_name)
        if not isinstance(config, info.config_type):
            raise ConfigError(
                f"Config for '{config.representation_name}' must be an {info.config_type.__name__}"
            )

        execution_request = execution or StructuralExecutionConfig()
        source_sha256 = sha256_file(Path(structure_input.source))
        parameters = config.to_dict()
        config_fingerprint = compute_config_fingerprint(
            config.representation_name,
            parameters,
        )
        parser_hint = parser_hint_from_source(structure_input.source)
        request_execution_identity = execution_request_cache_identity(execution_request)
        cache_key = compute_cache_key_v2(
            representation_name=config.representation_name,
            encoder_implementation=info.implementation,
            encoder_version=info.encoder_version,
            config_fingerprint=config_fingerprint,
            source_sha256=source_sha256,
            parser_hint=parser_hint,
            model_index=structure_input.model_index,
            chain_id=structure_input.chain_id,
            model=model.to_cache_identity(),
            execution=request_execution_identity,
        )

        artifact_cache = self._artifact_cache() if use_artifact_cache else None
        if artifact_cache is not None:
            cached = artifact_cache.get_v2(cache_key)
            if cached is not None and _matches_cache_identity(
                cached,
                structure_input=structure_input,
                info=info,
                config=config,
                model=model,
                source_sha256=source_sha256,
                parser_hint=parser_hint,
                parameters=parameters,
                config_fingerprint=config_fingerprint,
                execution_request=execution_request,
            ):
                return _refresh_input_source(
                    cached,
                    structure_input=structure_input,
                    source_sha256=source_sha256,
                )

        backend = self.registry.create(config.representation_name)
        resolved_execution = resolve_execution(
            execution_request,
            backend.execution_capabilities(),
        )
        if execution_cache_identity(resolved_execution) != request_execution_identity:
            raise ExecutionError(
                "Resolved execution precision does not match the artifact cache identity"
            )

        verified_checkpoint = self._resolve_checkpoint(model, checkpoint_path)
        preprocessed = backend.preprocess(structure_input, config=config)
        if not isinstance(preprocessed, StructuralPreprocessingResult):
            raise ExecutionError("Backend preprocess() must return StructuralPreprocessingResult")
        if not parser_hint_matches_format(parser_hint, preprocessed.input_format):
            raise ExecutionError(
                f"Parser hint {parser_hint!r} is incompatible with backend-confirmed "
                f"input format {preprocessed.input_format!r}"
            )

        input_metadata = build_input_metadata(
            source=structure_input.source,
            source_sha256=source_sha256,
            input_format=preprocessed.input_format,
            model_index=structure_input.model_index,
            chain_id=structure_input.chain_id,
        )
        model_instance = backend.load(
            verified_checkpoint,
            execution=resolved_execution,
        )
        result = backend.infer(
            model_instance,
            preprocessed.prepared,
            config=config,
            execution=resolved_execution,
        )
        validate_embedding_result(result, output_mode=config.output_mode)
        artifact = build_learned_artifact(
            config=config,
            result=result,
            model=model,
            input_metadata=input_metadata,
            encoder_metadata={
                "implementation": info.implementation,
                "version": info.encoder_version,
            },
            provenance_environment={
                "execution": build_execution_provenance(
                    execution_request,
                    resolved_execution,
                )
            },
        )
        artifact.validate(strict=True)

        if artifact_cache is not None:
            artifact_cache.set_v2(cache_key, artifact)
        return artifact


__all__ = ["StructuralEmbedderRunner"]
