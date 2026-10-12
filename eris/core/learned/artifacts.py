"""Artifact construction for learned embeddings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np

from eris.core.artifact_schema import (
    EncoderMetadata,
    InputMetadata,
    build_artifact_metadata,
    build_feature_keys,
    build_provenance,
)
from eris.core.artifacts import StructureArtifact
from eris.core.learned.checkpoints import StructuralModelDescriptor
from eris.core.learned.config import OutputMode, StructuralEmbedderConfig
from eris.core.learned.types import StructuralEmbeddingResult, validate_embedding_result

_GRANULARITY_BY_MODE = {
    "node": "residue",
    "global": "structure",
    "both": "mixed",
}

_OUTPUT_DECLARATIONS = {
    "node_embedding": {
        "name": "node_embedding",
        "role": "node_embedding",
        "granularity": "residue",
        "layout": "ND",
    },
    "global_embedding": {
        "name": "global_embedding",
        "role": "global_embedding",
        "granularity": "structure",
        "layout": "D",
    },
    "node_ids": {
        "name": "node_ids",
        "role": "node_ids",
        "granularity": "residue",
        "layout": "N",
    },
    "chain_id": {
        "name": "chain_id",
        "role": "chain_id",
        "granularity": "residue",
        "layout": "N",
    },
    "valid_mask": {
        "name": "valid_mask",
        "role": "valid_mask",
        "granularity": "residue",
        "layout": "N",
    },
}


def _ordered_data(
    result: StructuralEmbeddingResult,
    *,
    output_mode: OutputMode,
    chain_ids: list[str],
) -> dict[str, np.ndarray]:
    data: dict[str, np.ndarray] = {}

    if output_mode in {"node", "both"}:
        data["node_embedding"] = cast("np.ndarray", result.node_embedding)
    if output_mode in {"global", "both"}:
        data["global_embedding"] = cast("np.ndarray", result.global_embedding)
    if output_mode in {"node", "both"}:
        if len(chain_ids) != 1:
            raise ValueError("Learned residue outputs require exactly one selected chain in v0.1")
        data["node_ids"] = cast("np.ndarray", result.node_ids)
        data["chain_id"] = np.full(data["node_ids"].shape, chain_ids[0], dtype=np.str_)
        if result.valid_mask is not None:
            data["valid_mask"] = result.valid_mask

    return data


def build_learned_artifact(
    *,
    config: StructuralEmbedderConfig,
    result: StructuralEmbeddingResult,
    model: StructuralModelDescriptor,
    input_metadata: InputMetadata,
    encoder_metadata: EncoderMetadata,
    provenance_environment: Mapping[str, Any] | None = None,
) -> StructureArtifact:
    """Translate framework-independent learned outputs into a strict artifact schema artifact."""
    if not isinstance(config, StructuralEmbedderConfig):
        raise TypeError("config must be a StructuralEmbedderConfig")
    if not isinstance(model, StructuralModelDescriptor):
        raise TypeError("model must be a StructuralModelDescriptor")

    validate_embedding_result(result, output_mode=config.output_mode)

    data = _ordered_data(
        result,
        output_mode=config.output_mode,
        chain_ids=input_metadata["chain_ids"],
    )
    parameters = config.to_dict()
    counts = (
        {"n_nodes": int(cast("np.ndarray", result.node_embedding).shape[0])}
        if config.output_mode in {"node", "both"}
        else {}
    )
    if provenance_environment is not None and not isinstance(provenance_environment, Mapping):
        raise TypeError("provenance_environment must be a mapping or None")
    provenance = dict(
        build_provenance(
            representation_name=config.representation_name,
            parameters=parameters,
            chain_ids=input_metadata["chain_ids"],
        )
    )
    if provenance_environment is not None:
        provenance["environment"] = dict(provenance_environment)

    metadata = build_artifact_metadata(
        data=data,
        representation_name=config.representation_name,
        representation_family="learned",
        representation_type="embedding",
        granularity=_GRANULARITY_BY_MODE[config.output_mode],
        coordinate_source="model_specific",
        encoder=encoder_metadata,
        input_metadata=input_metadata,
        parameters=parameters,
        provenance=provenance,
        output_declarations=[_OUTPUT_DECLARATIONS[name] for name in data],
        feature_keys=build_feature_keys(),
        counts=counts,
        model=model.to_dict(),
    )
    artifact = StructureArtifact(data=data, meta=metadata)
    artifact.finalize()
    artifact.validate()
    return artifact


__all__ = ["build_learned_artifact"]
