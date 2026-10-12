"""Public encode and embed entry points."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from eris.core.learned import StructuralEmbedderRunner, StructuralExecutionConfig
from eris.core.learned.registry import learned_registry
from eris.core.registry import (
    get_encoder,
    resolve_representation_config,
    validate_resolved_representation_config,
)
from eris.core.types import StructureInput
from eris.exceptions import ConfigError, RepresentationNotFoundError
from eris.pooling import pool

if TYPE_CHECKING:
    from eris.core.artifacts import StructureArtifact
    from eris.core.config import CanonicalRepresentationConfig
    from eris.core.learned.config import OutputMode
    from eris.core.learned.execution import Precision


class _NoOverride:
    """Sentinel type for an argument that was not overridden."""

    def __repr__(self) -> str:
        return "<no override>"


_NO_OVERRIDE = _NoOverride()


def _normalize_chain_id(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        normalized = tuple(value)
    else:
        raise ConfigError("chain_id must be None, a string, or a sequence of strings")
    if not normalized:
        raise ConfigError("chain_id sequence must not be empty")
    if any(not isinstance(chain, str) for chain in normalized):
        raise ConfigError("chain_id sequence entries must be strings")
    if len(normalized) != len(set(normalized)):
        raise ConfigError("chain_id sequence must not contain duplicates")
    return normalized


def _validate_model_index(value: object) -> None:
    if type(value) is not int or value < 0:
        raise ConfigError("model_index must be a plain integer >= 0")


def _coerce_structure_input(
    input: str | Path | StructureInput,  # noqa: A002
    *,
    chain_id: str | Sequence[str] | _NoOverride | None,
    model_index: int | _NoOverride,
) -> StructureInput:
    if isinstance(input, StructureInput):
        if chain_id is not _NO_OVERRIDE:
            normalized_chain_id = _normalize_chain_id(chain_id)
            if normalized_chain_id != input.chain_id:
                raise ConfigError(
                    f"chain_id override conflicts with StructureInput: {chain_id!r} != {input.chain_id!r}"
                )
        if model_index is not _NO_OVERRIDE:
            _validate_model_index(model_index)
            if model_index != input.model_index:
                raise ConfigError(
                    "model_index override conflicts with StructureInput: "
                    f"{model_index!r} != {input.model_index!r}"
                )
        return input

    if not isinstance(input, (str, Path)):
        raise TypeError("input must be a string, pathlib.Path, or StructureInput")

    effective_chain = None if chain_id is _NO_OVERRIDE else chain_id
    effective_model = 0 if model_index is _NO_OVERRIDE else model_index
    effective_chain = _normalize_chain_id(effective_chain)
    _validate_model_index(effective_model)
    return StructureInput(
        source=str(input),
        chain_id=effective_chain,
        model_index=cast("int", effective_model),
    )


def _resolve_config(
    representation: str,
    *,
    config: CanonicalRepresentationConfig | None,
    params: dict[str, Any],
) -> CanonicalRepresentationConfig:
    if config is not None:
        return validate_resolved_representation_config(
            representation,
            config=config,
            params=params,
        )
    return resolve_representation_config(
        representation,
        params=params,
    )


def encode(
    input: str | Path | StructureInput,  # noqa: A002
    *,
    representation: str,
    chain_id: str | Sequence[str] | _NoOverride | None = _NO_OVERRIDE,
    model_index: int | _NoOverride = _NO_OVERRIDE,
    config: CanonicalRepresentationConfig | None = None,
    output: str | Path | None = None,
    use_cache: bool = True,
    **params: Any,
) -> StructureArtifact:
    """Encode one structure with a canonical explicit representation."""
    if type(use_cache) is not bool:
        raise ConfigError("use_cache must be a bool")
    if output is not None and not isinstance(output, (str, Path)):
        raise TypeError("output must be a string, pathlib.Path, or None")

    resolved_config = _resolve_config(
        representation,
        config=config,
        params=params,
    )
    structure_input = _coerce_structure_input(
        input,
        chain_id=chain_id,
        model_index=model_index,
    )
    encoder = get_encoder(
        representation,
        config=resolved_config,
        use_cache=use_cache,
    )
    artifact = encoder.encode(structure_input)
    if output is not None:
        artifact.save(Path(output))
    return artifact


def embed(
    input: str | Path | StructureInput,  # noqa: A002
    *,
    representation: str,
    chain_id: str | Sequence[str] | _NoOverride | None = _NO_OVERRIDE,
    model_index: int | _NoOverride = _NO_OVERRIDE,
    output_mode: str = "both",
    checkpoint_path: str | Path | None = None,
    device: str = "auto",
    precision: str = "fp32",
    output: str | Path | None = None,
    use_cache: bool = True,
) -> StructureArtifact:
    """Embed one structure with a registered learned representation."""
    try:
        info = learned_registry.get_info(representation)
    except KeyError as error:
        raise RepresentationNotFoundError(f"Unknown embedder {representation!r}") from error

    if type(use_cache) is not bool:
        raise ConfigError("use_cache must be a bool")
    if output is not None and not isinstance(output, (str, Path)):
        raise TypeError("output must be a string, pathlib.Path, or None")
    if checkpoint_path is not None and not isinstance(checkpoint_path, (str, Path)):
        raise TypeError("checkpoint_path must be a string, pathlib.Path, or None")

    structure_input = _coerce_structure_input(
        input,
        chain_id=chain_id,
        model_index=model_index,
    )
    config = info.config_type(
        representation_name=info.representation_name,
        output_mode=cast("OutputMode", output_mode),
    )
    execution = StructuralExecutionConfig(device=device, precision=cast("Precision", precision))

    artifact = StructuralEmbedderRunner().run(
        structure_input,
        config=config,
        model=info.model_descriptor,
        checkpoint_path=checkpoint_path,
        execution=execution,
        use_artifact_cache=use_cache,
    )
    if output is not None:
        artifact.save(Path(output))
    return artifact


__all__ = ["embed", "encode", "pool"]
