from __future__ import annotations

from pathlib import Path
from typing import Any

from eris.core.artifacts import StructureArtifact
from eris.core.config import CanonicalRepresentationConfig
from eris.core.registry import get_encoder
from eris.core.types import StructureInput
from eris.discovery import _get_explicit_registration
from eris.exceptions import ConfigError


class _NoOverride:
    def __repr__(self) -> str:
        return "<no override>"


_NO_OVERRIDE = _NoOverride()


def _validate_chain_id(value: object) -> None:
    if value is not None and not isinstance(value, str):
        raise ConfigError("chain_id must be a string or None")


def _validate_model_index(value: object) -> None:
    if type(value) is not int or value < 0:
        raise ConfigError("model_index must be a plain integer >= 0")


def _coerce_structure_input(
    input: str | Path | StructureInput,
    *,
    chain_id: str | None | _NoOverride,
    model_index: int | _NoOverride,
) -> StructureInput:
    if isinstance(input, StructureInput):
        if chain_id is not _NO_OVERRIDE:
            _validate_chain_id(chain_id)
            if chain_id != input.chain_id:
                raise ConfigError(
                    "chain_id override conflicts with StructureInput: "
                    f"{chain_id!r} != {input.chain_id!r}"
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
    _validate_chain_id(effective_chain)
    _validate_model_index(effective_model)
    return StructureInput(
        source=str(input),
        chain_id=effective_chain,
        model_index=effective_model,
    )


def _resolve_config(
    representation: str,
    *,
    config: CanonicalRepresentationConfig | None,
    params: dict[str, Any],
):
    info = _get_explicit_registration(representation)

    if config is not None:
        if params:
            names = ", ".join(sorted(params))
            raise ConfigError(f"config cannot be combined with representation parameters: {names}")
        if not isinstance(config, info.config_type):
            raise ConfigError(
                f"Config for {representation!r} must be an {info.config_type.__name__}"
            )
        return config

    allowed = {config_field.name for config_field in info.config_fields()}
    unknown = sorted(set(params) - allowed)
    if unknown:
        names = ", ".join(unknown)
        raise ConfigError(f"Unknown parameters for {representation!r}: {names}")

    try:
        return info.config_type(**params)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"Invalid parameters for {representation!r}: {error}") from error


def encode(
    input: str | Path | StructureInput,
    *,
    representation: str,
    chain_id: str | None | _NoOverride = _NO_OVERRIDE,
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


__all__ = ["encode"]
