from __future__ import annotations

import types
from collections.abc import Mapping
from dataclasses import MISSING, Field, dataclass, fields
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from eris.core.base_encoder.base_encoder import BaseEncoder
from eris.core.config import (
    _CONFIG_TYPES_BY_REPRESENTATION,
    CanonicalRepresentationConfig,
)
from eris.exceptions import ConfigError, RepresentationNotFoundError


def _render_literal(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    return repr(value)


def _render_type(annotation: Any) -> str:
    """Render supported config annotations as stable, human-readable strings."""
    if annotation is Any:
        return "any"
    if annotation is type(None):
        return "null"
    if annotation in {bool, int, float, str}:
        return annotation.__name__

    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is Literal:
        return " | ".join(_render_literal(value) for value in arguments)
    if origin in {types.UnionType, Union}:
        return " | ".join(_render_type(argument) for argument in arguments)
    if origin is list:
        item_type = _render_type(arguments[0]) if arguments else "any"
        return f"list[{item_type}]"
    if origin is tuple:
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return f"list[{_render_type(arguments[0])}]"
        return f"list[{', '.join(_render_type(argument) for argument in arguments)}]"
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation).removeprefix("typing.")


def _is_nullable(annotation: Any) -> bool:
    if annotation is type(None):
        return True
    origin = get_origin(annotation)
    return origin in {types.UnionType, Union} and type(None) in get_args(annotation)


def _literal_choices(annotation: Any) -> list[Any] | None:
    if get_origin(annotation) is Literal:
        return list(get_args(annotation))
    return None


@dataclass(frozen=True)
class RepresentationInfo:
    config_type: type[CanonicalRepresentationConfig]
    encoder_type: type[BaseEncoder]
    description: str = ""

    @property
    def name(self) -> str:
        return self.config_type.representation_name

    @property
    def family(self) -> str:
        return "explicit"

    @property
    def representation_type(self) -> str:
        return self.encoder_type.representation.representation_type

    @property
    def granularity(self) -> str:
        return self.config_type.granularity

    @property
    def default_coordinate_source(self) -> str:
        return self.default_config().coordinate_source

    def default_config(self) -> CanonicalRepresentationConfig:
        return self.config_type()

    def defaults(self) -> dict[str, Any]:
        return self.default_config().to_dict()

    def config_fields(self) -> tuple[Field, ...]:
        return fields(self.config_type)

    def params_schema(self) -> list[dict[str, Any]]:
        """Describe config fields for help/introspection, not config validation.

        Typed configs remain the sole authority for constraints and relationships that
        are not represented directly by annotations or dataclass field metadata.
        """
        defaults = self.default_config().to_dict()
        type_hints = get_type_hints(self.config_type)
        schema: list[dict[str, Any]] = []

        for config_field in self.config_fields():
            annotation = type_hints[config_field.name]
            entry: dict[str, Any] = {
                "name": config_field.name,
                "type": _render_type(annotation),
                "default": defaults[config_field.name],
                "required": (
                    config_field.default is MISSING and config_field.default_factory is MISSING
                ),
                "nullable": _is_nullable(annotation),
                "description": config_field.metadata.get("description"),
            }

            if "unit" in config_field.metadata:
                entry["unit"] = config_field.metadata["unit"]

            choices = _literal_choices(annotation)
            metadata_choices = config_field.metadata.get("choices")
            if choices is not None and metadata_choices is not None:
                if choices != list(metadata_choices):
                    raise ValueError(f"Incompatible choices for {self.name}.{config_field.name}")
            elif choices is None and metadata_choices is not None:
                choices = list(metadata_choices)
            if choices is not None:
                entry["choices"] = choices

            schema.append(entry)

        return schema


_REGISTRY: dict[str, RepresentationInfo] = {}


def _require_callable(obj: Any, attr: str, name: str) -> None:
    fn = getattr(obj, attr, None)
    if fn is None or not callable(fn):
        raise TypeError(f"Encoder '{name}' must implement {attr}().")


def register_encoder(
    config_type: type[CanonicalRepresentationConfig],
    encoder_type: type[BaseEncoder],
    *,
    description: str = "",
    allow_override: bool = False,
) -> None:
    """Register one canonical explicit representation."""
    if not isinstance(config_type, type) or not issubclass(
        config_type, CanonicalRepresentationConfig
    ):
        raise TypeError("config_type must be a CanonicalRepresentationConfig subclass")
    if not isinstance(encoder_type, type) or not issubclass(encoder_type, BaseEncoder):
        raise TypeError("encoder_type must be a BaseEncoder subclass")

    name = config_type.representation_name
    if not isinstance(name, str) or not name.strip():
        raise TypeError("config_type.representation_name must be a non-empty str")
    if _CONFIG_TYPES_BY_REPRESENTATION.get(name) is not config_type:
        raise TypeError(f"config_type is not the canonical config registered for {name!r}")
    if not allow_override and name in _REGISTRY:
        raise KeyError(f"Encoder already registered: {name}")

    info = RepresentationInfo(
        config_type=config_type,
        encoder_type=encoder_type,
        description=description,
    )
    default_config = info.default_config()
    if default_config.representation_name != name:
        raise TypeError(
            "Default config representation_name does not match its canonical name: "
            f"{default_config.representation_name!r} != {name!r}"
        )

    encoder = encoder_type()
    _require_callable(encoder, "encode", name)
    _require_callable(encoder, "fingerprint", name)

    if encoder.name != name:
        raise TypeError(f"Encoder name does not match canonical name: {encoder.name!r} != {name!r}")
    if encoder.granularity_contract.level != info.granularity:
        raise TypeError(
            "Encoder granularity does not match canonical config: "
            f"{encoder.granularity_contract.level!r} != {info.granularity!r}"
        )
    if info.representation_type not in {"cloud", "graph", "matrix", "grid"}:
        raise TypeError(
            f"Unsupported representation type for '{name}': {info.representation_type!r}"
        )

    fingerprint = encoder.fingerprint()
    if not isinstance(fingerprint, str) or not fingerprint.strip():
        raise TypeError(f"Encoder '{name}'.fingerprint() must return a non-empty str.")

    _REGISTRY[name] = info


def get_encoder(name: str, **kwargs: Any) -> BaseEncoder:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown encoder: {name}")
    return _REGISTRY[name].encoder_type(**kwargs)


def list_encoders() -> list[str]:
    return sorted(_REGISTRY)


def get_encoder_info(name: str) -> RepresentationInfo:
    if name not in _REGISTRY:
        raise KeyError(f"No encoder info: {name}")
    return _REGISTRY[name]


def _representation_info_or_raise(representation: str) -> RepresentationInfo:
    try:
        return get_encoder_info(representation)
    except KeyError as error:
        raise RepresentationNotFoundError(f"Unknown representation {representation!r}") from error


def validate_resolved_representation_config(
    representation: str,
    *,
    config: CanonicalRepresentationConfig,
    params: Mapping[str, Any] | None = None,
) -> CanonicalRepresentationConfig:
    """Validate an already-resolved explicit representation config without rebuilding it."""
    info = _representation_info_or_raise(representation)
    requested = {} if params is None else dict(params)
    if requested:
        names = ", ".join(sorted(requested))
        raise ConfigError(f"config cannot be combined with representation parameters: {names}")
    if not isinstance(config, info.config_type):
        raise ConfigError(f"Config for {representation!r} must be an {info.config_type.__name__}")
    return config


def resolve_representation_config(
    representation: str,
    *,
    config: CanonicalRepresentationConfig | None = None,
    params: Mapping[str, Any] | None = None,
) -> CanonicalRepresentationConfig:
    """Resolve one explicit representation through its canonical config type."""
    info = _representation_info_or_raise(representation)

    requested = {} if params is None else dict(params)
    if config is not None:
        return validate_resolved_representation_config(
            representation,
            config=config,
            params=requested,
        )

    allowed = {config_field.name for config_field in info.config_fields()}
    unknown = sorted(set(requested) - allowed)
    if unknown:
        names = ", ".join(unknown)
        raise ConfigError(f"Unknown parameters for {representation!r}: {names}")

    try:
        return info.config_type(**requested)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"Invalid parameters for {representation!r}: {error}") from error


def list_encoder_info() -> list[RepresentationInfo]:
    return [get_encoder_info(name) for name in list_encoders()]
