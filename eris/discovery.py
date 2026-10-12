from __future__ import annotations

import json
import types
from dataclasses import MISSING, fields
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from eris.core.registry import get_encoder_info, list_encoders
from eris.exceptions import RepresentationNotFoundError

_GRANULARITY_BY_OUTPUT_MODE = {
    "node": "residue",
    "global": "structure",
    "both": "mixed",
}


def _json_safe_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def _render_type(annotation: Any) -> str:
    if annotation is Any:
        return "any"
    if annotation is type(None):
        return "null"
    if annotation in {bool, int, float, str}:
        return annotation.__name__

    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is Literal:
        return " | ".join(repr(value) for value in arguments)
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


def _parameter_schema(config_type: type, *, representation_name: str) -> list[dict[str, Any]]:
    config = config_type(representation_name=representation_name)
    type_hints = get_type_hints(config_type)
    schema: list[dict[str, Any]] = []

    for config_field in fields(config):
        if config_field.name == "representation_name":
            continue

        annotation = type_hints[config_field.name]
        entry: dict[str, Any] = {
            "name": config_field.name,
            "type": _render_type(annotation),
            "default": getattr(config, config_field.name),
            "required": (config_field.default is MISSING and config_field.default_factory is MISSING),
            "nullable": _is_nullable(annotation),
            "description": config_field.metadata.get("description"),
        }
        if get_origin(annotation) is Literal:
            entry["choices"] = list(get_args(annotation))
        schema.append(entry)

    return schema


def _get_explicit_registration(name: str):
    try:
        return get_encoder_info(name)
    except KeyError as error:
        raise RepresentationNotFoundError(f"Unknown representation {name!r}") from error


def list_representations() -> list[str]:
    """Return canonical explicit representation names in Registry order."""
    return list_encoders()


def get_representation_info(name: str) -> dict[str, Any]:
    """Return JSON-safe public metadata for one explicit representation."""
    info = _get_explicit_registration(name)

    return _json_safe_copy(
        {
            "name": info.name,
            "family": "explicit",
            "type": info.representation_type,
            "granularity": info.granularity,
            "coordinate_source": info.default_coordinate_source,
            "description": info.description,
            "defaults": info.defaults(),
            "parameters": info.params_schema(),
        }
    )


def list_embedders() -> list[str]:
    """Return registered learned representation names without probing availability."""
    from eris.core.learned.registry import learned_registry

    return learned_registry.names()


def get_embedder_info(name: str) -> dict[str, Any]:
    """Return JSON-safe public metadata and lazy availability for one embedder."""
    from eris.core.learned.backends.gearnet import dependencies
    from eris.core.learned.registry import learned_registry

    try:
        info = learned_registry.get_info(name)
    except KeyError as error:
        raise RepresentationNotFoundError(f"Unknown embedder {name!r}") from error

    availability = dependencies.probe_gearnet_availability()
    output_modes = list(_GRANULARITY_BY_OUTPUT_MODE)
    return _json_safe_copy(
        {
            "name": info.representation_name,
            "family": "learned",
            "type": "embedding",
            "coordinate_source": "model_specific",
            "backend_id": info.backend_id,
            "encoder_version": info.encoder_version,
            "output_modes": output_modes,
            "granularity_by_output_mode": dict(_GRANULARITY_BY_OUTPUT_MODE),
            "parameters": _parameter_schema(
                info.config_type,
                representation_name=info.representation_name,
            ),
            "availability": {
                "available": availability.available,
                "reason": availability.reason,
            },
        }
    )


__all__ = [
    "get_embedder_info",
    "get_representation_info",
    "list_embedders",
    "list_representations",
]
