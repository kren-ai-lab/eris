from __future__ import annotations

from dataclasses import Field, dataclass, fields
from typing import Any

from eris.core.base_encoder.base_encoder import BaseEncoder
from eris.core.config import (
    _CONFIG_TYPES_BY_REPRESENTATION,
    CanonicalRepresentationConfig,
)


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


def list_encoder_info() -> list[RepresentationInfo]:
    return [get_encoder_info(name) for name in list_encoders()]
