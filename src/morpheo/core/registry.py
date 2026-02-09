from __future__ import annotations

from typing import Callable, Dict

from morpheo.core.types import StructuralEncoder
from morpheo.exceptions import EncoderNotFoundError

_REGISTRY: Dict[str, Callable[[], StructuralEncoder]] = {}


def register_encoder(name: str, factory: Callable[[], StructuralEncoder]) -> None:
    _REGISTRY[name] = factory


def list_encoders() -> list[str]:
    return sorted(_REGISTRY.keys())


def get_encoder(name: str) -> StructuralEncoder:
    if name not in _REGISTRY:
        raise EncoderNotFoundError(f"Encoder not found: {name}. Available: {list_encoders()}")
    return _REGISTRY[name]()
