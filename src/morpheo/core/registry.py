from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Any, List


@dataclass(frozen=True)
class EncoderInfo:
    name: str
    type: str  # graph|grid|matrix|cloud
    level: str = "ca"      # ca|all_atom|surface (future)
    family: str = "residue"  # residue|atom|surface (future)
    desc: str = ""


_REGISTRY: Dict[str, Callable[[], Any]] = {}
_INFO: Dict[str, EncoderInfo] = {}


def register_encoder(
    name: str,
    factory: Callable[[], Any],
    *,
    type: str = "graph",
    level: str = "ca",
    family: str = "residue",
    desc: str = "",
) -> None:
    if name in _REGISTRY:
        raise KeyError(f"Encoder already registered: {name}")
    _REGISTRY[name] = factory
    _INFO[name] = EncoderInfo(name=name, type=type, level=level, family=family, desc=desc)


def get_encoder(name: str) -> Any:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown encoder: {name}")
    return _REGISTRY[name]()


def list_encoders() -> List[str]:
    return sorted(_REGISTRY.keys())


def get_encoder_info(name: str) -> EncoderInfo:
    if name not in _INFO:
        raise KeyError(f"No encoder info: {name}")
    return _INFO[name]


def list_encoder_info() -> List[EncoderInfo]:
    return [get_encoder_info(n) for n in list_encoders()]
