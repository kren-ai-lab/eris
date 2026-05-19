from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class EncoderInfo:
    name: str
    type: str   # graph|matrix|cloud|grid|mesh
    level: str  # ca|residue|atom
    description: str = ""
    params_schema: Dict[str, Any] = field(default_factory=dict)


_REGISTRY: Dict[str, Callable[[], Any]] = {}
_INFO: Dict[str, EncoderInfo] = {}


def _require_callable(obj: Any, attr: str, name: str) -> None:
    fn = getattr(obj, attr, None)
    if fn is None or not callable(fn):
        raise TypeError(f"Encoder '{name}' must implement {attr}().")


def register_encoder(
    name: str,
    factory: Callable[[], Any],
    *,
    type: str,
    level: str,
    description: str = "",
    params_schema: Optional[Dict[str, Any]] = None,
    allow_override: bool = False,
) -> None:
    """
    Register an encoder factory and its metadata.

    Requirements (no BaseEncoder needed):
      - encoder.encode(...) must exist (callable)
      - encoder.fingerprint() must exist (callable) and return str
      - encoder.version is optional; if present should be str
    """
    if (not allow_override) and (name in _REGISTRY):
        raise KeyError(f"Encoder already registered: {name}")

    enc = factory()

    _require_callable(enc, "encode", name)
    _require_callable(enc, "fingerprint", name)

    fp = enc.fingerprint()
    if not isinstance(fp, str) or not fp.strip():
        raise TypeError(f"Encoder '{name}'.fingerprint() must return a non-empty str.")

    _REGISTRY[name] = factory
    _INFO[name] = EncoderInfo(
        name=name,
        type=type,
        level=level,
        description=description,
        params_schema=params_schema or {},
    )


def get_encoder(name: str, **kwargs):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown encoder: {name}")
    return _REGISTRY[name](**kwargs)


def list_encoders() -> List[str]:
    return sorted(_REGISTRY.keys())


def get_encoder_info(name: str) -> EncoderInfo:
    if name not in _INFO:
        raise KeyError(f"No encoder info: {name}")
    return _INFO[name]


def list_encoder_info() -> List[EncoderInfo]:
    return [get_encoder_info(n) for n in list_encoders()]
