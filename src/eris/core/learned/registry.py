from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from eris.core.learned.checkpoints import StructuralModelDescriptor
from eris.core.learned.config import StructuralEmbedderConfig
from eris.core.learned.protocol import StructuralEmbedderBackend
from eris.exceptions import ConfigError, ExecutionError

BackendFactory = Callable[[], StructuralEmbedderBackend[Any, Any, Any]]
AvailabilityProbe = Callable[[], "BackendAvailability"]

_IMPLEMENTATION_PATH = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")


@dataclass(frozen=True)
class BackendAvailability:
    """Dependency-level availability reported without instantiating a backend."""

    available: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.available) is not bool:
            raise TypeError("available must be a bool")
        if self.reason is not None:
            if not isinstance(self.reason, str):
                raise TypeError("reason must be a string or None")
            if not self.reason.strip():
                raise ValueError("reason must not be empty when provided")
        if not self.available and self.reason is None:
            raise ValueError("reason is required when the backend is unavailable")


@dataclass(frozen=True)
class EmbedderInfo:
    """One learned representation registration without an instantiated backend.

    ``encoder_version`` versions the Eris adapter contract, independently of model weights.
    """

    representation_name: str
    backend_id: str
    implementation: str
    encoder_version: str
    config_type: type[StructuralEmbedderConfig]
    backend_factory: BackendFactory = field(repr=False)
    availability_probe: AvailabilityProbe = field(repr=False)
    model_descriptor: StructuralModelDescriptor | None = None


def _implementation_path(instance: object) -> str:
    implementation_type = type(instance)
    return f"{implementation_type.__module__}.{implementation_type.__qualname__}"


class EmbedderRegistry:
    """Independent Registry for learned structural representation backends."""

    def __init__(self) -> None:
        self._entries: dict[str, EmbedderInfo] = {}

    def register(
        self,
        *,
        representation_name: str,
        backend_id: str,
        implementation: str,
        encoder_version: str,
        config_type: type[StructuralEmbedderConfig],
        model_descriptor: StructuralModelDescriptor | None = None,
        backend_factory: BackendFactory,
        availability_probe: AvailabilityProbe,
    ) -> EmbedderInfo:
        """Register one learned representation without probing or creating its backend."""

        if not isinstance(config_type, type) or not issubclass(
            config_type, StructuralEmbedderConfig
        ):
            raise TypeError("config_type must be StructuralEmbedderConfig or a subclass")
        if model_descriptor is not None and not isinstance(
            model_descriptor, StructuralModelDescriptor
        ):
            raise TypeError("model_descriptor must be a StructuralModelDescriptor or None")

        default_config = config_type(representation_name=representation_name)
        canonical_name = default_config.representation_name

        if not isinstance(backend_id, str) or not backend_id.strip():
            raise TypeError("backend_id must be a non-empty str")
        if backend_id != backend_id.strip():
            raise TypeError("backend_id must not contain surrounding whitespace")
        if (
            not isinstance(implementation, str)
            or _IMPLEMENTATION_PATH.fullmatch(implementation) is None
        ):
            raise TypeError("implementation must be a dotted import-path-like string")
        if not isinstance(encoder_version, str) or not encoder_version.strip():
            raise TypeError("encoder_version must be a non-empty str")
        if encoder_version != encoder_version.strip():
            raise TypeError("encoder_version must not contain surrounding whitespace")
        if not callable(backend_factory):
            raise TypeError("backend_factory must be callable")
        if not callable(availability_probe):
            raise TypeError("availability_probe must be callable")
        if canonical_name in self._entries:
            raise KeyError(f"Embedder already registered: {canonical_name}")

        info = EmbedderInfo(
            representation_name=canonical_name,
            backend_id=backend_id,
            implementation=implementation,
            encoder_version=encoder_version,
            config_type=config_type,
            model_descriptor=model_descriptor,
            backend_factory=backend_factory,
            availability_probe=availability_probe,
        )
        self._entries[canonical_name] = info
        return info

    def names(self) -> list[str]:
        """Return registered representation names in deterministic order."""

        return sorted(self._entries)

    def list_info(self) -> list[EmbedderInfo]:
        """Return registration descriptors without probing or creating backends."""

        return [self._entries[name] for name in self.names()]

    def get_info(self, name: str) -> EmbedderInfo:
        """Return one registration descriptor without probing or creating its backend."""

        if name not in self._entries:
            raise KeyError(f"Unknown embedder: {name}")
        return self._entries[name]

    def availability(self, name: str) -> BackendAvailability:
        """Run only the registered lightweight dependency-availability probe."""

        info = self.get_info(name)
        status = info.availability_probe()
        if not isinstance(status, BackendAvailability):
            raise ExecutionError(
                f"Availability probe for '{name}' did not return BackendAvailability"
            )
        return status

    def create(self, name: str) -> StructuralEmbedderBackend[Any, Any, Any]:
        """Create and validate an available backend without running its model lifecycle."""

        info = self.get_info(name)
        status = self.availability(name)
        if not status.available:
            raise ExecutionError(
                f"Embedder '{name}' backend '{info.backend_id}' is unavailable: {status.reason}"
            )

        instance = info.backend_factory()
        if not isinstance(instance, StructuralEmbedderBackend):
            raise ExecutionError(
                f"Backend factory for '{name}' did not return a StructuralEmbedderBackend"
            )
        if instance.backend_id != info.backend_id:
            raise ExecutionError(
                f"Backend ID mismatch for '{name}': {instance.backend_id!r} != {info.backend_id!r}"
            )

        actual_implementation = _implementation_path(instance)
        if actual_implementation != info.implementation:
            raise ExecutionError(
                f"Backend implementation mismatch for '{name}': "
                f"{actual_implementation!r} != {info.implementation!r}"
            )

        instance_available = instance.is_available()
        if type(instance_available) is not bool:
            raise ExecutionError(f"Backend '{name}'.is_available() must return a bool")
        if not instance_available:
            raise ExecutionError(f"Backend '{name}' contradicts its available Registry probe")
        return instance

    def create_config(self, name: str, **kwargs: Any) -> StructuralEmbedderConfig:
        """Create validated config bound to the registered representation identity."""

        info = self.get_info(name)
        if "representation_name" in kwargs:
            raise ConfigError("representation_name is fixed by the learned Registry")
        return info.config_type(representation_name=info.representation_name, **kwargs)


learned_registry = EmbedderRegistry()

__all__ = [
    "BackendAvailability",
    "EmbedderInfo",
    "EmbedderRegistry",
    "learned_registry",
]
