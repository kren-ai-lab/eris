from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError, dataclass

import pytest

from eris.core.learned import (
    BackendAvailability,
    EmbedderInfo,
    EmbedderRegistry,
    StructuralEmbedderBackend,
    StructuralEmbedderConfig,
    VerifiedCheckpoint,
    learned_registry,
)
from eris.core.registry import list_encoders
from eris.exceptions import ConfigError, ExecutionError


class FakeBackend:
    def __init__(self, backend_id: str = "fake", *, available: bool = True) -> None:
        self.backend_id = backend_id
        self._available = available
        self.is_available_calls = 0
        self.preprocess_calls = 0
        self.load_calls = 0
        self.infer_calls = 0

    def is_available(self) -> bool:
        self.is_available_calls += 1
        return self._available

    def preprocess(self, structure, *, config):
        self.preprocess_calls += 1
        return structure

    def load(self, checkpoint: VerifiedCheckpoint):
        del checkpoint
        self.load_calls += 1
        return object()

    def infer(self, model, prepared, *, config):
        self.infer_calls += 1
        raise AssertionError("inference must not run during Registry creation")


class InvalidBackend:
    backend_id = "fake"


@dataclass(frozen=True, kw_only=True)
class ExtendedConfig(StructuralEmbedderConfig):
    scale: int = 1


FAKE_IMPLEMENTATION = f"{FakeBackend.__module__}.{FakeBackend.__qualname__}"


def _available() -> BackendAvailability:
    return BackendAvailability(available=True)


def _register(
    registry: EmbedderRegistry,
    *,
    representation_name: str = "synthetic_embedding",
    backend_id: str = "fake",
    implementation: str = FAKE_IMPLEMENTATION,
    config_type: type[StructuralEmbedderConfig] = StructuralEmbedderConfig,
    backend_factory=None,
    availability_probe=None,
) -> EmbedderInfo:
    factory = backend_factory or (lambda: FakeBackend(backend_id))
    probe = availability_probe or _available
    return registry.register(
        representation_name=representation_name,
        backend_id=backend_id,
        implementation=implementation,
        config_type=config_type,
        backend_factory=factory,
        availability_probe=probe,
    )


def test_backend_availability_is_frozen_and_validated():
    available = BackendAvailability(available=True)
    unavailable = BackendAvailability(available=False, reason="fake dependency missing")

    assert available.reason is None
    assert unavailable.reason == "fake dependency missing"
    with pytest.raises(FrozenInstanceError):
        available.available = False
    with pytest.raises(TypeError, match="available must be a bool"):
        BackendAvailability(available=1)
    with pytest.raises(ValueError, match="reason is required"):
        BackendAvailability(available=False)
    with pytest.raises(ValueError, match="reason must not be empty"):
        BackendAvailability(available=False, reason="   ")


def test_isolated_and_production_registries_begin_empty():
    registry = EmbedderRegistry()

    assert registry.names() == []
    assert registry.list_info() == []
    assert learned_registry.names() == []
    assert learned_registry.list_info() == []


def test_registration_and_discovery_are_deterministic_and_lazy():
    registry = EmbedderRegistry()
    calls = {"probe": 0, "factory": 0}

    def probe() -> BackendAvailability:
        calls["probe"] += 1
        return BackendAvailability(available=True)

    def factory() -> FakeBackend:
        calls["factory"] += 1
        return FakeBackend()

    second = _register(
        registry,
        representation_name="z_embedding",
        backend_factory=factory,
        availability_probe=probe,
    )
    first = _register(
        registry,
        representation_name="a_embedding",
        backend_factory=factory,
        availability_probe=probe,
    )

    assert calls == {"probe": 0, "factory": 0}
    assert registry.names() == ["a_embedding", "z_embedding"]
    assert registry.list_info() == [first, second]
    assert registry.get_info("z_embedding") is second
    assert calls == {"probe": 0, "factory": 0}

    assert registry.availability("a_embedding") == BackendAvailability(available=True)
    assert calls == {"probe": 1, "factory": 0}


def test_unavailable_create_fails_before_factory():
    registry = EmbedderRegistry()
    calls = {"probe": 0, "factory": 0}

    def probe() -> BackendAvailability:
        calls["probe"] += 1
        return BackendAvailability(
            available=False,
            reason="fake optional dependency missing",
        )

    def factory() -> FakeBackend:
        calls["factory"] += 1
        return FakeBackend()

    _register(registry, backend_factory=factory, availability_probe=probe)

    status = registry.availability("synthetic_embedding")
    assert status == BackendAvailability(
        available=False,
        reason="fake optional dependency missing",
    )
    with pytest.raises(
        ExecutionError,
        match="synthetic_embedding.*fake.*fake optional dependency missing",
    ):
        registry.create("synthetic_embedding")

    assert calls == {"probe": 2, "factory": 0}


def test_available_create_invokes_factory_once_and_only_confirms_availability():
    registry = EmbedderRegistry()
    created: list[FakeBackend] = []
    calls = {"probe": 0, "factory": 0}

    def probe() -> BackendAvailability:
        calls["probe"] += 1
        return BackendAvailability(available=True)

    def factory() -> FakeBackend:
        calls["factory"] += 1
        backend = FakeBackend()
        created.append(backend)
        return backend

    _register(registry, backend_factory=factory, availability_probe=probe)

    backend = registry.create("synthetic_embedding")

    assert backend is created[0]
    assert isinstance(backend, StructuralEmbedderBackend)
    assert backend.backend_id == "fake"
    assert calls == {"probe": 1, "factory": 1}
    assert backend.is_available_calls == 1
    assert backend.preprocess_calls == 0
    assert backend.load_calls == 0
    assert backend.infer_calls == 0


def test_create_rejects_an_invalid_protocol_instance():
    registry = EmbedderRegistry()
    _register(registry, backend_factory=InvalidBackend)

    with pytest.raises(ExecutionError, match="did not return a StructuralEmbedderBackend"):
        registry.create("synthetic_embedding")


def test_create_rejects_backend_id_mismatch():
    registry = EmbedderRegistry()
    _register(registry, backend_id="fake-a", backend_factory=lambda: FakeBackend("fake-b"))

    with pytest.raises(ExecutionError, match="Backend ID mismatch"):
        registry.create("synthetic_embedding")


def test_create_rejects_implementation_mismatch():
    registry = EmbedderRegistry()
    _register(registry, implementation="fake.module.WrongBackend")

    with pytest.raises(ExecutionError, match="Backend implementation mismatch"):
        registry.create("synthetic_embedding")


def test_create_rejects_instance_availability_contradiction():
    registry = EmbedderRegistry()
    _register(registry, backend_factory=lambda: FakeBackend(available=False))

    with pytest.raises(ExecutionError, match="contradicts its available Registry probe"):
        registry.create("synthetic_embedding")


def test_duplicate_registration_is_rejected_without_overwrite():
    registry = EmbedderRegistry()
    original = _register(registry)

    with pytest.raises(KeyError, match="Embedder already registered: synthetic_embedding"):
        _register(registry)

    assert registry.get_info("synthetic_embedding") is original


@pytest.mark.parametrize("operation", ["get_info", "availability", "create", "create_config"])
def test_unknown_embedder_operations_use_consistent_key_error(operation):
    registry = EmbedderRegistry()

    with pytest.raises(KeyError, match="Unknown embedder: missing"):
        getattr(registry, operation)("missing")


def test_backend_id_can_be_shared_by_distinct_representations():
    registry = EmbedderRegistry()
    _register(registry, representation_name="first_embedding", backend_id="shared")
    _register(registry, representation_name="second_embedding", backend_id="shared")

    assert registry.names() == ["first_embedding", "second_embedding"]
    assert {info.backend_id for info in registry.list_info()} == {"shared"}


def test_create_config_binds_registered_identity_and_validates_options():
    registry = EmbedderRegistry()
    _register(registry)

    default = registry.create_config("synthetic_embedding")
    node = registry.create_config("synthetic_embedding", output_mode="node")

    assert default == StructuralEmbedderConfig(representation_name="synthetic_embedding")
    assert default.output_mode == "both"
    assert node.representation_name == "synthetic_embedding"
    assert node.output_mode == "node"
    with pytest.raises(ConfigError, match="representation_name is fixed"):
        registry.create_config("synthetic_embedding", representation_name="other")
    with pytest.raises(ConfigError, match="output_mode"):
        registry.create_config("synthetic_embedding", output_mode="local")


def test_registration_supports_compatible_config_subclasses():
    registry = EmbedderRegistry()
    info = _register(registry, config_type=ExtendedConfig)

    config = registry.create_config("synthetic_embedding", scale=3)

    assert info.config_type is ExtendedConfig
    assert config == ExtendedConfig(representation_name="synthetic_embedding", scale=3)


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        ("backend_id", "", "backend_id"),
        ("backend_id", " fake ", "surrounding whitespace"),
        ("implementation", "not-dotted", "implementation"),
        ("backend_factory", None, "backend_factory"),
        ("availability_probe", None, "availability_probe"),
    ],
)
def test_registration_rejects_invalid_descriptor_fields(field_name, value, match):
    registry = EmbedderRegistry()
    kwargs = {
        "representation_name": "synthetic_embedding",
        "backend_id": "fake",
        "implementation": FAKE_IMPLEMENTATION,
        "config_type": StructuralEmbedderConfig,
        "backend_factory": FakeBackend,
        "availability_probe": _available,
    }
    kwargs[field_name] = value

    with pytest.raises(TypeError, match=match):
        registry.register(**kwargs)


def test_registration_reuses_config_name_validation_without_running_callables():
    registry = EmbedderRegistry()
    calls = {"probe": 0, "factory": 0}

    def probe() -> BackendAvailability:
        calls["probe"] += 1
        return BackendAvailability(available=True)

    def factory() -> FakeBackend:
        calls["factory"] += 1
        return FakeBackend()

    info = _register(
        registry,
        representation_name=" normalized_embedding ",
        backend_factory=factory,
        availability_probe=probe,
    )

    assert info.representation_name == "normalized_embedding"
    assert registry.names() == ["normalized_embedding"]
    assert calls == {"probe": 0, "factory": 0}
    with pytest.raises(ConfigError, match="representation_name"):
        _register(EmbedderRegistry(), representation_name="   ")


def test_registration_rejects_unrelated_config_types():
    registry = EmbedderRegistry()

    with pytest.raises(TypeError, match="config_type"):
        registry.register(
            representation_name="synthetic_embedding",
            backend_id="fake",
            implementation=FAKE_IMPLEMENTATION,
            config_type=object,
            backend_factory=FakeBackend,
            availability_probe=_available,
        )


def test_availability_validates_probe_result_and_propagates_probe_errors():
    invalid = EmbedderRegistry()
    _register(invalid, availability_probe=lambda: True)

    with pytest.raises(ExecutionError, match="did not return BackendAvailability"):
        invalid.availability("synthetic_embedding")

    failing = EmbedderRegistry()

    def failing_probe() -> BackendAvailability:
        raise RuntimeError("probe bug")

    _register(failing, availability_probe=failing_probe)
    with pytest.raises(RuntimeError, match="probe bug"):
        failing.availability("synthetic_embedding")


def test_isolated_learned_registration_does_not_affect_explicit_registry():
    registry = EmbedderRegistry()
    _register(registry)

    assert len(list_encoders()) == 10
    assert "synthetic_embedding" not in list_encoders()
    assert learned_registry.names() == []


def test_importing_and_listing_learned_registry_does_not_load_frameworks():
    script = """
import sys
import eris
import eris.core.learned
from eris.core.learned.registry import learned_registry

assert learned_registry.names() == []
blocked = {'torch', 'torchdrug', 'torch_geometric', 'transformers'}
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise SystemExit(f'optional frameworks loaded: {loaded}')
"""

    subprocess.run([sys.executable, "-c", script], check=True)


def test_registry_has_only_internal_package_exports():
    import eris
    import eris.core.learned as learned

    assert learned.EmbedderRegistry is EmbedderRegistry
    assert learned.learned_registry is learned_registry
    assert not hasattr(eris, "list_embedders")
    assert not hasattr(eris, "get_embedder")
    assert not hasattr(eris, "get_embedder_info")
