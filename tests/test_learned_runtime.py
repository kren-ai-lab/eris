from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields

import pytest

from eris.core.artifact_schema_v2 import compute_config_fingerprint
from eris.core.learned import (
    ExecutionCapabilities,
    ResolvedExecution,
    StructuralEmbedderConfig,
    StructuralExecutionConfig,
    build_execution_provenance,
    execution_cache_identity,
    execution_request_cache_identity,
    resolve_execution,
)
from eris.exceptions import ConfigError, ExecutionError


def _cpu_capabilities() -> ExecutionCapabilities:
    return ExecutionCapabilities(
        devices=("cpu",),
        precisions=("fp32",),
        preferred_device="cpu",
    )


def test_execution_config_is_frozen_minimal_and_defaults_to_auto_fp32():
    request = StructuralExecutionConfig()

    assert request == StructuralExecutionConfig(device="auto", precision="fp32")
    assert {field.name for field in fields(request)} == {"device", "precision"}
    assert not hasattr(request, "to_dict")
    with pytest.raises(FrozenInstanceError):
        request.device = "cpu"


@pytest.mark.parametrize("device", ["auto", "cpu", "cuda:0"])
def test_execution_config_accepts_exact_device_request_vocabulary(device):
    assert StructuralExecutionConfig(device=device).device == device


@pytest.mark.parametrize(
    "device",
    [
        "",
        "cuda:",
        "mps",
    ],
)
def test_execution_config_rejects_noncanonical_device_requests(device):
    with pytest.raises(ConfigError, match="device"):
        StructuralExecutionConfig(device=device)


@pytest.mark.parametrize("precision", ["fp32"])
def test_execution_config_accepts_exact_precision_vocabulary(precision):
    assert StructuralExecutionConfig(precision=precision).precision == precision


@pytest.mark.parametrize(
    "precision",
    ["float32", None],
)
def test_execution_config_rejects_unknown_precisions(precision):
    with pytest.raises(ConfigError, match="precision"):
        StructuralExecutionConfig(precision=precision)


def test_capabilities_are_frozen_preserve_order_and_do_not_require_cpu():
    capabilities = ExecutionCapabilities(
        devices=("cuda:2", "cuda:0"),
        precisions=("bf16", "fp32"),
        preferred_device="cuda:2",
    )

    assert capabilities.devices == ("cuda:2", "cuda:0")
    assert capabilities.precisions == ("bf16", "fp32")
    with pytest.raises(FrozenInstanceError):
        capabilities.preferred_device = "cuda:0"


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"devices": ()}, "devices"),
        ({"devices": ("auto",)}, "concrete"),
        ({"precisions": ()}, "precisions"),
        ({"preferred_device": "cuda:0"}, "appear in devices"),
    ],
)
def test_capabilities_reject_invalid_descriptors(overrides, match):
    values = {
        "devices": ("cpu",),
        "precisions": ("fp32",),
        "preferred_device": "cpu",
    }
    values.update(overrides)

    with pytest.raises(ConfigError, match=match):
        ExecutionCapabilities(**values)


def test_resolved_execution_is_frozen_and_requires_concrete_values():
    resolved = ResolvedExecution(device="cuda:0", precision="fp16")

    assert resolved.device == "cuda:0"
    with pytest.raises(FrozenInstanceError):
        resolved.device = "cpu"
    with pytest.raises(ConfigError, match="concrete"):
        ResolvedExecution(device="auto", precision="fp32")
    with pytest.raises(ConfigError, match="concrete"):
        ResolvedExecution(device="cuda", precision="fp32")
    with pytest.raises(ConfigError, match="precision"):
        ResolvedExecution(device="cpu", precision="fp64")


def test_auto_resolves_exactly_to_backend_preferred_device():
    capabilities = ExecutionCapabilities(
        devices=("cpu", "cuda:0"),
        precisions=("fp32",),
        preferred_device="cuda:0",
    )

    assert resolve_execution(StructuralExecutionConfig(), capabilities) == ResolvedExecution(
        device="cuda:0",
        precision="fp32",
    )


def test_explicit_cpu_resolves_only_when_advertised():
    request = StructuralExecutionConfig(device="cpu")

    assert resolve_execution(request, _cpu_capabilities()).device == "cpu"

    cuda_only = ExecutionCapabilities(
        devices=("cuda:0",),
        precisions=("fp32",),
        preferred_device="cuda:0",
    )
    with pytest.raises(ExecutionError, match="cpu.*cuda:0"):
        resolve_execution(request, cuda_only)


def test_generic_cuda_uses_lowest_numeric_index_not_tuple_or_lexical_order():
    capabilities = ExecutionCapabilities(
        devices=("cpu", "cuda:10", "cuda:2"),
        precisions=("fp32",),
        preferred_device="cpu",
    )

    resolved = resolve_execution(StructuralExecutionConfig(device="cuda"), capabilities)

    assert resolved.device == "cuda:2"


def test_generic_cuda_unavailable_is_actionable_and_never_falls_back():
    with pytest.raises(ExecutionError, match="cuda.*available devices: cpu"):
        resolve_execution(StructuralExecutionConfig(device="cuda"), _cpu_capabilities())


def test_explicit_cuda_index_requires_exact_membership_without_fallback():
    capabilities = ExecutionCapabilities(
        devices=("cpu", "cuda:1"),
        precisions=("fp32",),
        preferred_device="cpu",
    )

    assert (
        resolve_execution(StructuralExecutionConfig(device="cuda:1"), capabilities).device
        == "cuda:1"
    )
    with pytest.raises(ExecutionError, match="cuda:0.*cpu, cuda:1"):
        resolve_execution(StructuralExecutionConfig(device="cuda:0"), capabilities)


@pytest.mark.parametrize("precision", ["fp16"])
def test_valid_but_unsupported_precision_fails_without_fp32_fallback(precision):
    with pytest.raises(ExecutionError, match=f"{precision}.*supported precisions: fp32"):
        resolve_execution(
            StructuralExecutionConfig(precision=precision),
            _cpu_capabilities(),
        )


def test_resolution_is_deterministic():
    request = StructuralExecutionConfig(device="cuda", precision="bf16")
    capabilities = ExecutionCapabilities(
        devices=("cuda:10", "cpu", "cuda:2"),
        precisions=("bf16", "fp32"),
        preferred_device="cpu",
    )

    assert resolve_execution(request, capabilities) == resolve_execution(request, capabilities)


def test_execution_provenance_is_exact_deterministic_and_json_safe():
    request = StructuralExecutionConfig(device="auto", precision="fp32")
    resolved = ResolvedExecution(device="cuda:0", precision="fp32")
    expected = {
        "requested": {"device": "auto", "precision": "fp32"},
        "resolved": {"device": "cuda:0", "precision": "fp32"},
    }

    assert build_execution_provenance(request, resolved) == expected
    assert build_execution_provenance(request, resolved) == expected
    assert json.loads(json.dumps(expected)) == expected


def test_execution_cache_identity_uses_only_resolved_precision():
    cpu_fp32 = ResolvedExecution(device="cpu", precision="fp32")
    cuda_fp32 = ResolvedExecution(device="cuda:0", precision="fp32")
    cpu_fp16 = ResolvedExecution(device="cpu", precision="fp16")

    assert execution_cache_identity(cpu_fp32) == {"precision": "fp32"}
    assert execution_cache_identity(cpu_fp32) == execution_cache_identity(cuda_fp32)
    assert execution_cache_identity(cpu_fp32) != execution_cache_identity(cpu_fp16)


@pytest.mark.parametrize("precision", ["fp32"])
def test_request_cache_identity_matches_successful_resolved_projection(precision):
    request = StructuralExecutionConfig(device="cpu", precision=precision)
    capabilities = ExecutionCapabilities(
        devices=("cpu",),
        precisions=("fp32", "fp16", "bf16"),
        preferred_device="cpu",
    )
    resolved = resolve_execution(request, capabilities)

    assert execution_request_cache_identity(request) == {"precision": precision}
    assert execution_request_cache_identity(request) == execution_cache_identity(resolved)
    assert "device" not in execution_request_cache_identity(request)


def test_scientific_fingerprint_is_independent_of_execution_requests():
    config = StructuralEmbedderConfig(
        representation_name="synthetic_embedding",
        output_mode="both",
    )
    fingerprint = compute_config_fingerprint(
        config.representation_name, config.to_dict(), chain_ids=["A"]
    )

    assert StructuralExecutionConfig(device="auto", precision="fp32") != (
        StructuralExecutionConfig(device="cpu", precision="fp16")
    )
    assert (
        compute_config_fingerprint(config.representation_name, config.to_dict(), chain_ids=["A"])
        == fingerprint
    )
    assert set(config.to_dict()) == {"representation_name", "output_mode"}


def test_importing_execution_contract_does_not_load_optional_frameworks():
    script = """
import sys
import eris
import eris.core.learned
import eris.core.learned.execution

blocked = {'torch', 'torchdrug', 'torch_geometric', 'transformers'}
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise SystemExit(f'optional frameworks loaded: {loaded}')
assert not hasattr(eris, 'StructuralExecutionConfig')
"""

    subprocess.run([sys.executable, "-c", script], check=True)
