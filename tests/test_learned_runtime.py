from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields

import pytest

from eris.core.artifact_schema import compute_config_fingerprint
from eris.core.learned import (
    ResolvedExecution,
    StructuralEmbedderConfig,
    StructuralExecutionConfig,
    build_execution_provenance,
    execution_request_cache_identity,
    resolve_execution,
)
from eris.exceptions import ConfigError


def test_execution_config_is_frozen_minimal_and_defaults_to_auto_fp32():
    request = StructuralExecutionConfig()

    assert request == StructuralExecutionConfig(device="auto", precision="fp32")
    assert {field.name for field in fields(request)} == {"device", "precision"}
    assert not hasattr(request, "to_dict")
    with pytest.raises(FrozenInstanceError):
        request.device = "cpu"  # pyrefly: ignore[read-only]


@pytest.mark.parametrize("device", ["auto", "cpu"])
def test_execution_config_accepts_exact_device_request_vocabulary(device):
    assert StructuralExecutionConfig(device=device).device == device


@pytest.mark.parametrize("device", ["", "cuda", "cuda:0", "cuda:", "mps", None])
def test_execution_config_rejects_non_cpu_device_requests(device):
    with pytest.raises(ConfigError, match="device must be one of: auto, cpu"):
        StructuralExecutionConfig(device=device)


@pytest.mark.parametrize("precision", ["fp16", "bf16", "float32", None])
def test_execution_config_rejects_non_fp32_precisions(precision):
    with pytest.raises(ConfigError, match="precision must be fp32"):
        StructuralExecutionConfig(precision=precision)


@pytest.mark.parametrize("device", ["auto", "cpu"])
def test_requests_resolve_to_cpu_fp32(device):
    request = StructuralExecutionConfig(device=device)

    assert resolve_execution(request) == ResolvedExecution(device="cpu", precision="fp32")


def test_execution_provenance_is_exact_deterministic_and_json_safe():
    request = StructuralExecutionConfig(device="auto", precision="fp32")
    resolved = resolve_execution(request)
    expected = {
        "requested": {"device": "auto", "precision": "fp32"},
        "resolved": {"device": "cpu", "precision": "fp32"},
    }

    assert build_execution_provenance(request, resolved) == expected
    assert build_execution_provenance(request, resolved) == expected
    assert json.loads(json.dumps(expected)) == expected


@pytest.mark.parametrize("device", ["auto", "cpu"])
def test_request_cache_identity_is_precision_only(device):
    request = StructuralExecutionConfig(device=device)

    assert execution_request_cache_identity(request) == {"precision": "fp32"}


def test_scientific_fingerprint_is_independent_of_execution_requests():
    config = StructuralEmbedderConfig(
        representation_name="synthetic_embedding",
        output_mode="both",
    )
    fingerprint = compute_config_fingerprint(config.representation_name, config.to_dict(), chain_ids=["A"])

    assert StructuralExecutionConfig(device="auto") != StructuralExecutionConfig(device="cpu")
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

    subprocess.run([sys.executable, "-c", script], check=True)  # noqa: S603
