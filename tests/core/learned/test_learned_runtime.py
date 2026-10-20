from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from eris.core.learned import (
    ResolvedExecution,
    StructuralExecutionConfig,
    build_execution_provenance,
    execution_request_cache_identity,
    resolve_execution,
)
from eris.exceptions import ConfigError


def test_execution_config_is_frozen_and_defaults_to_auto_fp32():
    request = StructuralExecutionConfig()

    assert request == StructuralExecutionConfig(device="auto", precision="fp32")
    with pytest.raises(FrozenInstanceError):
        request.device = "cpu"  # pyrefly: ignore[read-only]


@pytest.mark.parametrize("device", ["cuda", "mps", None])
def test_execution_config_rejects_non_cpu_device_requests(device):
    with pytest.raises(ConfigError, match="device must be one of: auto, cpu"):
        StructuralExecutionConfig(device=device)


@pytest.mark.parametrize("precision", ["fp16", None])
def test_execution_config_rejects_non_fp32_precisions(precision):
    with pytest.raises(ConfigError, match="precision must be fp32"):
        StructuralExecutionConfig(precision=precision)


@pytest.mark.parametrize("device", ["auto", "cpu"])
def test_requests_resolve_to_cpu_fp32_and_cache_only_precision(device):
    request = StructuralExecutionConfig(device=device)

    assert resolve_execution(request) == ResolvedExecution(device="cpu", precision="fp32")
    assert execution_request_cache_identity(request) == {"precision": "fp32"}


def test_execution_provenance_records_request_and_resolution():
    request = StructuralExecutionConfig(device="auto", precision="fp32")
    provenance = build_execution_provenance(request, resolve_execution(request))

    assert provenance == {
        "requested": {"device": "auto", "precision": "fp32"},
        "resolved": {"device": "cpu", "precision": "fp32"},
    }
    assert json.loads(json.dumps(provenance)) == provenance
