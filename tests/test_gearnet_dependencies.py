from __future__ import annotations

import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

from eris.core.learned import BackendAvailability
from eris.core.learned.backends.gearnet import dependencies


@pytest.fixture
def compatible_environment(monkeypatch):
    versions = {"torch": "2.11.0+cpu"}

    def distribution_version(name: str) -> str:
        try:
            return versions[name]
        except KeyError:
            raise metadata.PackageNotFoundError(name) from None

    monkeypatch.setattr(dependencies, "_current_python_version", lambda: (3, 10))
    monkeypatch.setattr(dependencies, "_distribution_version", distribution_version)
    return versions


def test_frozen_native_dependency_contract_is_exact():
    contract = {
        item.distribution: (item.import_name, item.requirement)
        for item in dependencies.GEARNET_DEPENDENCY_REQUIREMENTS
    }

    assert dependencies.GEARNET_PYTHON_REQUIREMENT == ">=3.10,<3.14"
    assert contract == {"torch": ("torch", "==2.11.0")}


@pytest.mark.parametrize("python_version", [(3, 10), (3, 11), (3, 12), (3, 13)])
def test_supported_python_and_cpu_wheel_version_are_accepted(
    compatible_environment, monkeypatch, python_version
):
    monkeypatch.setattr(dependencies, "_current_python_version", lambda: python_version)

    assert dependencies.probe_gearnet_availability() == BackendAvailability(available=True)


@pytest.mark.parametrize("python_version", [(3, 9), (3, 14)])
def test_unsupported_python_is_actionable(monkeypatch, python_version):
    monkeypatch.setattr(dependencies, "_current_python_version", lambda: python_version)

    status = dependencies.probe_gearnet_availability()

    assert not status.available
    assert ">=3.10,<3.14" in status.reason
    assert f"{python_version[0]}.{python_version[1]}" in status.reason


def test_missing_torch_distribution_is_actionable(compatible_environment):
    compatible_environment.pop("torch")

    status = dependencies.probe_gearnet_availability()

    assert not status.available
    assert "not installed: torch" in status.reason
    assert "native GearNet CPU runtime" in status.reason


@pytest.mark.parametrize("installed", ["2.10.0", "2.12.1+cpu", "2.14.0"])
def test_unsupported_torch_version_is_rejected(compatible_environment, installed):
    compatible_environment["torch"] = installed

    status = dependencies.probe_gearnet_availability()

    assert not status.available
    assert f"torch {installed}" in status.reason
    assert "expected ==2.11.0" in status.reason


def test_torch_214_cannot_enter_the_native_runtime(compatible_environment):
    compatible_environment["torch"] = "2.14.0"

    with pytest.raises(ImportError) as captured:
        dependencies.import_native_gearnet_torch()

    assert "torch==2.11.0" in str(captured.value)
    assert "found torch 2.14.0" in str(captured.value)


def test_gearnet_extra_is_intentionally_legacy_until_gmr5():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    extra = pyproject["project"]["optional-dependencies"]["gearnet"]

    assert any(requirement.startswith("torchdrug==0.2.1") for requirement in extra)
    assert any(requirement.startswith("rdkit-pypi==2022.9.5") for requirement in extra)
    assert all("python_version < '3.11'" in requirement for requirement in extra)


def test_gearnet_dependency_import_is_framework_lazy_and_registry_neutral():
    script = """
import sys
import eris
import eris.core.learned
import eris.core.learned.backends.gearnet.dependencies
from eris.core.learned.registry import learned_registry

assert learned_registry.names() == ['gearnet_edge']
blocked = {'torch', 'torchdrug', 'torch_scatter', 'torch_cluster', 'torch_sparse', 'torch_geometric', 'rdkit', 'ninja'}
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise SystemExit(f'optional GearNet frameworks loaded: {loaded}')
"""

    subprocess.run([sys.executable, "-c", script], check=True)
