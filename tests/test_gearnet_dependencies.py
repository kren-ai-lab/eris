from __future__ import annotations

import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest
import tomllib

from eris.core.learned import BackendAvailability
from eris.core.learned.backends.gearnet import dependencies

EXPECTED_VERSIONS = {
    "numpy": "1.26.4",
    "torch": "2.11.0+cpu",
    "torchdrug": "0.2.1",
    "torch-scatter": "2.1.2+pt211cpu",
    "torch-cluster": "1.6.3+pt211cpu",
    "rdkit-pypi": "2022.9.5",
    "ninja": "1.13.0",
}


@pytest.fixture
def compatible_environment(monkeypatch):
    versions = EXPECTED_VERSIONS.copy()

    def distribution_version(name: str) -> str:
        try:
            return versions[name]
        except KeyError:
            raise metadata.PackageNotFoundError(name) from None

    monkeypatch.setattr(dependencies, "_current_python_version", lambda: (3, 10))
    monkeypatch.setattr(dependencies, "_distribution_version", distribution_version)
    monkeypatch.setattr(dependencies, "_module_exists", lambda name: True)
    monkeypatch.setattr(dependencies, "_run_import_smoke", lambda: None)
    return versions


def test_frozen_dependency_contract_is_exact_and_complete():
    contract = {
        item.distribution: (item.import_name, item.requirement)
        for item in dependencies.GEARNET_DEPENDENCY_REQUIREMENTS
    }

    assert dependencies.GEARNET_PYTHON_REQUIREMENT == ">=3.10,<3.11"
    assert contract == {
        "numpy": ("numpy", ">=1.26,<2"),
        "torch": ("torch", "==2.11.0"),
        "torchdrug": ("torchdrug", "==0.2.1"),
        "torch-scatter": ("torch_scatter", "==2.1.2"),
        "torch-cluster": ("torch_cluster", "==1.6.3"),
        "rdkit-pypi": ("rdkit", "==2022.9.5"),
        "ninja": ("ninja", ">=1.11,<2"),
    }


def test_python_310_and_local_wheel_versions_are_accepted(compatible_environment):
    assert dependencies.probe_gearnet_availability() == BackendAvailability(available=True)


@pytest.mark.parametrize("python_version", [(3, 9), (3, 11), (3, 12)])
def test_unsupported_python_is_actionable(monkeypatch, python_version):
    monkeypatch.setattr(dependencies, "_current_python_version", lambda: python_version)

    status = dependencies.probe_gearnet_availability()

    assert not status.available
    assert ">=3.10,<3.11" in status.reason
    assert f"{python_version[0]}.{python_version[1]}" in status.reason
    assert "Eris core remains supported" in status.reason


def test_missing_distribution_is_actionable(compatible_environment):
    compatible_environment.pop("torchdrug")

    status = dependencies.probe_gearnet_availability()

    assert not status.available
    assert "not installed: torchdrug" in status.reason
    assert "documented Eris GearNet CPU environment" in status.reason


def test_missing_import_module_is_actionable(monkeypatch, compatible_environment):
    monkeypatch.setattr(
        dependencies,
        "_module_exists",
        lambda name: name != "torch_cluster",
    )

    status = dependencies.probe_gearnet_availability()

    assert not status.available
    assert "import modules are missing: torch_cluster" in status.reason


@pytest.mark.parametrize(
    ("distribution", "installed", "expected"),
    [
        ("numpy", "2.0.0", ">=1.26,<2"),
        ("torch", "2.12.1+cpu", "==2.11.0"),
        ("torchdrug", "0.2.0", "==0.2.1"),
        ("torch-scatter", "2.1.1+pt211cpu", "==2.1.2"),
        ("torch-cluster", "1.6.2+pt211cpu", "==1.6.3"),
        ("rdkit-pypi", "2023.9.1", "==2022.9.5"),
        ("ninja", "2.0.0", ">=1.11,<2"),
    ],
)
def test_incompatible_version_is_actionable(
    compatible_environment,
    distribution,
    installed,
    expected,
):
    compatible_environment[distribution] = installed

    status = dependencies.probe_gearnet_availability()

    assert not status.available
    assert f"{distribution} {installed}" in status.reason
    assert f"expected {expected}" in status.reason


def test_broken_import_smoke_is_unavailable_without_traceback(
    monkeypatch,
    compatible_environment,
):
    monkeypatch.setattr(
        dependencies,
        "_run_import_smoke",
        lambda: "GearNet dependency import smoke failed: ImportError: missing DLL",
    )

    status = dependencies.probe_gearnet_availability()

    assert not status.available
    assert status.reason == "GearNet dependency import smoke failed: ImportError: missing DLL"
    assert "Traceback" not in status.reason


def test_import_smoke_extracts_actionable_error_without_traceback(monkeypatch):
    result = subprocess.CompletedProcess(
        args=[],
        returncode=1,
        stdout="",
        stderr=(
            "Traceback (most recent call last):\n"
            '  File "<string>", line 1, in <module>\n'
            "ImportError: DLL load failed\n"
        ),
    )
    monkeypatch.setattr(dependencies.subprocess, "run", lambda *args, **kwargs: result)

    error = dependencies._run_import_smoke()

    assert error == "GearNet dependency import smoke failed: ImportError: DLL load failed"
    assert "Traceback" not in error


def test_gearnet_extra_matches_the_frozen_dependency_contract():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert pyproject["project"]["optional-dependencies"]["gearnet"] == [
        "numpy>=1.26,<2",
        "torch==2.11.0",
        "torchdrug==0.2.1",
        "torch-scatter==2.1.2",
        "torch-cluster==1.6.3",
        "rdkit-pypi==2022.9.5",
        "ninja>=1.11,<2",
    ]


def test_gearnet_dependency_import_is_framework_lazy_and_registry_neutral():
    script = """
import sys
import eris
import eris.core.learned
import eris.core.learned.backends.gearnet.dependencies
from eris.core.learned.registry import learned_registry

assert learned_registry.names() == []
blocked = {'torch', 'torchdrug', 'torch_scatter', 'torch_cluster', 'rdkit'}
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise SystemExit(f'optional GearNet frameworks loaded: {loaded}')
"""

    subprocess.run([sys.executable, "-c", script], check=True)
