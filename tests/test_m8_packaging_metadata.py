from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
GEARNET_MARKER = "; python_version >= '3.10' and python_version < '3.11'"


def _pyproject() -> dict[str, object]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_project_metadata_describes_eris_v0_1() -> None:
    project = _pyproject()["project"]

    assert project["name"] == "eris"
    assert project["version"] == "0.1.0"
    assert project["requires-python"] == ">=3.10,<3.13"
    assert project["readme"] == "README.md"
    assert project["license"] == {"file": "LICENSE"}
    assert "Development Status :: 3 - Alpha" in project["classifiers"]
    assert all("MORPHEO" not in url for url in project["urls"].values())
    assert project["urls"] == {
        "Homepage": "https://github.com/kren-ai-lab/Eris",
        "Repository": "https://github.com/kren-ai-lab/Eris",
        "Documentation": "https://github.com/kren-ai-lab/Eris#readme",
        "Issues": "https://github.com/kren-ai-lab/Eris/issues",
        "Changelog": "https://github.com/kren-ai-lab/Eris/blob/main/CHANGELOG.md",
    }


def test_optional_extras_match_implemented_or_development_contracts() -> None:
    extras = _pyproject()["project"]["optional-dependencies"]

    assert set(extras) == {"gearnet", "pyg", "dev"}
    assert extras["pyg"] == ["torch", "torch-geometric"]
    assert "mypy" in extras["dev"]
    assert all(requirement.endswith(GEARNET_MARKER) for requirement in extras["gearnet"])
