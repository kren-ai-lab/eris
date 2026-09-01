from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
RUFF_EXCLUDES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "build",
    "dist",
    "notebooks",
    "out",
}


def _pyproject() -> dict[str, object]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_static_baseline_configuration_is_reproducible() -> None:
    config = _pyproject()
    ruff = config["tool"]["ruff"]

    assert config["project"]["optional-dependencies"]["dev"] == [
        "pytest",
        "ruff==0.16.5",
        "build",
        "pre-commit",
        "tomli>=1.1; python_version < '3.11'",
    ]
    assert ruff["target-version"] == "py310"
    assert ruff["force-exclude"] is True
    assert set(ruff["exclude"]) == RUFF_EXCLUDES
    assert ruff["lint"]["select"] == ["E9", "F63", "F7", "F82"]


def test_pre_commit_uses_the_frozen_ruff_baseline_without_mypy() -> None:
    config = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")

    assert "repo: https://github.com/astral-sh/ruff-pre-commit" in config
    assert "rev: v0.16.5" in config
    assert "id: ruff" in config
    assert "id: ruff-format" in config
    assert "mirrors-mypy" not in config
    assert "id: mypy" not in config
