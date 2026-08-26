from importlib.metadata import version as distribution_version
from pathlib import Path

import eris

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib


def test_version_matches_distribution_metadata():
    assert eris.__version__ == distribution_version("eris")


def test_distribution_metadata_matches_pyproject():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert distribution_version("eris") == data["project"]["version"]
