import tomllib
from pathlib import Path
import morpheo

def test_version_matches_pyproject():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert morpheo.__version__ == data["project"]["version"]
