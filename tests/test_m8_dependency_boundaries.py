from __future__ import annotations

import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
PDB = ROOT / "tests" / "data" / "1AQ2.pdb"
OPTIONAL_FRAMEWORKS = {
    "torch",
    "torch_geometric",
    "torchdrug",
    "torch_scatter",
    "torch_cluster",
    "rdkit",
}


def _run_without_optional_frameworks(script: str) -> None:
    guard = f"""
import builtins

_optional = {OPTIONAL_FRAMEWORKS!r}
_real_import = builtins.__import__

def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.partition('.')[0] in _optional:
        raise ImportError(f"optional framework deliberately unavailable: {{name}}")
    return _real_import(name, globals, locals, fromlist, level)

builtins.__import__ = _blocked_import
"""
    subprocess.run(
        [sys.executable, "-c", guard + script],
        cwd=ROOT,
        check=True,
    )


def test_core_declares_direct_runtime_imports_and_python_range() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = set(pyproject["project"]["dependencies"])

    assert pyproject["project"]["requires-python"] == ">=3.10,<3.13"
    for requirement in ("numpy", "gemmi", "platformdirs", "typer>=0.9", "rich"):
        assert requirement in dependencies


def test_core_explicit_workflow_and_cli_work_without_optional_frameworks(tmp_path: Path) -> None:
    output = repr(str(tmp_path / "artifact"))
    _run_without_optional_frameworks(
        f"""
from pathlib import Path
import sys

import eris
from eris.cli.main import app
from eris.core.artifacts import StructureArtifact
from typer.testing import CliRunner

blocked = {OPTIONAL_FRAMEWORKS!r}
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise SystemExit(f"optional frameworks loaded during core import: {{loaded}}")

assert eris.list_representations()
assert eris.get_representation_info("residue_contact_map")["name"] == "residue_contact_map"
artifact = eris.encode({str(PDB)!r}, representation="residue_contact_map", use_cache=False)
target = Path({output})
artifact.save(target)
loaded_artifact = StructureArtifact.load(target, strict=True)
assert loaded_artifact.meta["schema_version"] == 2

runner = CliRunner()
for args in (("--help",), ("representations", "list"), ("embedders", "list")):
    result = runner.invoke(app, list(args))
    if result.exit_code != 0:
        raise SystemExit(result.output)

try:
    eris.to_torch(artifact)
except ImportError as error:
    if "optional dependency 'torch'" not in str(error):
        raise
else:
    raise SystemExit("Torch export unexpectedly succeeded without torch")
"""
    )


def test_learned_metadata_and_dependency_probe_are_framework_lazy() -> None:
    _run_without_optional_frameworks(
        """
import sys

import eris
import eris.core.learned
from eris.core.learned.backends.gearnet import dependencies

blocked = {"torch", "torch_geometric", "torchdrug", "torch_scatter", "torch_cluster", "rdkit"}
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise SystemExit(f"optional frameworks loaded during learned metadata import: {loaded}")

assert eris.get_embedder_info("gearnet_edge")["name"] == "gearnet_edge"
assert dependencies.GEARNET_PYTHON_REQUIREMENT == ">=3.10,<3.11"
dependencies._module_exists = lambda name: name != "torch"
status = dependencies.probe_gearnet_availability()
if status.available:
    raise SystemExit("GearNet cannot be available while its frameworks are blocked")
if sys.version_info[:2] >= (3, 11):
    if "requires Python" not in status.reason:
        raise SystemExit("GearNet unavailable status must identify its Python boundary")
elif "torch" not in status.reason:
    raise SystemExit("GearNet unavailable status must identify the missing dependency")
"""
    )
