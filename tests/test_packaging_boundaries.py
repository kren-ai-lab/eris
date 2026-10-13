from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PDB = ROOT / "tests" / "data" / "1AQ2.pdb"
OPTIONAL_FRAMEWORKS = {
    "torch",
    "torch_geometric",
    "torchdrug",
    "torch_scatter",
    "torch_cluster",
    "rdkit",
    "transformers",
}
GEARNET_NUMERICAL = ("backend", "loading", "native_graph", "native_model", "preprocessing")


def _run(script: str) -> None:
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True)  # noqa: S603


@pytest.mark.parametrize(
    ("entry_point", "lazy_extra"),
    [
        (
            # Core import surface, discovery and an explicit encode (Biotite is feature-local).
            f"""
import eris, eris.export, eris.core.learned.execution
from eris.datasets import ErisFolderDataset
assert eris.list_representations()
assert eris.list_embedders() == ["gearnet_edge"]
assert eris.get_embedder_info("gearnet_edge")["name"] == "gearnet_edge"
eris.encode({str(PDB)!r}, representation="residue_contact_map", use_cache=False)
""",
            {"biotite"},
        ),
        (
            """
from typer.testing import CliRunner
from eris.cli.main import app
for args in (["--help"], ["embed", "--help"], ["representations", "list"], ["embedders", "list"]):
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
""",
            set(),
        ),
    ],
    ids=["api", "cli"],
)
def test_entry_points_do_not_import_optional_frameworks(entry_point: str, lazy_extra: set[str]) -> None:
    _run(
        entry_point
        + f"""
import sys
loaded = sorted({OPTIONAL_FRAMEWORKS | lazy_extra!r}.intersection(sys.modules))
loaded += sorted(
    m for m in sys.modules
    if m.startswith("eris.core.learned.backends.gearnet.") and m.rsplit(".", 1)[-1] in {GEARNET_NUMERICAL!r}
)
if loaded:
    raise SystemExit(f"eagerly imported: {{loaded}}")
"""
    )


def test_core_workflow_and_cli_work_without_optional_frameworks(tmp_path: Path) -> None:
    blocked = OPTIONAL_FRAMEWORKS | {"biotite"}
    _run(
        f"""
import builtins
import importlib.util

_real_import = builtins.__import__

def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.partition('.')[0] in {blocked!r}:
        raise ImportError(f"optional framework deliberately unavailable: {{name}}")
    return _real_import(name, globals, locals, fromlist, level)

builtins.__import__ = _blocked_import

from pathlib import Path

import eris
from eris.cli.main import app
from eris.core.artifacts import StructureArtifact
from eris.core.learned.backends.gearnet import dependencies
from typer.testing import CliRunner

artifact = eris.encode({str(PDB)!r}, representation="residue_contact_map", use_cache=False)
target = Path({str(tmp_path / "artifact")!r})
artifact.save(target)
StructureArtifact.load(target)

for args in (("--help",), ("representations", "list"), ("embedders", "list")):
    result = CliRunner().invoke(app, list(args))
    if result.exit_code != 0:
        raise SystemExit(result.output)

try:
    eris.to_torch(artifact)
except ImportError as error:
    if "optional dependency 'torch'" not in str(error):
        raise
else:
    raise SystemExit("Torch export unexpectedly succeeded without torch")

importlib.util.find_spec = lambda name, package=None: None
status = dependencies.probe_gearnet_availability()
if status.available or "torch" not in status.reason:
    raise SystemExit(f"GearNet must report missing torch, got {{status}}")
"""
    )
