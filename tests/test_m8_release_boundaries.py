from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import venv
import zipfile
from email.parser import BytesParser
from email.policy import default
from importlib.metadata import version as distribution_version
from pathlib import Path

import pytest

import eris
from eris.core.artifacts import StructureArtifact
from eris.datasets import ErisFolderDataset

ROOT = Path(__file__).resolve().parents[1]
PDB = ROOT / "tests" / "data" / "1AQ2.pdb"
PUBLIC_ROOT_NAMES = (
    "RepresentationNotFoundError",
    "__version__",
    "encode",
    "embed",
    "get_embedder_info",
    "get_representation_info",
    "list_embedders",
    "list_representations",
    "batch_encode",
    "batch_embed",
    "pack",
    "to_torch",
    "to_pyg",
)


def test_release_public_api_allowlist_and_workflow_boundaries() -> None:
    for name in PUBLIC_ROOT_NAMES:
        assert hasattr(eris, name), name

    assert eris.__version__ == distribution_version("eris")
    assert eris.list_representations()
    assert eris.get_representation_info("residue_contact_graph")["name"] == "residue_contact_graph"
    assert eris.list_embedders() == ["gearnet_edge"]
    assert eris.get_embedder_info("gearnet_edge")["name"] == "gearnet_edge"
    assert ErisFolderDataset is not None


def test_release_explicit_artifact_flow_is_schema_v2(tmp_path: Path) -> None:
    artifact = eris.encode(PDB, representation="residue_contact_map", use_cache=False)
    artifact.validate(strict=True)

    target = tmp_path / "artifact"
    artifact.save(target)
    assert {path.name for path in target.iterdir()} == {"meta.json", "data.npz"}

    loaded = StructureArtifact.load(target, strict=True)
    assert loaded.meta["schema_version"] == 2
    assert loaded.meta == artifact.meta
    assert list(loaded.data) == [output["name"] for output in loaded.meta["outputs"]]


def _create_clean_build_source(target: Path) -> None:
    target.mkdir()
    for filename in ("pyproject.toml", "README.md", "LICENSE", "CHANGELOG.md"):
        source = ROOT / filename
        if source.is_file():
            shutil.copy2(source, target / filename)
    shutil.copytree(ROOT / "src", target / "src")

    # Sentinels prove runtime output and local task/context files are not packaged.
    (target / "out").mkdir()
    (target / "out" / "generated.pt").write_bytes(b"generated")
    (target / "docs").mkdir()
    (target / "docs" / "current_task_local.md").write_text("local", encoding="utf-8")
    (target / "docs" / "development_context.md").write_text("local", encoding="utf-8")


def _built_members(wheel: Path, sdist: Path) -> tuple[list[str], list[str]]:
    with zipfile.ZipFile(wheel) as archive:
        wheel_members = archive.namelist()
    with tarfile.open(sdist, "r:gz") as archive:
        sdist_members = archive.getnames()
    return wheel_members, sdist_members


def _assert_unwanted_content_is_absent(members: list[str]) -> None:
    forbidden_parts = ("out/", ".git/", ".pytest_cache/", ".ruff_cache/")
    forbidden_names = ("current_task_", "development_context.md", ".pth")
    for member in members:
        normalized = member.replace("\\", "/")
        assert not any(part in normalized for part in forbidden_parts), normalized
        assert not any(name in normalized for name in forbidden_names), normalized


def test_release_build_contains_eris_and_excludes_local_outputs(tmp_path: Path) -> None:
    pytest.importorskip("build")
    source = tmp_path / "source"
    _create_clean_build_source(source)
    dist = tmp_path / "dist"

    subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(dist)],
        cwd=source,
        check=True,
    )

    wheel = next(dist.glob("*.whl"))
    sdist = next(dist.glob("*.tar.gz"))
    wheel_members, sdist_members = _built_members(wheel, sdist)
    assert "eris/__init__.py" in wheel_members
    assert any(member.endswith("/src/eris/__init__.py") for member in sdist_members)
    _assert_unwanted_content_is_absent(wheel_members)
    _assert_unwanted_content_is_absent(sdist_members)

    with zipfile.ZipFile(wheel) as archive:
        metadata_name = next(name for name in wheel_members if name.endswith(".dist-info/METADATA"))
        metadata = BytesParser(policy=default).parsebytes(archive.read(metadata_name))
    assert metadata["Name"] == "eris"
    assert metadata["Version"] == eris.__version__
    assert set(metadata["Requires-Python"].split(",")) == {">=3.10", "<3.14"}

    environment = tmp_path / "wheel-env"
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run(
        [python, "-m", "pip", "install", "--no-index", "--no-deps", "--force-reinstall", wheel],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        [
            python,
            "-c",
            "import eris; from importlib.metadata import version; "
            "assert eris.__version__ == version('eris'); "
            "assert eris.list_representations(); "
            "assert eris.list_embedders() == ['gearnet_edge']; "
            "assert callable(eris.batch_encode); assert callable(eris.pack); "
            "assert callable(eris.to_torch); assert callable(eris.to_pyg)",
        ],
        cwd=tmp_path,
        check=True,
    )
