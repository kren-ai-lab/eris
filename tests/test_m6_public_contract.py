from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

import eris
import eris.cli.cache as cache_cli
import eris.cli.models as models_cli
from eris.cli.main import app
from eris.core.artifacts import StructureArtifact

PDB = Path("tests/data/1AQ2.pdb")
README = Path("README.md")
OPTIONAL_FRAMEWORKS = {"torch", "torchdrug", "torch_scatter", "torch_cluster", "rdkit"}
PUBLIC_ROOT_API = [
    "RepresentationNotFoundError",
    "__version__",
    "encode",
    "embed",
    "get_embedder_info",
    "get_representation_info",
    "list_embedders",
    "list_representations",
]
CANONICAL_REPRESENTATIONS = [
    "atom_knn_graph",
    "atom_point_cloud",
    "residue_contact_graph",
    "residue_contact_map",
    "residue_contact_map_bins",
    "residue_distogram",
    "residue_distogram_bins",
    "residue_knn_graph",
    "residue_point_cloud",
    "residue_voxel_grid",
]

runner = CliRunner()


def _schema_v1_artifact(target: Path) -> None:
    target.mkdir()
    np.savez_compressed(target / "data.npz", pos=np.zeros((1, 3), dtype=np.float32))
    (target / "meta.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "encoder_name": "legacy_cloud",
                "encoder_version": "0.1.0",
                "encoder_fingerprint": "legacy",
                "type": "cloud",
                "level": "residue",
                "params": {},
                "input": "legacy",
                "feature_keys": {},
                "shapes": {"pos": [1, 3]},
                "dtypes": {"pos": "float32"},
                "counts": {"n_nodes": 1},
            }
        ),
        encoding="utf-8",
    )


def _assert_no_traceback(result) -> None:
    assert "Traceback" not in result.output


def test_m6_root_python_api_is_public_surface_only():
    assert eris.__all__ == PUBLIC_ROOT_API
    assert eris.list_representations() == CANONICAL_REPRESENTATIONS
    assert eris.list_embedders() == ["gearnet_edge"]
    assert eris.get_representation_info("residue_knn_graph")["name"] == "residue_knn_graph"
    assert eris.get_embedder_info("gearnet_edge")["name"] == "gearnet_edge"

    for internal_name in (
        "Registry",
        "get_encoder",
        "BaseEncoder",
        "ResidueKNNGraph",
        "StructuralEmbedderRunner",
        "StructuralEmbedderConfig",
        "StructuralExecutionConfig",
        "learned_registry",
        "GearNetEdgeBackend",
        "GearNetEdgeConfig",
        "CheckpointCache",
        "CheckpointDescriptor",
        "StructuralModelDescriptor",
    ):
        assert not hasattr(eris, internal_name)


def test_m6_readme_level_python_explicit_quickstart(tmp_path):
    artifact = eris.encode(
        str(PDB),
        representation="residue_knn_graph",
        k=8,
        use_cache=False,
    )

    artifact.validate(strict=True)
    assert artifact.meta["schema_version"] == 2
    assert artifact.meta["representation_name"] == "residue_knn_graph"
    assert artifact.meta["parameters"]["k"] == 8

    output = tmp_path / "artifact_knn"
    artifact.save(output)
    loaded = StructureArtifact.load(output, strict=True)
    assert loaded.meta == artifact.meta
    assert list(loaded.data) == list(artifact.data)


def test_m6_readme_level_cli_core_quickstart(monkeypatch, tmp_path):
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(cache_cli, "user_cache_dir", lambda app_name: str(cache_root))
    output = tmp_path / "artifact_knn"

    commands = [
        ["--version"],
        ["representations", "list"],
        ["representations", "info", "residue_knn_graph"],
        [
            "encode",
            str(PDB),
            "--representation",
            "residue_knn_graph",
            "--params",
            '{"k": 8}',
            "--out",
            str(output),
            "--no-cache",
        ],
        ["info", "validate", str(output)],
        ["info", "show", str(output)],
        ["cache", "info"],
        ["cache", "clear", "--target", "artifacts", "--yes"],
    ]

    for command in commands:
        result = runner.invoke(app, command)
        assert result.exit_code == 0, (command, result.output)

    artifact = StructureArtifact.load(output, strict=True)
    assert artifact.meta["representation_name"] == "residue_knn_graph"
    assert artifact.meta["parameters"]["k"] == 8


def test_m6_readme_level_learned_discovery_and_model_info(monkeypatch, tmp_path):
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(models_cli, "user_cache_dir", lambda app_name: str(cache_root))

    commands = [
        ["embedders", "list"],
        ["embedders", "info", "gearnet_edge"],
        ["models", "list"],
        ["models", "info", "gearnet_edge"],
    ]

    for command in commands:
        result = runner.invoke(app, command)
        assert result.exit_code == 0, (command, result.output)
        assert "gearnet_edge" in result.output


def test_m6_cli_help_contract_exposes_only_public_m6_hierarchy():
    for command in (
        ["--help"],
        ["representations", "--help"],
        ["embedders", "--help"],
        ["encode", "--help"],
        ["embed", "--help"],
        ["info", "--help"],
        ["models", "--help"],
        ["cache", "--help"],
        ["cache", "clear", "--help"],
    ):
        result = runner.invoke(app, command)
        assert result.exit_code == 0, (command, result.output)

    root_help = runner.invoke(app, ["--help"])
    clear_help = runner.invoke(app, ["cache", "clear", "--help"])
    assert "representations" in root_help.output
    assert "embedders" in root_help.output
    assert "encode" in root_help.output
    assert "embed" in root_help.output
    assert "info" in root_help.output
    assert "models" in root_help.output
    assert "cache" in root_help.output
    assert "encoders" not in root_help.output
    assert "--target" in clear_help.output
    assert "artifacts" in clear_help.output
    assert "checkpoints" in clear_help.output
    assert "all" in clear_help.output

    for historical in (
        ["list"],
        ["encoders", "list"],
        ["encode", "single"],
        ["encode", "batch"],
        ["embed", "single"],
        ["embed", "batch"],
    ):
        assert runner.invoke(app, historical).exit_code == 2


def test_m6_error_contracts_are_clean(monkeypatch, tmp_path):
    monkeypatch.setattr(cache_cli, "user_cache_dir", lambda app_name: str(tmp_path / "cache"))
    v1 = tmp_path / "schema-v1"
    _schema_v1_artifact(v1)

    cases = [
        (["representations", "info", "missing"], "Unknown representation"),
        (["embedders", "info", "missing"], "Unknown embedder"),
        (["models", "info", "missing"], "Unknown model/embedder"),
        (
            [
                "encode",
                str(PDB),
                "--representation",
                "missing",
                "--out",
                str(tmp_path / "missing"),
            ],
            "Unknown representation",
        ),
        (
            [
                "encode",
                str(PDB),
                "--representation",
                "residue_knn_graph",
                "--params",
                '{"unknown": true}',
                "--out",
                str(tmp_path / "bad-params"),
            ],
            "Unknown parameters",
        ),
        (["info", "validate", str(tmp_path / "does-not-exist")], "does not exist"),
        (["info", "validate", str(v1)], "Schema v1 is no longer supported"),
        (["cache", "clear", "--target", "invalid", "--yes"], "Invalid value"),
    ]

    for command, expected in cases:
        result = runner.invoke(app, command)
        assert result.exit_code == 2, (command, result.output)
        assert expected in result.output
        _assert_no_traceback(result)

    with pytest.raises(eris.RepresentationNotFoundError):
        eris.get_representation_info("missing")
    with pytest.raises(eris.RepresentationNotFoundError):
        eris.get_embedder_info("missing")
    with pytest.raises(eris.RepresentationNotFoundError):
        eris.encode(PDB, representation="missing")


def test_m6_core_import_and_cli_discovery_are_framework_lazy():
    script = """
import sys
import eris
from typer.testing import CliRunner
from eris.cli.main import app

assert eris.list_representations() == [
    "atom_knn_graph",
    "atom_point_cloud",
    "residue_contact_graph",
    "residue_contact_map",
    "residue_contact_map_bins",
    "residue_distogram",
    "residue_distogram_bins",
    "residue_knn_graph",
    "residue_point_cloud",
    "residue_voxel_grid",
]
assert eris.list_embedders() == ["gearnet_edge"]
assert eris.get_embedder_info("gearnet_edge")["name"] == "gearnet_edge"

runner = CliRunner()
for args in (
    ["--help"],
    ["representations", "list"],
    ["embedders", "list"],
    ["models", "list"],
    ["cache", "--help"],
):
    result = runner.invoke(app, args)
    assert result.exit_code == 0, (args, result.output)

blocked = {"torch", "torchdrug", "torch_scatter", "torch_cluster", "rdkit"}
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise SystemExit(f"optional frameworks loaded: {loaded}")
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_m6_readme_examples_match_public_contract():
    text = README.read_text(encoding="utf-8")

    for name in CANONICAL_REPRESENTATIONS:
        assert f"`{name}`" in text
    for snippet in (
        "python -m pip install .",
        'python -m pip install -e ".[dev]"',
        'python -m pip install ".[gearnet]"',
        "eris representations list",
        "eris representations info residue_knn_graph",
        "eris encode protein.pdb",
        "--representation residue_knn_graph",
        "eris info validate artifact_knn",
        "eris info show artifact_knn",
        "eris embedders list",
        "eris embedders info gearnet_edge",
        "eris models list",
        "eris models info gearnet_edge",
        "eris embed protein.pdb",
        "--representation gearnet_edge",
        "--checkpoint path/to/mc_gearnet_edge.pth",
        "eris cache info",
        "eris cache clear --target artifacts",
        "eris cache clear --target checkpoints",
        "eris cache clear --target all",
    ):
        assert snippet in text
    for obsolete in ("eris encode single", "eris encode batch", "eris embed single"):
        assert obsolete not in text
