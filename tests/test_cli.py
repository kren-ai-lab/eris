from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from rich.text import Text
from typer.testing import CliRunner

import eris
import eris.cli.embed as embed_cli
from eris.cli.main import EXIT_CONFIG, EXIT_INTERNAL, EXIT_WORKFLOW, app, main
from eris.core.artifacts import StructureArtifact
from eris.core.learned.backends.gearnet.metadata import GEARNET_EDGE_MODEL_DESCRIPTOR
from eris.exceptions import ExecutionError
from tests.helpers import PDB, atom_line, write_pdb

RESIDUES = (
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
)

runner = CliRunner()


def _assert_artifacts_equivalent(left: StructureArtifact, right: StructureArtifact) -> None:
    for key in (
        "representation_name",
        "model",
        "parameters",
        "provenance",
        "input",
        "feature_keys",
        "shapes",
        "dtypes",
        "counts",
    ):
        assert left.meta[key] == right.meta[key]
    assert list(left.data) == list(right.data)
    for name in left.data:
        np.testing.assert_array_equal(left.data[name], right.data[name])


def _write_gearnet_fixture(tmp_path: Path) -> Path:
    return write_pdb(
        tmp_path / "canonical-12.pdb",
        [
            atom_line(i, "CA", name, i, (3.8 * (i - 1), 1.25 * ((i - 1) % 3), 0.75 * ((i - 1) % 2)))
            for i, name in enumerate(RESIDUES, start=1)
        ],
    )


def _checkpoint() -> Path:
    configured = os.environ.get("ERIS_GEARNET_CHECKPOINT")
    if not configured:
        pytest.skip("ERIS_GEARNET_CHECKPOINT is not configured")
    path = Path(configured)
    if not path.is_file():
        pytest.fail(f"ERIS_GEARNET_CHECKPOINT is not a file: {path}")
    return path


def _require_gearnet_stack() -> None:
    pytest.importorskip("torch")


def test_direct_embed_command_delegates_to_public_api(monkeypatch, tmp_path):
    calls = []

    def fake_embed(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(
            meta={
                "representation_name": kwargs["representation"],
                "parameters": {"output_mode": kwargs["output_mode"]},
            }
        )

    monkeypatch.setattr(embed_cli.eris, "embed", fake_embed)
    checkpoint = tmp_path / "checkpoint.pth"

    result = runner.invoke(
        app,
        [
            "embed",
            "input.pdb",
            "--representation",
            "gearnet_edge",
            "--chain",
            "A",
            "--model",
            "1",
            "--output-mode",
            "node",
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
            "--precision",
            "fp32",
            "--out",
            str(tmp_path / "artifact"),
            "--no-cache",
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        (
            (),
            {
                "input": "input.pdb",
                "representation": "gearnet_edge",
                "output_mode": "node",
                "device": "cpu",
                "precision": "fp32",
                "output": tmp_path / "artifact",
                "use_cache": False,
                "chain_id": "A",
                "model_index": 1,
                "checkpoint_path": checkpoint,
            },
        )
    ]


def test_absent_chain_model_and_checkpoint_are_not_forwarded(monkeypatch, tmp_path):
    captured = {}

    def fake_embed(*args, **kwargs):
        del args
        captured.update(kwargs)
        return SimpleNamespace(
            meta={
                "representation_name": kwargs["representation"],
                "parameters": {"output_mode": kwargs["output_mode"]},
            }
        )

    monkeypatch.setattr(embed_cli.eris, "embed", fake_embed)

    result = runner.invoke(
        app,
        [
            "embed",
            "input.pdb",
            "--representation",
            "gearnet_edge",
            "--out",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert captured["output_mode"] == "both"
    assert captured["device"] == "auto"
    assert captured["precision"] == "fp32"
    assert captured["use_cache"] is True
    assert "chain_id" not in captured
    assert "model_index" not in captured
    assert "checkpoint_path" not in captured


def test_required_representation_and_out_are_enforced(tmp_path):
    missing_representation = runner.invoke(
        app,
        ["embed", str(PDB), "--out", str(tmp_path)],
        color=False,
        terminal_width=200,
    )
    missing_out = runner.invoke(
        app,
        ["embed", str(PDB), "--representation", "gearnet_edge"],
        color=False,
        terminal_width=200,
    )
    missing_representation_output = Text.from_ansi(missing_representation.output).plain
    missing_out_output = Text.from_ansi(missing_out.output).plain

    assert missing_representation.exit_code == 2
    assert "--representation" in missing_representation_output
    assert missing_out.exit_code == 2
    assert "--out" in missing_out_output


def test_batch_from_folder_delegates_sorted_rows(monkeypatch, tmp_path):
    import eris.cli.batch as batch_cli

    folder = tmp_path / "structures"
    folder.mkdir()
    (folder / "b.pdb").write_text("", encoding="ascii")
    (folder / "a.pdb").write_text("", encoding="ascii")
    captured = {}

    def fake_batch_encode(manifest, **kwargs):
        captured.update(kwargs, manifest=manifest)
        return pl.DataFrame(schema={"status": pl.String})

    monkeypatch.setattr(batch_cli.eris, "batch_encode", fake_batch_encode)

    result = runner.invoke(
        app,
        [
            "batch",
            "from-folder",
            str(folder),
            "--representation",
            "residue_knn_graph",
            "--out",
            str(tmp_path / "artifacts"),
            "--pattern",
            "*.pdb",
            "--params",
            '{"k":8}',
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["manifest"]["id"].to_list() == ["a", "b"]
    assert captured["representation"] == "residue_knn_graph"
    assert captured["params"] == {"k": 8}
    assert captured["use_cache"] is True


def test_batch_from_folder_rejects_duplicate_stems(capsys, tmp_path):
    folder = tmp_path / "structures"
    (folder / "one").mkdir(parents=True)
    (folder / "two").mkdir()
    (folder / "one" / "protein.pdb").write_text("", encoding="ascii")
    (folder / "two" / "protein.pdb").write_text("", encoding="ascii")

    code = main(
        [
            "batch",
            "from-folder",
            str(folder),
            "--recursive",
            "--representation",
            "residue_knn_graph",
            "--out",
            str(tmp_path / "artifacts"),
        ],
    )

    assert code == EXIT_WORKFLOW
    assert "duplicate id 'protein'" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["--representation", "missing"], "Unknown embedder"),
        (["--representation", "gearnet_edge", "--precision", "wide"], "precision"),
    ],
)
def test_expected_request_errors_are_clean(args, expected, capsys, tmp_path):
    code = main(["embed", str(PDB), *args, "--out", str(tmp_path)])

    err = capsys.readouterr().err
    assert code == EXIT_CONFIG
    assert expected in err
    assert "Traceback" not in err


def test_unavailable_backend_error_is_clean(monkeypatch, capsys, tmp_path):
    def fail_embed(*args, **kwargs):
        del args, kwargs
        raise ExecutionError(
            "Embedder 'gearnet_edge' backend 'gearnet-edge' is unavailable: "
            "Install the documented Eris GearNet CPU environment."
        )

    monkeypatch.setattr(embed_cli.eris, "embed", fail_embed)

    code = main(
        [
            "embed",
            str(PDB),
            "--representation",
            "gearnet_edge",
            "--out",
            str(tmp_path),
        ],
    )

    output = capsys.readouterr().err
    assert code == EXIT_WORKFLOW
    assert "unavailable" in output
    assert "ModuleNotFoundError" not in output
    assert "Traceback" not in output


def _assert_real_gearnet_artifact(
    artifact: StructureArtifact,
    *,
    output_mode: str,
    granularity: str,
) -> None:
    artifact.validate()
    assert artifact.meta["representation_name"] == "gearnet_edge"
    assert artifact.meta["representation_family"] == "learned"
    assert artifact.meta["representation_type"] == "embedding"
    assert artifact.meta["granularity"] == granularity
    assert artifact.meta["coordinate_source"] == "model_specific"
    assert artifact.meta["parameters"]["output_mode"] == output_mode
    assert artifact.meta["model"]["checkpoint"]["sha256"] == GEARNET_EDGE_MODEL_DESCRIPTOR.checkpoint.sha256
    assert "valid_mask" not in artifact.data
    if output_mode == "both":
        assert artifact.data["node_embedding"].shape[1] == artifact.data["global_embedding"].shape[0]
    if output_mode in {"node", "both"}:
        assert artifact.data["node_embedding"].shape[0] == len(RESIDUES)
        assert artifact.data["node_embedding"].dtype == np.float32
        assert np.isfinite(artifact.data["node_embedding"]).all()
        assert artifact.data["node_ids"].dtype.kind == "U"
        assert len(set(artifact.data["node_ids"].tolist())) == len(RESIDUES)
    if output_mode in {"global", "both"}:
        assert artifact.data["global_embedding"].dtype == np.float32
        assert np.isfinite(artifact.data["global_embedding"]).all()


def test_real_gearnet_cli_and_public_api_are_equivalent(tmp_path):
    _require_gearnet_stack()
    checkpoint = _checkpoint()
    source = _write_gearnet_fixture(tmp_path)
    cli_output = tmp_path / "cli"
    api_output = tmp_path / "api"

    result = runner.invoke(
        app,
        [
            "embed",
            str(source),
            "--representation",
            "gearnet_edge",
            "--output-mode",
            "both",
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
            "--precision",
            "fp32",
            "--out",
            str(cli_output),
            "--no-cache",
        ],
    )

    assert result.exit_code == 0, result.output
    api_artifact = eris.embed(
        str(source),
        representation="gearnet_edge",
        output_mode="both",
        checkpoint_path=checkpoint,
        device="cpu",
        precision="fp32",
        output=api_output,
        use_cache=False,
    )
    _assert_real_gearnet_artifact(StructureArtifact.load(cli_output), output_mode="both", granularity="mixed")
    _assert_artifacts_equivalent(StructureArtifact.load(cli_output), api_artifact)
    _assert_artifacts_equivalent(
        StructureArtifact.load(cli_output),
        StructureArtifact.load(api_output),
    )


def test_real_gearnet_checkpoint_cache_and_artifact_cache_work_through_cli(tmp_path):
    _require_gearnet_stack()
    checkpoint = _checkpoint()
    source = _write_gearnet_fixture(tmp_path)
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    third_output = tmp_path / "third"

    first = runner.invoke(
        app,
        [
            "embed",
            str(source),
            "--representation",
            "gearnet_edge",
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
            "--precision",
            "fp32",
            "--out",
            str(first_output),
        ],
    )

    second = runner.invoke(
        app,
        [
            "embed",
            str(source),
            "--representation",
            "gearnet_edge",
            "--device",
            "cpu",
            "--precision",
            "fp32",
            "--out",
            str(second_output),
        ],
    )

    third = runner.invoke(
        app,
        [
            "embed",
            str(source),
            "--representation",
            "gearnet_edge",
            "--device",
            "cpu",
            "--precision",
            "fp32",
            "--out",
            str(third_output),
        ],
    )

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert third.exit_code == 0, third.output
    _assert_artifacts_equivalent(
        StructureArtifact.load(first_output),
        StructureArtifact.load(second_output),
    )
    _assert_artifacts_equivalent(
        StructureArtifact.load(second_output),
        StructureArtifact.load(third_output),
    )


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ExecutionError("backend failed"), EXIT_WORKFLOW),
        (FileNotFoundError("missing.pdb"), EXIT_WORKFLOW),
        (RuntimeError("bug [x]"), EXIT_INTERNAL),
    ],
)
def test_main_maps_failures_to_exit_codes(monkeypatch, capsys, tmp_path, error, expected):
    def fail(**kwargs):
        del kwargs
        raise error

    monkeypatch.setattr(eris, "encode", fail)

    code = main(["encode", str(PDB), "--representation", "residue_knn_graph", "--out", str(tmp_path)])

    assert code == expected
    assert str(error) in capsys.readouterr().err


def test_python_module_entry_point_returns_exit_code():
    result = subprocess.run(
        [sys.executable, "-m", "eris", "--version"], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0
    assert result.stdout.strip() == eris.__version__
