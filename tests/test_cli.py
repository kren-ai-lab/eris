from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

import eris
import eris.cli.embed as embed_cli
from eris.cli.main import app
from eris.core.artifacts import StructureArtifact
from eris.exceptions import ExecutionError

PDB = Path("tests/data/1AQ2.pdb")
EXPECTED_CHECKPOINT_SHA256 = "6dc11e2db6eff3c90f90672af227c647125adb7476de18a2bb860457bdb598ec"
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


def _atom_line(index: int, residue_name: str) -> str:
    x = 3.8 * (index - 1)
    y = 1.25 * ((index - 1) % 3)
    z = 0.75 * ((index - 1) % 2)
    return (
        f"ATOM  {index:5d}  CA  {residue_name:>3s} A{index:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{10.0:6.2f}           C  "
    )


def _write_gearnet_fixture(tmp_path: Path) -> Path:
    source = tmp_path / "canonical-12.pdb"
    source.write_text(
        "\n".join(
            [
                *(_atom_line(index, name) for index, name in enumerate(RESIDUES, start=1)),
                "TER",
                "END",
            ]
        )
        + "\n",
        encoding="ascii",
    )
    return source


def _checkpoint() -> Path:
    configured = os.environ.get("ERIS_GEARNET_CHECKPOINT")
    if not configured:
        pytest.skip("ERIS_GEARNET_CHECKPOINT is not configured")
    path = Path(configured)
    if not path.is_file():
        pytest.fail(f"ERIS_GEARNET_CHECKPOINT is not a file: {path}")
    return path


def _require_gearnet_stack() -> None:
    torch = pytest.importorskip("torch")
    if torch.__version__.partition("+")[0] != "2.11.0":
        pytest.skip("GearNet CLI validation requires torch==2.11.0")


def test_embed_help_exposes_public_direct_command_only():
    help_result = runner.invoke(app, ["embed", "--help"])

    assert help_result.exit_code == 0
    for expected in (
        "INPUT",
        "--representation",
        "--out",
        "--chain",
        "--model",
        "--output-mode",
        "--checkpoint",
        "--device",
        "--precision",
        "--no-cache",
    ):
        assert expected in help_result.output
    for historical in (
        "single",
        "batch",
        "--encoder",
        "--params",
        "--allow-unknown-params",
        "--id",
    ):
        assert historical not in help_result.output


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
    missing_representation = runner.invoke(app, ["embed", str(PDB), "--out", str(tmp_path)])
    missing_out = runner.invoke(app, ["embed", str(PDB), "--representation", "gearnet_edge"])

    assert missing_representation.exit_code == 2
    assert "--representation" in missing_representation.output
    assert missing_out.exit_code == 2
    assert "--out" in missing_out.output


def test_batch_from_folder_delegates_sorted_rows(monkeypatch, tmp_path):
    import eris.cli.batch as batch_cli

    folder = tmp_path / "structures"
    folder.mkdir()
    (folder / "b.pdb").write_text("", encoding="ascii")
    (folder / "a.pdb").write_text("", encoding="ascii")
    captured = {}

    def fake_batch_encode(**kwargs):
        captured.update(kwargs)
        return []

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
    assert [row.id for row in captured["rows"]] == ["a", "b"]
    assert captured["representation"] == "residue_knn_graph"
    assert captured["params"] == {"k": 8}
    assert captured["use_cache"] is True


def test_batch_from_folder_rejects_duplicate_stems(tmp_path):
    folder = tmp_path / "structures"
    (folder / "one").mkdir(parents=True)
    (folder / "two").mkdir()
    (folder / "one" / "protein.pdb").write_text("", encoding="ascii")
    (folder / "two" / "protein.pdb").write_text("", encoding="ascii")

    result = runner.invoke(
        app,
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

    assert result.exit_code == 2
    assert "duplicate id 'protein'" in result.output


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["--representation", "missing"], "Unknown embedder"),
        (["--representation", "gearnet_edge", "--precision", "wide"], "precision"),
    ],
)
def test_expected_request_errors_are_clean(args, expected, tmp_path):
    result = runner.invoke(app, ["embed", str(PDB), *args, "--out", str(tmp_path)])

    assert result.exit_code == 2
    assert expected in result.output
    assert "Traceback" not in result.output


def test_unavailable_backend_error_is_clean(monkeypatch, tmp_path):
    def fail_embed(*args, **kwargs):
        del args, kwargs
        raise ExecutionError(
            "Embedder 'gearnet_edge' backend 'gearnet-edge' is unavailable: "
            "Install the documented Eris GearNet CPU environment."
        )

    monkeypatch.setattr(embed_cli.eris, "embed", fail_embed)

    result = runner.invoke(
        app,
        [
            "embed",
            str(PDB),
            "--representation",
            "gearnet_edge",
            "--out",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 2
    assert "unavailable" in result.output
    assert "ModuleNotFoundError" not in result.output
    assert "Traceback" not in result.output
    assert not (tmp_path / "meta.json").exists()
    assert not (tmp_path / "data.npz").exists()


def test_missing_checkpoint_error_is_clean(monkeypatch, tmp_path):
    def fail_embed(*args, **kwargs):
        del args, kwargs
        raise ExecutionError(
            "No cached checkpoint found for 'mc_gearnet_edge.pth'; "
            "a local checkpoint_path is required"
        )

    monkeypatch.setattr(embed_cli.eris, "embed", fail_embed)

    result = runner.invoke(
        app,
        [
            "embed",
            str(PDB),
            "--representation",
            "gearnet_edge",
            "--out",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 2
    assert "mc_gearnet_edge.pth" in result.output
    assert "checkpoint_path is required" in result.output
    assert "Traceback" not in result.output
    assert not (tmp_path / "meta.json").exists()
    assert not (tmp_path / "data.npz").exists()


def test_historical_embed_subcommands_and_flags_are_not_public():
    root_help = runner.invoke(app, ["--help"])
    embed_help = runner.invoke(app, ["embed", "--help"])

    assert root_help.exit_code == 0
    assert embed_help.exit_code == 0
    assert "embed" in root_help.output
    assert "single" not in embed_help.output
    assert "batch" not in embed_help.output
    assert "--encoder" not in embed_help.output
    assert "--params" not in embed_help.output
    assert runner.invoke(app, ["embed", "batch"]).exit_code == 2


def test_embed_help_does_not_import_optional_learned_frameworks():
    script = """
import sys
from typer.testing import CliRunner
from eris.cli.main import app

result = CliRunner().invoke(app, ['embed', '--help'])
assert result.exit_code == 0, result.output
blocked = {'torch', 'torchdrug', 'torch_scatter', 'torch_cluster', 'rdkit'}
loaded = sorted(blocked.intersection(sys.modules))
if loaded:
    raise SystemExit(f'optional frameworks loaded: {loaded}')
"""

    subprocess.run([sys.executable, "-c", script], check=True)


def _assert_real_gearnet_artifact(
    artifact: StructureArtifact,
    *,
    output_mode: str,
    granularity: str,
) -> None:
    artifact.validate(strict=True)
    assert artifact.meta["representation_name"] == "gearnet_edge"
    assert artifact.meta["representation_family"] == "learned"
    assert artifact.meta["representation_type"] == "embedding"
    assert artifact.meta["granularity"] == granularity
    assert artifact.meta["coordinate_source"] == "model_specific"
    assert artifact.meta["parameters"]["output_mode"] == output_mode
    assert artifact.meta["model"]["checkpoint"]["sha256"] == EXPECTED_CHECKPOINT_SHA256
    assert artifact.meta["feature_keys"] == {}
    assert "valid_mask" not in artifact.data
    if output_mode in {"node", "both"}:
        assert artifact.data["node_embedding"].shape == (12, 3072)
        assert artifact.data["node_embedding"].dtype == np.float32
        assert np.isfinite(artifact.data["node_embedding"]).all()
        assert artifact.data["node_ids"].shape == (12,)
        assert artifact.data["node_ids"].dtype.kind == "U"
        assert len(set(artifact.data["node_ids"].tolist())) == 12
    if output_mode in {"global", "both"}:
        assert artifact.data["global_embedding"].shape == (3072,)
        assert artifact.data["global_embedding"].dtype == np.float32
        assert np.isfinite(artifact.data["global_embedding"]).all()


@pytest.mark.parametrize(
    ("output_mode", "granularity"),
    [("both", "mixed")],
)
def test_real_gearnet_cli_modes(output_mode, granularity, tmp_path):
    _require_gearnet_stack()
    checkpoint = _checkpoint()
    source = _write_gearnet_fixture(tmp_path)
    output = tmp_path / output_mode

    result = runner.invoke(
        app,
        [
            "embed",
            str(source),
            "--representation",
            "gearnet_edge",
            "--output-mode",
            output_mode,
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
            "--precision",
            "fp32",
            "--out",
            str(output),
            "--no-cache",
        ],
    )

    assert result.exit_code == 0, result.output
    _assert_real_gearnet_artifact(
        StructureArtifact.load(output, strict=True),
        output_mode=output_mode,
        granularity=granularity,
    )


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
    _assert_artifacts_equivalent(StructureArtifact.load(cli_output, strict=True), api_artifact)
    _assert_artifacts_equivalent(
        StructureArtifact.load(cli_output, strict=True),
        StructureArtifact.load(api_output, strict=True),
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
        StructureArtifact.load(first_output, strict=True),
        StructureArtifact.load(second_output, strict=True),
    )
    _assert_artifacts_equivalent(
        StructureArtifact.load(second_output, strict=True),
        StructureArtifact.load(third_output, strict=True),
    )
