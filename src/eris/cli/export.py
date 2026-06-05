from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from eris.export.torch import save_artifact_to_torch
from eris.export.pyg import save_artifact_to_pyg
from eris.pipelines import batch_export

app = typer.Typer(add_completion=False, help="Exports artifacts to torch or PyG.")


@app.command("to-torch")
def to_torch(
    artifact_dir: Path = typer.Option(..., "--artifact-dir", exists=True, readable=True),
    out_path: Optional[Path] = typer.Option(None, "--out", help="Output .pt file"),
    prefer: str = typer.Option("grid_or_graph", "--prefer"),
    device: Optional[str] = typer.Option(None, "--device"),
    matrix_as_channel: bool = typer.Option(True, "--matrix-as-channel"),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """
    Convert artifact to torch format and save it.
    """

    if out_path is None:
        out_path = artifact_dir / "data.pt"

    if out_path.exists() and not overwrite:
        raise ValueError("Artifact already exported as torch tensor. Use --overwrite to replace.")

    save_artifact_to_torch(
        artifact_dir,
        out_path=out_path,
        prefer=prefer,
        device=device,
        matrix_as_channel=matrix_as_channel,
    )

    typer.echo(f"Saved Torch artifact to {out_path}")



@app.command("to-pyg")
def to_pyg(
    artifact_dir: Path = typer.Option(..., "--artifact-dir", exists=True, readable=True),
    out_path: Optional[Path] = typer.Option(None, "--out", help="Output .pt file"),
    prefer: str = typer.Option("grid_or_graph", "--prefer"),
    device: Optional[str] = typer.Option(None, "--device"),
    matrix_as_channel: bool = typer.Option(True, "--matrix-as-channel"),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    """
    Convert artifact to PyG format and save it.
    """
    if out_path is None:
        out_path = artifact_dir / "data_pyg.pt"

    if out_path.exists() and not overwrite:
        raise ValueError("Artifact already exported as torch tensor. Use --overwrite to replace.")

    save_artifact_to_pyg(
        artifact_dir,
        out_path=out_path,
        prefer=prefer,
        device=device,
        matrix_as_channel=matrix_as_channel
    )

    typer.echo(f"Saved PyG artifact to {out_path}")


@app.command("batch-to-torch")
def batch_to_torch(
    manifest_path: Path = typer.Option(..., "--artifact-dir", exists=True, readable=True),
    out: Optional[Path] = typer.Option(None, "--out", help="Output .pt file"),
    *,
    overwrite: bool = typer.Option(False, "--overwrite"),
    fail_fast: bool = typer.Option(False, "--fail-fast"),
    prefer: str = typer.Option("grid_or_graph", "--prefer"),
    device: Optional[str] = typer.Option(None, "--device"),
    matrix_as_channel: bool = typer.Option(True, "--matrix-as-channel"),
):
    """
    Convert multiple artifacts to torch format and saves them.
    """
    torch_format = "torch"
    stats = batch_export(
        manifest_path = manifest_path,
        out=out,
        format=torch_format,
        overwrite=overwrite,
        fail_fast=fail_fast,
        prefer=prefer,
        device=device,
        matrix_as_channel=matrix_as_channel,
    )
    typer.echo(f"Batch done | {stats}")


@app.command("batch-to-pyg")
def to_pyg(
    manifest_path: Path = typer.Option(..., "--artifact-dir", exists=True, readable=True),
    out: Optional[Path] = typer.Option(None, "--out", help="Output .pt file"),
    *,
    overwrite: bool = typer.Option(False, "--overwrite"),
    fail_fast: bool = typer.Option(False, "--fail-fast"),
    prefer: str = typer.Option("grid_or_graph", "--prefer"),
    device: Optional[str] = typer.Option(None, "--device"),
    matrix_as_channel: bool = typer.Option(True, "--matrix-as-channel"),
):
    """
    Convert multiple artifacts to PyG format and saves them.
    """
    pyg_format = "pyg"
    stats = batch_export(
        manifest_path = manifest_path,
        out=out,
        format=pyg_format,
        overwrite=overwrite,
        fail_fast=fail_fast,
        prefer=prefer,
        device=device,
        matrix_as_channel=matrix_as_channel,
    )
    typer.echo(f"Batch done | {stats}")
