from __future__ import annotations

from pathlib import Path

import typer

from eris.export.pyg import save_artifact_to_pyg
from eris.export.torch import save_artifact_to_torch

app = typer.Typer(add_completion=False, help="Exports artifacts to torch or PyG.")


@app.command("to-torch")
def to_torch(
    artifact_dir: Path = typer.Option(..., "--artifact-dir", exists=True, readable=True),
    out_path: Path | None = typer.Option(None, "--out", help="Output .pt file"),
    device: str | None = typer.Option(None, "--device"),
    matrix_as_channel: bool = typer.Option(True, "--matrix-as-channel"),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Convert artifact to torch format and save it."""
    if out_path is None:
        out_path = artifact_dir / "data.pt"

    if out_path.exists() and not overwrite:
        raise ValueError("Artifact already exported as torch tensor. Use --overwrite to replace.")

    save_artifact_to_torch(
        artifact_dir,
        out_path=out_path,
        device=device,
        matrix_as_channel=matrix_as_channel,
    )

    typer.echo(f"Saved Torch artifact to {out_path}")


@app.command("to-pyg")
def to_pyg(
    artifact_dir: Path = typer.Option(..., "--artifact-dir", exists=True, readable=True),
    out_path: Path | None = typer.Option(None, "--out", help="Output .pt file"),
    device: str | None = typer.Option(None, "--device"),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    """Convert artifact to PyG format and save it."""
    if out_path is None:
        out_path = artifact_dir / "data_pyg.pt"

    if out_path.exists() and not overwrite:
        raise ValueError("Artifact already exported as torch tensor. Use --overwrite to replace.")

    save_artifact_to_pyg(
        artifact_dir,
        out_path=out_path,
        device=device,
    )

    typer.echo(f"Saved PyG artifact to {out_path}")
