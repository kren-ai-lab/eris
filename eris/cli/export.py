"""Commands that export artifacts to Torch and PyG."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from eris._provenance import utc_now
from eris.cli._shared import CONFIG_OPTION, write_run_provenance
from eris.export.pyg import save_artifact_to_pyg
from eris.export.torch import save_artifact_to_torch

app = typer.Typer(add_completion=False, help="Exports artifacts to torch or PyG.")


@app.command("to-torch")
def to_torch(
    ctx: typer.Context,
    artifact_dir: Annotated[Path, typer.Option("--artifact-dir")],
    out_path: Annotated[Path | None, typer.Option("--out", help="Output .pt file")] = None,
    device: str | None = typer.Option(None, "--device"),
    matrix_as_channel: bool = typer.Option(True, "--matrix-as-channel"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    config: Annotated[Path | None, CONFIG_OPTION] = None,
) -> None:
    """Convert artifact to torch format and save it."""
    del config
    created_at = utc_now()
    if out_path is None:
        out_path = artifact_dir / "data.pt"

    if out_path.exists() and not overwrite:
        raise typer.BadParameter("output exists; use --overwrite to replace.", param_hint="--overwrite")

    save_artifact_to_torch(
        artifact_dir,
        out_path=out_path,
        device=device,
        matrix_as_channel=matrix_as_channel,
    )

    write_run_provenance(
        ctx, out_path, command="export to-torch", created_at=created_at, inputs=[artifact_dir]
    )
    typer.echo(f"Saved Torch artifact to {out_path}")


@app.command("to-pyg")
def to_pyg(
    ctx: typer.Context,
    artifact_dir: Annotated[Path, typer.Option("--artifact-dir")],
    out_path: Annotated[Path | None, typer.Option("--out", help="Output .pt file")] = None,
    device: str | None = typer.Option(None, "--device"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    config: Annotated[Path | None, CONFIG_OPTION] = None,
) -> None:
    """Convert artifact to PyG format and save it."""
    del config
    created_at = utc_now()
    if out_path is None:
        out_path = artifact_dir / "data_pyg.pt"

    if out_path.exists() and not overwrite:
        raise typer.BadParameter("output exists; use --overwrite to replace.", param_hint="--overwrite")

    save_artifact_to_pyg(
        artifact_dir,
        out_path=out_path,
        device=device,
    )

    write_run_provenance(ctx, out_path, command="export to-pyg", created_at=created_at, inputs=[artifact_dir])
    typer.echo(f"Saved PyG artifact to {out_path}")
