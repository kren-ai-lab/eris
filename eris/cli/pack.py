"""Command that writes a metadata-only index over artifact directories."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

import eris
from eris._provenance import utc_now
from eris.cli._shared import write_run_provenance


def pack(
    ctx: typer.Context,
    artifact_root: Annotated[Path, typer.Argument(metavar="ARTIFACT_ROOT")],
    out: Annotated[Path, typer.Option("--out", help="Output index.csv path.")],
) -> None:
    """Recursively validate Artifact Schema directories and write a metadata-only index."""
    created_at = utc_now()
    rows = eris.pack(artifact_root, out)
    write_run_provenance(ctx, out, command="pack", created_at=created_at, inputs=[artifact_root])

    typer.echo(f"Packed {len(rows)} artifacts to {out}")


__all__ = ["pack"]
