"""Command that writes a metadata-only index over artifact directories."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

import eris


def pack(
    artifact_root: Annotated[Path, typer.Argument(metavar="ARTIFACT_ROOT")],
    out: Annotated[Path, typer.Option("--out", help="Output index.csv path.")],
) -> None:
    """Recursively validate Artifact Schema directories and write a metadata-only index."""
    rows = eris.pack(artifact_root, out)

    typer.echo(f"Packed {len(rows)} artifacts to {out}")


__all__ = ["pack"]
