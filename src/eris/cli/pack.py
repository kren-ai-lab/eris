from __future__ import annotations

from pathlib import Path
from typing import Annotated, NoReturn

import typer

from eris.pipelines.pack import PackIndexError, pack_artifacts


def _usage_error(message: str) -> NoReturn:
    typer.echo(message, err=True)
    raise typer.Exit(code=2)


def pack(
    artifact_root: Annotated[Path, typer.Argument(metavar="ARTIFACT_ROOT")],
    out: Annotated[Path, typer.Option("--out", help="Output index.csv path.")],
) -> None:
    """
    Recursively validate Artifact Schema v2 directories and write a metadata-only index.
    """

    try:
        rows = pack_artifacts(artifact_root, out)
    except (PackIndexError, OSError) as error:
        _usage_error(str(error))

    typer.echo(f"Packed {len(rows)} artifacts to {out}")


__all__ = ["pack"]
