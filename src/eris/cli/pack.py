from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

import eris
from eris.pipelines.pack import PackIndexError
from eris.cli._utils import usage_error


def pack(
    artifact_root: Annotated[Path, typer.Argument(metavar="ARTIFACT_ROOT")],
    out: Annotated[Path, typer.Option("--out", help="Output index.csv path.")],
) -> None:
    """
    Recursively validate Artifact Schema directories and write a metadata-only index.
    """

    try:
        rows = eris.pack_artifacts(artifact_root, out)
    except (PackIndexError, OSError) as error:
        usage_error(str(error))

    typer.echo(f"Packed {len(rows)} artifacts to {out}")


__all__ = ["pack"]
