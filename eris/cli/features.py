"""Command that writes a structure-level feature table for tabular machine learning."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

import eris
from eris.pipelines.manifest import write_atomically


def features(
    index: Annotated[Path, typer.Argument(metavar="INDEX", exists=True, readable=True)],
    out: Annotated[Path, typer.Option("--out", help="Output CSV path.")],
    source: Annotated[
        str | None,
        typer.Option("--source", help="global_embedding, pooled_attr or graph_attr (default: auto)."),
    ] = None,
) -> None:
    """Write one row of structure-level features per artifact in a packed index."""
    table = eris.feature_table(index, source=source)
    write_atomically(out, table.write_csv)
    typer.echo(f"Wrote {table.height} rows x {table.width - 1} features to {out}")


__all__ = ["features"]
