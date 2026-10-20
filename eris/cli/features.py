"""Command that writes a structure-level feature table for tabular machine learning."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

import eris
from eris._provenance import utc_now
from eris.cli._shared import CONFIG_OPTION, write_run_provenance
from eris.pipelines.manifest import write_atomically


def features(
    ctx: typer.Context,
    index: Annotated[Path, typer.Argument(metavar="INDEX")],
    out: Annotated[Path, typer.Option("--out", help="Output CSV path.")],
    source: Annotated[
        str | None,
        typer.Option("--source", help="global_embedding, pooled_attr or graph_attr (default: auto)."),
    ] = None,
    config: Annotated[Path | None, CONFIG_OPTION] = None,
) -> None:
    """Write one row of structure-level features per artifact in a packed index."""
    del config
    created_at = utc_now()
    table = eris.feature_table(index, source=source)
    write_atomically(out, table.write_csv)
    write_run_provenance(ctx, out, command="features", created_at=created_at, inputs=[index])
    typer.echo(f"Wrote {table.height} rows x {table.width - 1} features to {out}")


__all__ = ["features"]
