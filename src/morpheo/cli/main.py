from __future__ import annotations

import typer

from morpheo.core.registry import list_encoders

app = typer.Typer(add_completion=False)


@app.command()
def encoders() -> None:
    """List registered encoders."""
    for name in list_encoders():
        typer.echo(name)
