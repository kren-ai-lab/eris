"""Root command-line application."""

from __future__ import annotations

import typer

import eris
from eris.cli.batch import app as batch_app
from eris.cli.cache import app as cache_app
from eris.cli.discovery import app as discovery_app
from eris.cli.embed import embed as embed_command
from eris.cli.encode import encode as encode_command
from eris.cli.export import app as torch_app
from eris.cli.info_cmd import app as info_app
from eris.cli.models import app as models_app
from eris.cli.pack import pack as pack_command

app = typer.Typer(add_completion=False)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(eris.__version__)
        raise typer.Exit


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the Eris version.",
    ),
) -> None:
    """Eris command-line interface."""
    del version


# register subcommands
app.add_typer(cache_app, name="cache")
app.add_typer(batch_app, name="batch")
app.command("encode")(encode_command)
app.command("embed")(embed_command)
app.command("pack")(pack_command)
app.add_typer(info_app, name="info")
app.add_typer(models_app, name="models")
app.add_typer(discovery_app)
app.add_typer(torch_app, name="export")
