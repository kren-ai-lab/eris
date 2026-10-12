from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

import eris
from eris.cli._utils import usage_error

app = typer.Typer(add_completion=False)
representations_app = typer.Typer(
    add_completion=False,
    help="Explore explicit structural representations.",
)
embedders_app = typer.Typer(
    add_completion=False,
    help="Explore learned structural embedders.",
)
console = Console()


def _print_names(title: str, names: list[str]) -> None:
    table = Table(title=title)
    table.add_column("Name", style="bold cyan")
    for name in names:
        table.add_row(name)
    console.print(table)


def _print_info(info: dict[str, Any]) -> None:
    table = Table(show_header=False)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value")
    for key, value in info.items():
        if isinstance(value, (dict, list)):
            rendered = Syntax(
                json.dumps(value, indent=2, ensure_ascii=False),
                "json",
                word_wrap=True,
            )
        else:
            rendered = str(value)
        table.add_row(key, rendered)
    console.print(table)


def _info_command(
    name: str,
    *,
    lookup: Callable[[str], dict[str, Any]],
) -> None:
    try:
        info = lookup(name)
    except eris.RepresentationNotFoundError as error:
        usage_error(str(error))
    _print_info(info)


@representations_app.command("list")
def list_representations() -> None:
    """List explicit structural representations."""
    _print_names("Eris Representations", eris.list_representations())


@representations_app.command("info")
def representation_info(name: str) -> None:
    """Show metadata for one explicit representation."""
    _info_command(name, lookup=eris.get_representation_info)


@embedders_app.command("list")
def list_embedders() -> None:
    """List learned structural embedders without checking availability."""
    _print_names("Eris Embedders", eris.list_embedders())


@embedders_app.command("info")
def embedder_info(name: str) -> None:
    """Show metadata and availability for one learned embedder."""
    _info_command(name, lookup=eris.get_embedder_info)


app.add_typer(representations_app, name="representations")
app.add_typer(embedders_app, name="embedders")


__all__ = ["app"]
