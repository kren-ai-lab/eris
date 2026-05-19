from __future__ import annotations

import json
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

from eris.core.registry import list_encoder_info

app = typer.Typer(add_completion=False, help="List registered encoders.")
console = Console()


@app.command()
def list(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show params_schema."),
) -> None:
    infos = sorted(list_encoder_info(), key=lambda x: x.name)

    table = Table(title="Eris Encoders")
    table.add_column("Name", style="bold cyan")
    table.add_column("Type")
    table.add_column("Level")
    table.add_column("Description")

    for info in infos:
        table.add_row(info.name, info.type, info.level, info.description or "")

    console.print(table)

    if verbose:
        console.print()
        for info in infos:
            if not info.params_schema:
                continue
            schema_json = json.dumps(info.params_schema, indent=2, ensure_ascii=False)
            console.print(
                Panel(
                    Syntax(schema_json, "json", word_wrap=True),
                    title=f"[bold]{info.name}[/bold] params_schema",
                    expand=False,
                )
            )
