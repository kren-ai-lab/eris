"""Root command-line application."""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.markup import escape

import eris
from eris.cli.batch import app as batch_app
from eris.cli.cache import app as cache_app
from eris.cli.discovery import app as discovery_app
from eris.cli.embed import embed as embed_command
from eris.cli.encode import encode as encode_command
from eris.cli.export import app as torch_app
from eris.cli.features import features as features_command
from eris.cli.info_cmd import app as info_app
from eris.cli.models import app as models_app
from eris.cli.pack import pack as pack_command
from eris.cli.validate import validate as validate_command
from eris.exceptions import ConfigError, ErisError, RepresentationNotFoundError

if TYPE_CHECKING:
    from collections.abc import Sequence

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_WORKFLOW = 3
EXIT_INTERNAL = 4

app = typer.Typer(add_completion=False)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(eris.__version__)
        raise typer.Exit


@app.callback()
def _root(
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
app.command("validate-config")(validate_command)
app.command("features")(features_command)
app.add_typer(info_app, name="info")
app.add_typer(models_app, name="models")
app.add_typer(discovery_app)
app.add_typer(torch_app, name="export")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and map failures to the documented exit codes."""
    error_console = Console(stderr=True, soft_wrap=True)
    try:
        # Click exits with 0 on success and 2 on usage errors, which is EXIT_CONFIG.
        typer.main.get_command(app).main(args=None if argv is None else list(argv), prog_name="eris")
    except SystemExit as exc:
        if exc.code is None:
            return EXIT_OK
        return exc.code if isinstance(exc.code, int) else EXIT_INTERNAL
    except (ConfigError, RepresentationNotFoundError) as exc:
        error_console.print(f"[bold red]Configuration error[/bold red]\n{escape(str(exc))}")
        return EXIT_CONFIG
    except ErisError as exc:
        error_console.print(f"[bold red]eris workflow failed[/bold red]\n{escape(str(exc))}")
        return EXIT_WORKFLOW
    except (OSError, ValueError) as exc:
        error_console.print(f"[bold red]Error[/bold red]\n{escape(str(exc))}")
        return EXIT_WORKFLOW
    except Exception as exc:  # noqa: BLE001 - last-resort CLI boundary
        error_console.print(f"[bold red]Unexpected error[/bold red]\n{escape(str(exc))}")
        return EXIT_INTERNAL
    return EXIT_OK
