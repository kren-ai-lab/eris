from __future__ import annotations

import shutil
from pathlib import Path
from typing import NoReturn

import typer
from platformdirs import user_cache_dir
from rich.console import Console
from rich.table import Table

from eris.core.cache import default_cache_dir
from eris.core.learned.backends.gearnet.metadata import GEARNET_EDGE_CHECKPOINT_DESCRIPTOR
from eris.core.learned.checkpoint_cache import CheckpointCache
from eris.exceptions import CacheError

app = typer.Typer(add_completion=False, help="Cache utilities.")
console = Console()


def _usage_error(message: str) -> NoReturn:
    typer.echo(message, err=True)
    raise typer.Exit(code=2)


def _cache_root() -> Path:
    return Path(user_cache_dir("eris"))


def _count_artifact_v2_entries(cache_root: Path) -> int:
    artifact_root = cache_root / "v2"
    if not artifact_root.is_dir():
        return 0
    return sum(
        1
        for entry in artifact_root.iterdir()
        if entry.is_dir() and (entry / "meta.json").is_file() and (entry / "data.npz").is_file()
    )


def _official_checkpoint_cached(cache_root: Path) -> bool:
    try:
        checkpoint = CheckpointCache(cache_dir=cache_root).get(GEARNET_EDGE_CHECKPOINT_DESCRIPTOR)
    except CacheError as error:
        del error
        _usage_error("Unable to inspect gearnet_edge checkpoint cache.")
    return checkpoint is not None


@app.command("clear")
def clear_cache() -> None:
    """
    Remove Eris disk cache directory.
    """
    cache_dir = default_cache_dir()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        typer.echo(f"Cache cleared: {cache_dir}")
    else:
        typer.echo(f"Cache directory does not exist: {cache_dir}")


@app.command("info")
def cache_info() -> None:
    """
    Inspect Eris cache state without creating, deleting, or repairing entries.
    """

    cache_root = _cache_root()
    artifact_root = cache_root / "v2"
    checkpoint_root = cache_root / "checkpoints"

    table = Table(title="Eris Cache", show_header=False)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value")
    table.add_row("artifact cache", "present" if artifact_root.is_dir() else "absent")
    table.add_row("artifact v2 entries", str(_count_artifact_v2_entries(cache_root)))
    table.add_row("checkpoint cache", "present" if checkpoint_root.is_dir() else "absent")
    table.add_row(
        "gearnet_edge checkpoint cached",
        str(_official_checkpoint_cached(cache_root)).lower(),
    )
    console.print(table)
