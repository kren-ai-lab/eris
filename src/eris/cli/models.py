from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from eris.core.learned.checkpoint_cache import CheckpointCache
from eris.core.learned.registry import learned_registry
from eris.exceptions import CacheError, RepresentationNotFoundError
from eris.cli._utils import cache_root, usage_error

app = typer.Typer(add_completion=False, help="Inspect registered learned model assets.")
console = Console(width=240)


def _model_names() -> list[str]:
    return [
        info.representation_name
        for info in learned_registry.list_info()
        if info.model_descriptor is not None
    ]


def _model_info(name: str):
    try:
        info = learned_registry.get_info(name)
    except KeyError as error:
        raise RepresentationNotFoundError(f"Unknown model/embedder {name!r}") from error
    if info.model_descriptor is None:
        raise RepresentationNotFoundError(f"Unknown model/embedder {name!r}")
    return info


def _checkpoint_cached(info) -> bool:
    try:
        checkpoint = CheckpointCache(cache_dir=cache_root()).get(info.model_descriptor.checkpoint)
    except CacheError as error:
        del error
        usage_error(
            f"Unable to inspect checkpoint cache for model/embedder {info.representation_name!r}."
        )
    return checkpoint is not None


@app.command("list")
def list_models() -> None:
    """List registered learned model asset identities."""

    table = Table(title="Eris Models")
    table.add_column("Name", style="bold cyan")
    for name in _model_names():
        table.add_row(name)
    console.print(table)


@app.command("info")
def model_info(name: str) -> None:
    """Show identity and checkpoint cache state for a registered learned model."""

    try:
        info = _model_info(name)
    except RepresentationNotFoundError as error:
        usage_error(str(error))

    model = info.model_descriptor
    checkpoint = model.checkpoint
    table = Table(title=f"Eris Model: {info.representation_name}", show_header=False)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value", no_wrap=True)
    table.add_row("representation/embedder", info.representation_name)
    table.add_row("architecture", model.architecture)
    table.add_row("checkpoint identifier", checkpoint.identifier)
    table.add_row("checkpoint revision", checkpoint.revision or "")
    table.add_row("checkpoint source", checkpoint.source or "")
    table.add_row("checkpoint sha256", checkpoint.sha256)
    table.add_row("checkpoint cached", str(_checkpoint_cached(info)).lower())
    console.print(table)


__all__ = ["app"]
