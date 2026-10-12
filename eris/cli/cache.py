from __future__ import annotations

import shutil
from enum import Enum
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from eris.cli._utils import cache_root as default_cache_root
from eris.cli._utils import usage_error
from eris.core.learned.backends.gearnet.metadata import GEARNET_EDGE_CHECKPOINT_DESCRIPTOR
from eris.core.learned.checkpoint_cache import CheckpointCache
from eris.exceptions import CacheError

app = typer.Typer(add_completion=False, help="Cache utilities.")
console = Console()


class CacheClearTarget(str, Enum):
    ARTIFACTS = "artifacts"
    CHECKPOINTS = "checkpoints"
    ALL = "all"


def _count_artifact_entries(cache_root: Path) -> int:
    artifact_root = cache_root / "artifacts"
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
        usage_error("Unable to inspect gearnet_edge checkpoint cache.")
    return checkpoint is not None


def _ensure_cache_child(cache_root: Path, target: Path) -> None:
    cache_root_resolved = cache_root.resolve(strict=False)
    target_resolved = target.resolve(strict=False)
    if not target_resolved.is_relative_to(cache_root_resolved):
        raise CacheError(f"Refusing to clear cache namespace outside Eris cache root: {target}")


def _remove_cache_namespace(cache_root: Path, target: Path, label: str) -> bool:
    if not target.exists():
        return False
    if not target.is_dir():
        raise CacheError(f"Expected Eris {label} cache namespace to be a directory: {target}")
    _ensure_cache_child(cache_root, target)
    try:
        shutil.rmtree(target)
    except OSError as error:
        raise CacheError(f"Unable to clear Eris {label} cache at {target}: {error}") from error
    return True


def _target_namespaces(cache_root: Path, target: CacheClearTarget) -> list[tuple[str, Path]]:
    artifact_namespace = ("artifact", cache_root / "artifacts")
    checkpoint_namespace = ("checkpoint", cache_root / "checkpoints")
    if target is CacheClearTarget.ARTIFACTS:
        return [artifact_namespace]
    if target is CacheClearTarget.CHECKPOINTS:
        return [checkpoint_namespace]
    return [artifact_namespace, checkpoint_namespace]


def _remove_empty_cache_root(cache_root: Path) -> bool:
    if not cache_root.is_dir():
        return False
    if any(cache_root.iterdir()):
        return False
    try:
        cache_root.rmdir()
    except OSError as error:
        raise CacheError(
            f"Unable to remove empty Eris cache root at {cache_root}: {error}"
        ) from error
    return True


@app.command("clear")
def clear_cache(
    target: CacheClearTarget = typer.Option(  # noqa: B008
        ...,
        "--target",
        help=(
            "Cache namespace to clear: artifacts=cached representation results; "
            "checkpoints=learned model checkpoint assets; all=both known namespaces."
        ),
    ),
    yes: bool = typer.Option(  # noqa: B008
        False,
        "--yes",
        "-y",
        help="Confirm cache deletion without an interactive prompt.",
    ),
) -> None:
    """
    Clear one known Eris cache namespace.
    """
    cache_root = default_cache_root()
    if not yes and not typer.confirm(
        f"Clear Eris {target.value} cache? This cannot be undone.",
        default=False,
    ):
        typer.echo("Cache clear cancelled.")
        return

    if not cache_root.exists():
        typer.echo(f"No Eris cache root exists; nothing to clear for target '{target.value}'.")
        return
    if not cache_root.is_dir():
        usage_error(f"Eris cache root is not a directory: {cache_root}")

    try:
        removed = [
            label
            for label, namespace in _target_namespaces(cache_root, target)
            if _remove_cache_namespace(cache_root, namespace, label)
        ]
        if target is CacheClearTarget.ALL:
            root_removed = _remove_empty_cache_root(cache_root)
        else:
            root_removed = False
    except CacheError as error:
        usage_error(str(error))

    if removed:
        typer.echo(f"Cleared Eris {target.value} cache namespaces: {', '.join(removed)}.")
    else:
        typer.echo(f"No Eris {target.value} cache namespaces were present.")
    if root_removed:
        typer.echo("Removed empty Eris cache root.")


@app.command("info")
def cache_info() -> None:
    """
    Inspect Eris cache state without creating, deleting, or repairing entries.
    """

    cache_root = default_cache_root()
    artifact_root = cache_root / "artifacts"
    checkpoint_root = cache_root / "checkpoints"

    table = Table(title="Eris Cache", show_header=False)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value")
    table.add_row("artifact cache", "present" if artifact_root.is_dir() else "absent")
    table.add_row("artifact entries", str(_count_artifact_entries(cache_root)))
    table.add_row("checkpoint cache", "present" if checkpoint_root.is_dir() else "absent")
    table.add_row(
        "gearnet_edge checkpoint cached",
        str(_official_checkpoint_cached(cache_root)).lower(),
    )
    console.print(table)
