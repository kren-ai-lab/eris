from __future__ import annotations

import shutil
import typer

from morpheo.core.cache import default_cache_dir

app = typer.Typer(add_completion=False, help="Cache utilities.")


@app.command("clear")
def clear_cache() -> None:
    """
    Remove Morpheo disk cache directory.
    """
    cache_dir = default_cache_dir()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        typer.echo(f"Cache cleared: {cache_dir}")
    else:
        typer.echo(f"Cache directory does not exist: {cache_dir}")
