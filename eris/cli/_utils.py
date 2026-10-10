from __future__ import annotations

import json
from pathlib import Path
from typing import NoReturn

import typer
from platformdirs import user_cache_dir


def usage_error(message: str) -> NoReturn:
    typer.echo(message, err=True)
    raise typer.Exit(code=2)


def parse_json_object(params: str | None) -> dict[str, object]:
    if params is None:
        return {}
    try:
        value = json.loads(params)
    except json.JSONDecodeError as error:
        usage_error(f"Invalid JSON for --params: {error.msg}")
    if not isinstance(value, dict):
        usage_error("--params must be a JSON object")
    return value


def cache_root() -> Path:
    return Path(user_cache_dir("eris"))
