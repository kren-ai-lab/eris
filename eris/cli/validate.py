"""Command that validates a ``--config`` file without reading structures, the network or a GPU."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer

from eris.cli._config_file import report_issues
from eris.cli._shared import config_issues


class Target(StrEnum):
    """Command whose config shape is validated."""

    encode_cmd = "encode"
    embed_cmd = "embed"
    batch_encode = "batch-encode"
    batch_embed = "batch-embed"
    batch_from_folder = "batch-from-folder"


def _command(target: Target) -> Any:
    from eris.cli.batch import app as batch_app  # noqa: PLC0415
    from eris.cli.embed import embed  # noqa: PLC0415
    from eris.cli.encode import encode  # noqa: PLC0415

    if target.value.startswith("batch-"):
        return typer.main.get_command(batch_app).commands[target.value.removeprefix("batch-")]  # type: ignore[attr-defined]
    single = typer.Typer()
    single.command()(embed if target is Target.embed_cmd else encode)
    return typer.main.get_command(single)


def validate(
    target: Annotated[Target, typer.Argument(help="Command the config is for.")],
    config: Annotated[Path, typer.Argument(help="YAML/JSON config file.")],
    as_json: Annotated[
        bool, typer.Option("--json", help="Print problems as a JSON array on stdout.")
    ] = False,
) -> None:
    """Validate CONFIG as accepted by `eris TARGET --config CONFIG`."""
    _, issues = config_issues(_command(target), config)
    raise typer.Exit(report_issues(issues, as_json=as_json))


__all__ = ["validate"]
