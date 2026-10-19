"""Command that validates a ``--config`` file without reading structures, the network or a GPU."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from os import PathLike
from pathlib import Path
from typing import Annotated, Any

import typer

from eris.cli._config_file import ConfigIssue, read_config, report_issues
from eris.cli._shared import mapping_issues


class Target(StrEnum):
    """Command whose config shape is validated."""

    encode_cmd = "encode"
    embed_cmd = "embed"
    batch_encode = "batch-encode"
    batch_embed = "batch-embed"
    batch_from_folder = "batch-from-folder"
    features = "features"
    export_to_torch = "export-to-torch"
    export_to_pyg = "export-to-pyg"


def _command(target: Target) -> Any:
    from eris.cli.batch import app as batch_app  # noqa: PLC0415
    from eris.cli.embed import embed  # noqa: PLC0415
    from eris.cli.encode import encode  # noqa: PLC0415
    from eris.cli.export import app as export_app  # noqa: PLC0415
    from eris.cli.features import features  # noqa: PLC0415

    if target.value.startswith("batch-"):
        return typer.main.get_command(batch_app).commands[target.value.removeprefix("batch-")]  # type: ignore[attr-defined]
    if target.value.startswith("export-"):
        return typer.main.get_command(export_app).commands[target.value.removeprefix("export-")]  # type: ignore[attr-defined]
    single = typer.Typer()
    single.command()({Target.embed_cmd: embed, Target.features: features}.get(target, encode))
    return typer.main.get_command(single)


def validate_config(config: Mapping[str, Any] | str | PathLike[str], target: str) -> list[ConfigIssue]:
    """Check ``config`` (a mapping, or the path of a YAML/JSON file) as accepted by ``eris TARGET --config``.

    ``target`` is ``encode``, ``embed``, ``batch-encode``, ``batch-embed``, ``batch-from-folder``,
    ``features``, ``export-to-torch`` or ``export-to-pyg``.
    Returns the problems found (empty when valid); an unknown target or a missing or unparsable
    file is one issue at ``""``.
    """
    try:
        command = _command(Target(target))
    except ValueError:
        return [ConfigIssue("", f"unknown target {target!r}; expected one of: {', '.join(Target)}")]
    if isinstance(config, Mapping):
        return mapping_issues(command, dict(config))
    raw, issues = read_config(Path(config))
    return issues or mapping_issues(command, raw)


def validate(
    target: Annotated[Target, typer.Argument(help="Command the config is for.")],
    config: Annotated[Path, typer.Argument(help="YAML/JSON config file.")],
    as_json: Annotated[
        bool, typer.Option("--json", help="Print problems as a JSON array on stdout.")
    ] = False,
) -> None:
    """Validate CONFIG as accepted by `eris TARGET --config CONFIG`."""
    issues = validate_config(config, target)
    raise typer.Exit(report_issues(issues, as_json=as_json))


__all__ = ["validate", "validate_config"]
