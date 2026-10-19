"""Shared CLI pieces: ``--config`` handling, config validation and the provenance sidecar."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

import typer

import eris
from eris._provenance import write_provenance
from eris.cli._config_file import ConfigIssue, check_config, read_config
from eris.core.registry import get_encoder_info

# Options that stay on the command line and are not provenance parameters.
CONFIG_EXCLUDE = frozenset({"out", "results", "checkpoint", "config"})


def check_params(representation: str, params: Any) -> list[ConfigIssue]:
    """Check representation parameters against the representation's own config dataclass."""
    try:
        config_type = get_encoder_info(representation).config_type
    except KeyError:
        return [ConfigIssue("representation", f"unknown representation {representation!r}")]
    if not isinstance(params, Mapping):
        return [ConfigIssue("params", f"expected a mapping, got {type(params).__name__}")]
    known = {f.name for f in fields(config_type)}
    issues = [ConfigIssue(f"params.{key}", "unknown parameter") for key in params if key not in known]
    for key in params.keys() & known:
        try:
            config_type(**{key: params[key]})
        except Exception as exc:  # noqa: BLE001 - any constructor failure is a config problem
            issues.append(ConfigIssue(f"params.{key}", str(exc)))
    if not issues:
        try:
            config_type(**params)
        except Exception as exc:  # noqa: BLE001 - any constructor failure is a config problem
            issues.append(ConfigIssue("params", str(exc)))
    return issues


def config_issues(command: Any, path: Path) -> tuple[dict[str, Any], list[ConfigIssue]]:
    """Read ``path`` and collect every problem with its options and nested ``params``."""
    raw, issues = read_config(path)
    if issues:
        return raw, issues
    rest = {key: value for key, value in raw.items() if key != "params"}
    issues = check_config(command, rest, exclude=CONFIG_EXCLUDE)
    if "params" in raw:
        if any(getattr(p, "name", None) == "params" for p in command.params):
            if isinstance(raw.get("representation"), str):
                issues += check_params(raw["representation"], raw["params"])
            elif not isinstance(raw["params"], Mapping):
                issues.append(
                    ConfigIssue("params", f"expected a mapping, got {type(raw['params']).__name__}")
                )
        else:
            issues.append(ConfigIssue("params", "unknown key"))
    return raw, issues


def config_callback(ctx: typer.Context, value: Path | None) -> Path | None:
    """Eager ``--config`` callback: install the file as defaults, so flags still win."""
    if value is None:
        return None
    raw, issues = config_issues(ctx.command, value)
    if issues:
        lines = "\n".join(f"  {issue.path or '<root>'}: {issue.message}" for issue in issues)
        raise typer.BadParameter(f"{value}:\n{lines}", ctx=ctx, param_hint="'--config'")
    defaults = dict(raw)
    if "params" in defaults:  # the option takes a JSON string; the file holds a mapping
        defaults["params"] = json.dumps(defaults["params"])
    ctx.default_map = {**(ctx.default_map or {}), **defaults}
    return value


CONFIG_OPTION = typer.Option(
    "--config",
    callback=config_callback,
    is_eager=True,
    dir_okay=False,
    help="YAML/JSON file with option values; representation parameters go under 'params'.",
)


def write_run_provenance(
    ctx: typer.Context,
    output: Path,
    *,
    command: str,
    created_at: str,
    inputs: list[Path],
    extensions: dict[str, Any],
) -> None:
    """Write the provenance sidecar for a finished run."""
    parameters = {k: v for k, v in ctx.params.items() if k not in CONFIG_EXCLUDE}
    for path_option in ("input", "manifest", "folder"):
        parameters.pop(path_option, None)
    if parameters.get("params") is not None:  # record the parsed mapping, not the JSON string
        parameters["params"] = json.loads(parameters["params"])
    write_provenance(
        output,
        tool="eris",
        version=eris.__version__,
        command=command,
        created_at=created_at,
        parameters=parameters,
        inputs=inputs,
        packages=("erislib", "numpy", "polars"),
        extensions=extensions,
    )
