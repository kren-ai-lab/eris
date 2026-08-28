from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, NoReturn

import typer

import eris
from eris.core.artifacts import ArtifactValidationError
from eris.core.io import StructureLoadError
from eris.exceptions import CacheError, ConfigError, RepresentationNotFoundError


def _usage_error(message: str) -> NoReturn:
    typer.echo(message, err=True)
    raise typer.Exit(code=2)


def _parse_params(params: str | None) -> dict[str, object]:
    if params is None:
        return {}
    try:
        parsed = json.loads(params)
    except json.JSONDecodeError as error:
        _usage_error(f"Invalid JSON for --params: {error.msg}")
    if not isinstance(parsed, dict):
        _usage_error("--params must be a JSON object")
    return parsed


def encode(
    input: Annotated[
        str,
        typer.Argument(..., metavar="INPUT", help="Structure file to encode."),
    ],
    representation: Annotated[
        str,
        typer.Option("--representation", "-r", help="Canonical explicit representation name."),
    ],
    out: Annotated[Path, typer.Option("--out", help="Artifact output directory.")],
    chain: Annotated[str | None, typer.Option("--chain", help="Chain identifier.")] = None,
    model_index: Annotated[int | None, typer.Option("--model", help="Model index.")] = None,
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Disable cache for this run."),
    ] = False,
    params: Annotated[
        str | None,
        typer.Option("--params", help="JSON object with representation parameters."),
    ] = None,
) -> None:
    """Encode one structure file into a Schema v2 artifact."""

    encode_kwargs = {
        "input": input,
        "representation": representation,
        "output": out,
        "use_cache": not no_cache,
        **_parse_params(params),
    }
    if chain is not None:
        encode_kwargs["chain_id"] = chain
    if model_index is not None:
        encode_kwargs["model_index"] = model_index

    try:
        artifact = eris.encode(**encode_kwargs)
    except FileNotFoundError as error:
        missing = error.filename if error.filename is not None else input
        _usage_error(f"File not found: {missing}")
    except (
        RepresentationNotFoundError,
        ConfigError,
        StructureLoadError,
        ArtifactValidationError,
        CacheError,
    ) as error:
        _usage_error(str(error))

    typer.echo(f"Encoded {artifact.meta['representation_name']} -> {out}")


__all__ = ["encode"]
