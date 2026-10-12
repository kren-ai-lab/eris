"""Command that encodes one structure."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

import eris
from eris.cli._utils import parse_json_object, usage_error
from eris.core.artifacts import ArtifactValidationError
from eris.core.io import StructureLoadError
from eris.exceptions import CacheError, ConfigError, RepresentationNotFoundError


def encode(
    input: Annotated[  # noqa: A002
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
    """Encode one structure file into an Eris artifact."""
    encode_kwargs: dict[str, Any] = {
        "input": input,
        "representation": representation,
        "output": out,
        "use_cache": not no_cache,
        **parse_json_object(params),
    }
    if chain is not None:
        encode_kwargs["chain_id"] = chain
    if model_index is not None:
        encode_kwargs["model_index"] = model_index

    try:
        artifact = eris.encode(**encode_kwargs)
    except FileNotFoundError as error:
        missing = error.filename if error.filename is not None else input
        usage_error(f"File not found: {missing}")
    except (
        RepresentationNotFoundError,
        ConfigError,
        StructureLoadError,
        ArtifactValidationError,
        CacheError,
    ) as error:
        usage_error(str(error))

    typer.echo(f"Encoded {artifact.meta['representation_name']} -> {out}")


__all__ = ["encode"]
