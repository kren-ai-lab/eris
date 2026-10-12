"""Command that encodes one structure."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

import eris
from eris.cli._utils import parse_json_object


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

    artifact = eris.encode(**encode_kwargs)

    typer.echo(f"Encoded {artifact.meta['representation_name']} -> {out}")


__all__ = ["encode"]
