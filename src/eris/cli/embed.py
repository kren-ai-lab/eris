from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

import eris
from eris.core.artifacts import ArtifactValidationError
from eris.core.io import StructureLoadError
from eris.exceptions import (
    CacheError,
    ConfigError,
    ExecutionError,
    RepresentationNotFoundError,
)
from eris.cli._utils import usage_error


def embed(
    input: Annotated[
        str,
        typer.Argument(..., metavar="INPUT", help="Structure file to embed."),
    ],
    representation: Annotated[
        str,
        typer.Option("--representation", "-r", help="Canonical learned representation name."),
    ],
    out: Annotated[Path, typer.Option("--out", help="Artifact output directory.")],
    chain: Annotated[str | None, typer.Option("--chain", help="Chain identifier.")] = None,
    model_index: Annotated[int | None, typer.Option("--model", help="Model index.")] = None,
    output_mode: Annotated[
        str,
        typer.Option("--output-mode", help="Learned output mode: node, global, or both."),
    ] = "both",
    checkpoint: Annotated[
        Path | None,
        typer.Option("--checkpoint", help="Local official checkpoint path."),
    ] = None,
    device: Annotated[str, typer.Option("--device", help="Requested execution device.")] = "auto",
    precision: Annotated[str, typer.Option("--precision", help="Requested precision.")] = "fp32",
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Disable artifact cache for this run."),
    ] = False,
) -> None:
    """Embed one structure file into a learned Eris artifact."""

    embed_kwargs = {
        "input": input,
        "representation": representation,
        "output_mode": output_mode,
        "device": device,
        "precision": precision,
        "output": out,
        "use_cache": not no_cache,
    }
    if chain is not None:
        embed_kwargs["chain_id"] = chain
    if model_index is not None:
        embed_kwargs["model_index"] = model_index
    if checkpoint is not None:
        embed_kwargs["checkpoint_path"] = checkpoint

    try:
        artifact = eris.embed(**embed_kwargs)
    except FileNotFoundError as error:
        missing = error.filename if error.filename is not None else input
        usage_error(f"File not found: {missing}")
    except (
        RepresentationNotFoundError,
        ConfigError,
        StructureLoadError,
        ArtifactValidationError,
        CacheError,
        ExecutionError,
    ) as error:
        usage_error(str(error))

    mode = artifact.meta.get("parameters", {}).get("output_mode", output_mode)
    typer.echo(f"Embedded {artifact.meta['representation_name']} ({mode}) -> {out}")


__all__ = ["embed"]
