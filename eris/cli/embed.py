"""Command that embeds one structure."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

import eris
from eris._provenance import utc_now
from eris.cli._shared import CONFIG_OPTION, write_run_provenance


def embed(
    ctx: typer.Context,
    input: Annotated[  # noqa: A002
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
        typer.Option(
            "--checkpoint",
            help=(
                "Optional local official checkpoint path. If omitted, Eris reuses the verified "
                "checkpoint cache or downloads the registered official checkpoint."
            ),
        ),
    ] = None,
    device: Annotated[str, typer.Option("--device", help="Requested execution device.")] = "auto",
    precision: Annotated[str, typer.Option("--precision", help="Requested precision.")] = "fp32",
    no_cache: Annotated[
        bool,
        typer.Option("--no-cache", help="Disable artifact cache for this run."),
    ] = False,
    config: Annotated[Path | None, CONFIG_OPTION] = None,
) -> None:
    """Embed one structure file into a learned Eris artifact."""
    del config
    created_at = utc_now()
    embed_kwargs: dict[str, Any] = {
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

    artifact = eris.embed(**embed_kwargs)
    write_run_provenance(
        ctx,
        out,
        command="embed",
        created_at=created_at,
        inputs=[Path(input)],
        extensions={"representation": representation},
    )

    mode = artifact.meta.get("parameters", {}).get("output_mode", output_mode)
    typer.echo(f"Embedded {artifact.meta['representation_name']} ({mode}) -> {out}")


__all__ = ["embed"]
