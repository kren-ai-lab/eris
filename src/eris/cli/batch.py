from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

import eris
from eris.exceptions import ConfigError, ExecutionError, RepresentationNotFoundError
from eris.pipelines.manifest import ManifestValidationError
from eris.cli._utils import parse_json_object, usage_error

app = typer.Typer(add_completion=False, help="Run canonical batch encode/embed workflows.")


def _result_path(artifact_root: Path, results: Path | None) -> Path:
    return results if results is not None else artifact_root / "batch_results.csv"


def _summary(rows: list[object], verb: str) -> None:
    successes = sum(getattr(row, "status", None) == "success" for row in rows)
    failures = len(rows) - successes
    typer.echo(f"Batch {verb}: {successes} succeeded, {failures} failed")


@app.command("encode")
def encode(
    manifest: Annotated[Path, typer.Argument(metavar="MANIFEST", exists=True, readable=True)],
    representation: Annotated[
        str,
        typer.Option("--representation", "-r", help="Canonical explicit representation name."),
    ],
    out: Annotated[Path, typer.Option("--out", help="Artifact root directory.")],
    results: Annotated[
        Path | None,
        typer.Option("--results", help="Batch-result manifest path."),
    ] = None,
    resume: Annotated[
        bool, typer.Option("--resume", help="Reuse equivalent completed artifacts.")
    ] = False,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Replace existing artifacts.")
    ] = False,
    fail_fast: Annotated[
        bool, typer.Option("--fail-fast", help="Stop on the first failed row.")
    ] = False,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Disable representation cache.")
    ] = False,
    params: Annotated[
        str | None, typer.Option("--params", help="JSON representation parameters.")
    ] = None,
) -> None:
    """Encode one canonical explicit representation for every manifest row."""
    try:
        rows = eris.batch_encode(
            manifest_path=manifest,
            representation=representation,
            artifact_root=out,
            results_path=_result_path(out, results),
            resume=resume,
            overwrite=overwrite,
            fail_fast=fail_fast,
            no_cache=no_cache,
            params=parse_json_object(params),
        )
    except (
        ConfigError,
        ManifestValidationError,
        RepresentationNotFoundError,
        ExecutionError,
        OSError,
    ) as error:
        usage_error(str(error))
    _summary(rows, "encode")


@app.command("embed")
def embed(
    manifest: Annotated[Path, typer.Argument(metavar="MANIFEST", exists=True, readable=True)],
    representation: Annotated[
        str,
        typer.Option("--representation", "-r", help="Canonical learned representation name."),
    ],
    out: Annotated[Path, typer.Option("--out", help="Artifact root directory.")],
    results: Annotated[
        Path | None, typer.Option("--results", help="Batch-result manifest path.")
    ] = None,
    output_mode: Annotated[str, typer.Option("--output-mode")] = "both",
    checkpoint: Annotated[Path | None, typer.Option("--checkpoint")] = None,
    device: Annotated[str, typer.Option("--device")] = "auto",
    precision: Annotated[str, typer.Option("--precision")] = "fp32",
    resume: Annotated[bool, typer.Option("--resume")] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    fail_fast: Annotated[bool, typer.Option("--fail-fast")] = False,
    no_cache: Annotated[bool, typer.Option("--no-cache")] = False,
) -> None:
    """Embed one canonical learned representation for every manifest row."""
    try:
        rows = eris.batch_embed(
            manifest_path=manifest,
            representation=representation,
            artifact_root=out,
            output_mode=output_mode,
            checkpoint_path=checkpoint,
            device=device,
            precision=precision,
            results_path=_result_path(out, results),
            resume=resume,
            overwrite=overwrite,
            fail_fast=fail_fast,
            use_cache=not no_cache,
        )
    except (
        ConfigError,
        ManifestValidationError,
        RepresentationNotFoundError,
        ExecutionError,
        OSError,
    ) as error:
        usage_error(str(error))
    _summary(rows, "embed")


__all__ = ["app"]
