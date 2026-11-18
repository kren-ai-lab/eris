"""Batch command-line entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import polars as pl
import typer

import eris
from eris._provenance import utc_now
from eris.cli._shared import CONFIG_OPTION, write_run_provenance
from eris.cli._utils import parse_json_object
from eris.pipelines.manifest import ManifestValidationError

app = typer.Typer(add_completion=False, help="Run canonical batch encode/embed workflows.")

# Same value as ``eris.cli.main.EXIT_WORKFLOW`` (not imported: main imports this module).
_EXIT_WORKFLOW = 3

ALLOW_FAILURES_OPTION = typer.Option(
    "--allow-failures",
    help="Exit 0 when some rows fail, as long as at least one succeeds (failures stay in the results file).",
)


def _result_path(artifact_root: Path, results: Path | None) -> Path:
    return results if results is not None else artifact_root / "batch_results.csv"


def _summary(
    results: pl.DataFrame,
    verb: str,
    ctx: typer.Context,
    *,
    command: str,
    created_at: str,
    out: Path,
    source: Path,
    representation: str,
    results_path: Path,
    allow_failures: bool,
) -> None:
    successes = results.filter(pl.col("status") == "success").height
    failures = results.height - successes
    typer.echo(f"Batch {verb}: {successes} succeeded, {failures} failed")
    write_run_provenance(
        ctx,
        out,
        command=command,
        created_at=created_at,
        inputs=[source],
        extensions={"representation": representation, "records_ok": successes, "records_failed": failures},
    )
    if failures and (not allow_failures or successes == 0):
        hint = "" if allow_failures else " (pass --allow-failures to accept partial results)"
        typer.echo(f"{failures} of {results.height} rows failed; see {results_path}{hint}", err=True)
        raise typer.Exit(code=_EXIT_WORKFLOW)


def _folder_manifest(folder: Path, pattern: str, recursive: bool) -> pl.DataFrame:
    paths = sorted(
        (folder.rglob(pattern) if recursive else folder.glob(pattern)),
        key=lambda path: path.as_posix(),
    )
    files = [path for path in paths if path.is_file()]
    if not files:
        raise ManifestValidationError(f"No files matching '{pattern}' found in {folder}")

    # Duplicate stems are rejected by manifest validation.
    return pl.DataFrame({"id": [path.stem for path in files], "path": [str(path) for path in files]})


@app.command("encode")
def encode(
    ctx: typer.Context,
    manifest: Annotated[Path, typer.Argument(metavar="MANIFEST")],
    representation: Annotated[
        str,
        typer.Option("--representation", "-r", help="Canonical explicit representation name."),
    ],
    out: Annotated[Path, typer.Option("--out", help="Artifact root directory.")],
    results: Annotated[
        Path | None,
        typer.Option("--results", help="Batch-result manifest path."),
    ] = None,
    resume: Annotated[bool, typer.Option("--resume", help="Reuse equivalent completed artifacts.")] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Replace existing artifacts.")] = False,
    fail_fast: Annotated[bool, typer.Option("--fail-fast", help="Stop on the first failed row.")] = False,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Disable representation cache.")] = False,
    params: Annotated[str | None, typer.Option("--params", help="JSON representation parameters.")] = None,
    allow_failures: Annotated[bool, ALLOW_FAILURES_OPTION] = False,
    config: Annotated[Path | None, CONFIG_OPTION] = None,
) -> None:
    """Encode one canonical explicit representation for every manifest row."""
    del config
    created_at = utc_now()
    results_path = _result_path(out, results)
    outcome = eris.batch_encode(
        manifest,
        representation=representation,
        artifact_root=out,
        results_path=results_path,
        resume=resume,
        overwrite=overwrite,
        fail_fast=fail_fast,
        use_cache=not no_cache,
        params=parse_json_object(params),
    )
    _summary(
        outcome,
        "encode",
        ctx,
        command="batch encode",
        created_at=created_at,
        out=out,
        source=manifest,
        representation=representation,
        results_path=results_path,
        allow_failures=allow_failures,
    )


@app.command("embed")
def embed(
    ctx: typer.Context,
    manifest: Annotated[Path, typer.Argument(metavar="MANIFEST")],
    representation: Annotated[
        str,
        typer.Option("--representation", "-r", help="Canonical learned representation name."),
    ],
    out: Annotated[Path, typer.Option("--out", help="Artifact root directory.")],
    results: Annotated[Path | None, typer.Option("--results", help="Batch-result manifest path.")] = None,
    output_mode: Annotated[str, typer.Option("--output-mode")] = "both",
    checkpoint: Annotated[Path | None, typer.Option("--checkpoint")] = None,
    device: Annotated[str, typer.Option("--device")] = "auto",
    precision: Annotated[str, typer.Option("--precision")] = "fp32",
    resume: Annotated[bool, typer.Option("--resume")] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    fail_fast: Annotated[bool, typer.Option("--fail-fast")] = False,
    no_cache: Annotated[bool, typer.Option("--no-cache")] = False,
    allow_failures: Annotated[bool, ALLOW_FAILURES_OPTION] = False,
    config: Annotated[Path | None, CONFIG_OPTION] = None,
) -> None:
    """Embed one canonical learned representation for every manifest row."""
    del config
    created_at = utc_now()
    results_path = _result_path(out, results)
    outcome = eris.batch_embed(
        manifest,
        representation=representation,
        artifact_root=out,
        output_mode=output_mode,
        checkpoint_path=checkpoint,
        device=device,
        precision=precision,
        results_path=results_path,
        resume=resume,
        overwrite=overwrite,
        fail_fast=fail_fast,
        use_cache=not no_cache,
    )
    _summary(
        outcome,
        "embed",
        ctx,
        command="batch embed",
        created_at=created_at,
        out=out,
        source=manifest,
        representation=representation,
        results_path=results_path,
        allow_failures=allow_failures,
    )


@app.command("from-folder")
def from_folder(
    ctx: typer.Context,
    folder: Annotated[Path, typer.Argument(metavar="FOLDER")],
    representation: Annotated[
        str,
        typer.Option("--representation", "-r", help="Canonical explicit representation name."),
    ],
    out: Annotated[Path, typer.Option("--out", help="Artifact root directory.")],
    pattern: Annotated[str, typer.Option("--pattern", help="File matching pattern.")] = "*.pdb",
    recursive: Annotated[bool, typer.Option("--recursive", help="Search nested folders.")] = False,
    results: Annotated[
        Path | None,
        typer.Option("--results", help="Batch-result manifest path."),
    ] = None,
    resume: Annotated[bool, typer.Option("--resume", help="Reuse equivalent completed artifacts.")] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Replace existing artifacts.")] = False,
    fail_fast: Annotated[bool, typer.Option("--fail-fast", help="Stop on the first failed row.")] = False,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Disable representation cache.")] = False,
    params: Annotated[str | None, typer.Option("--params", help="JSON representation parameters.")] = None,
    allow_failures: Annotated[bool, ALLOW_FAILURES_OPTION] = False,
    config: Annotated[Path | None, CONFIG_OPTION] = None,
) -> None:
    """Encode one explicit representation for every matching structure file."""
    del config
    created_at = utc_now()
    results_path = _result_path(out, results)
    outcome = eris.batch_encode(
        _folder_manifest(folder, pattern, recursive),
        representation=representation,
        artifact_root=out,
        results_path=results_path,
        resume=resume,
        overwrite=overwrite,
        fail_fast=fail_fast,
        use_cache=not no_cache,
        params=parse_json_object(params),
    )
    _summary(
        outcome,
        "encode",
        ctx,
        command="batch from-folder",
        created_at=created_at,
        out=out,
        source=folder,
        representation=representation,
        results_path=results_path,
        allow_failures=allow_failures,
    )


__all__ = ["app"]
