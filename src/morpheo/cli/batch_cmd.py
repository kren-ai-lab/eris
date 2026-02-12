from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List

import typer

from morpheo.cli.batch import batch_encode

app = typer.Typer(add_completion=False, help="Batch encoding utilities.")


@app.command()
def run(
    manifest: Path = typer.Option(..., "--manifest", exists=True, readable=True),
    encoder: List[str] = typer.Option(..., "--encoder", help="Repeatable. Example: --encoder residue_contact_graph_ns"),
    out: Path = typer.Option(Path("out"), "--out"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    no_cache: bool = typer.Option(False, "--no-cache"),
    params_json: Optional[str] = typer.Option(None, "--params", help="JSON dict: {encoder_name: {param: value}}"),
    fail_fast: bool = typer.Option(False, "--fail-fast"),
) -> None:
    params = json.loads(params_json) if params_json else None
    stats = batch_encode(
        manifest_path=manifest,
        encoders=encoder,
        out=out,
        overwrite=overwrite,
        no_cache=no_cache,
        params=params,
        fail_fast=fail_fast,
    )
    typer.echo(f"Batch done | {stats}")
