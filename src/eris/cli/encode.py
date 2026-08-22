from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Any, List

import typer

from eris.core.registry import get_encoder
from eris.core.types import StructureInput
from eris.cli._utils import apply_params

from eris.pipelines import batch_encode

app = typer.Typer(add_completion=False, help="Encode structures into Eris artifacts.")


@app.command("single")
def run(
    input_path: Path = typer.Option(..., "--input", exists=True, readable=True),
    encoder: str = typer.Option("residue_distogram", "--encoder"),
    chain: Optional[str] = typer.Option(None, "--chain"),
    model_index: int = typer.Option(0, "--model"),
    out: Path = typer.Option(Path("out"), "--out"),
    sample_id: Optional[str] = typer.Option(None, "--id"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache for this run"),
    params: str = typer.Option("{}", "--params", help="JSON dict with encoder parameters"),
    allow_unknown_params: bool = typer.Option(False, "--allow-unknown-params", help="Ignore unknown keys in --params"),
) -> None:
    """
    Encode a single structure file into an artifact folder.
    """
    enc: Any = get_encoder(encoder)

    apply_params(enc, params, allow_unknown=allow_unknown_params)

    if no_cache and hasattr(enc, "use_cache"):
        enc.use_cache = False

    inp = StructureInput(source=str(input_path), chain_id=chain, model_index=model_index)
    art = enc.encode(inp)

    sid = sample_id or input_path.stem
    outdir = out / sid / encoder
    art.save(outdir)
    typer.echo(str(outdir))


@app.command("batch")
def run(
    manifest: Path = typer.Option(..., "--manifest", exists=True, readable=True),
    encoder: List[str] = typer.Option(..., "--encoder", help="Repeatable. Example: --encoder residue_knn_graph"),
    out: Path = typer.Option(Path("out"), "--out"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    no_cache: bool = typer.Option(False, "--no-cache"),
    params_json: Optional[str] = typer.Option(None, "--params", help="JSON dict: {encoder_name: {param: value}}"),
    fail_fast: bool = typer.Option(False, "--fail-fast"),
) -> None:
    """
    Encode multiple structure files into multiple artifacts.
    """
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
