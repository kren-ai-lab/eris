from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from morpheo.core.registry import get_encoder, list_encoders
from morpheo.core.types import StructureInput

app = typer.Typer(add_completion=False)


@app.command()
def encoders() -> None:
    for name in list_encoders():
        typer.echo(name)


@app.command()
def encode(
    input_path: Path = typer.Option(..., "--input", exists=True, readable=True),
    encoder: str = typer.Option("distogram_ca", "--encoder"),
    chain: Optional[str] = typer.Option(None, "--chain"),
    model_index: int = typer.Option(0, "--model"),
    out: Path = typer.Option(Path("out"), "--out"),
    sample_id: Optional[str] = typer.Option(None, "--id"),
) -> None:
    enc = get_encoder(encoder)
    inp = StructureInput(source=str(input_path), chain_id=chain, model_index=model_index)
    art = enc.encode(inp)

    sid = sample_id or input_path.stem
    outdir = out / sid / encoder
    art.save(outdir)
    typer.echo(str(outdir))
