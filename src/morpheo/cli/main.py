from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from morpheo.core.registry import get_encoder, list_encoders
from morpheo.core.types import StructureInput
from morpheo.core.embeddings import load_node_embeddings

import shutil
from morpheo.core.cache import default_cache_dir

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
    node_emb: Optional[Path] = typer.Option(None, "--node-emb"),
    node_emb_key: str = typer.Option("node_emb", "--node-emb-key"),
    no_cache: bool = typer.Option(False, "--no-cache"),
) -> None:
    enc = get_encoder(encoder)

    if no_cache and hasattr(enc, "use_cache"):
        enc.use_cache = False

    inp = StructureInput(source=str(input_path), chain_id=chain, model_index=model_index)
    art = enc.encode(inp)

    if node_emb is not None:
        emb = load_node_embeddings(node_emb, key=node_emb_key)
        n = art.data["pos"].shape[0]
        if emb.ndim != 2 or emb.shape[0] != n:
            raise ValueError(
                f"node_emb shape mismatch. Expected (N,D) with N={n}, got {emb.shape}"
            )
        art.data["node_emb"] = emb

    sid = sample_id or input_path.stem
    outdir = out / sid / encoder
    art.save(outdir)
    typer.echo(str(outdir))

@app.command("clear-cache")
def clear_cache() -> None:
    """
    Remove Morpheo disk cache directory.
    """
    cache_dir = default_cache_dir()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        typer.echo(f"Cache cleared: {cache_dir}")
    else:
        typer.echo(f"Cache directory does not exist: {cache_dir}")
