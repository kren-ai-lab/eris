from __future__ import annotations

from pathlib import Path
from typing import Optional, Any

import typer

from morpheo.core.embeddings import load_node_embeddings
from morpheo.core.registry import get_encoder
from morpheo.core.types import StructureInput
from morpheo.cli._utils import apply_params

app = typer.Typer(add_completion=False, help="Encode single structures into Morpheo artifacts.")


@app.command()
def run(
    input_path: Path = typer.Option(..., "--input", exists=True, readable=True),
    encoder: str = typer.Option("distogram_ca", "--encoder"),
    chain: Optional[str] = typer.Option(None, "--chain"),
    model_index: int = typer.Option(0, "--model"),
    out: Path = typer.Option(Path("out"), "--out"),
    sample_id: Optional[str] = typer.Option(None, "--id"),
    node_emb: Optional[Path] = typer.Option(None, "--node-emb"),
    node_emb_key: str = typer.Option("node_emb", "--node-emb-key"),
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

    # Optional: attach external node embeddings (residue-level)
    if node_emb is not None:
        emb = load_node_embeddings(node_emb, key=node_emb_key)
        n = art.data["pos"].shape[0]
        if emb.ndim != 2 or emb.shape[0] != n:
            raise ValueError(f"node_emb shape mismatch. Expected (N,D) with N={n}, got {emb.shape}")
        art.data[node_emb_key] = emb

    sid = sample_id or input_path.stem
    outdir = out / sid / encoder
    art.save(outdir)
    typer.echo(str(outdir))
