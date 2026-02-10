from __future__ import annotations

import csv
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

import typer

from morpheo.core.cache import default_cache_dir
from morpheo.core.embeddings import load_node_embeddings
from morpheo.core.registry import get_encoder, list_encoders
from morpheo.core.types import StructureInput

app = typer.Typer(add_completion=False)


def _apply_params(enc: Any, params_json: str, allow_unknown: bool = False) -> None:
    """
    Apply JSON parameters to an encoder instance via setattr.

    Parameters
    ----------
    enc : encoder instance
    params_json : str
        JSON object. Example: '{"cutoff": 8.0, "include_self": false}'
    allow_unknown : bool
        If True, silently ignore unknown keys.
    """
    if not params_json or params_json.strip() in ("{}", ""):
        return

    try:
        params = json.loads(params_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for --params: {e}") from e

    if not isinstance(params, dict):
        raise ValueError("--params must be a JSON object/dict, e.g. '{\"cutoff\": 8.0}'")

    for k, v in params.items():
        if hasattr(enc, k):
            setattr(enc, k, v)
        else:
            if allow_unknown:
                continue
            raise ValueError(f"Encoder '{getattr(enc, 'name', enc.__class__.__name__)}' has no parameter '{k}'")


@app.command()
def encoders() -> None:
    """
    List available encoders.
    """
    for name in list_encoders():
        typer.echo(name)


@app.command()
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
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache for this run"),
    params: str = typer.Option("{}", "--params", help="JSON dict with encoder parameters"),
    allow_unknown_params: bool = typer.Option(
        False, "--allow-unknown-params", help="Ignore unknown keys in --params"
    ),
) -> None:
    """
    Encode a single structure file into an artifact folder.

    Examples
    --------
    morpheo encode --input 1AQ2.pdb --encoder residue_contact_graph --params '{"cutoff": 10.0}'
    morpheo encode --input 1AQ2.pdb --encoder distogram_bins_ca --params '{"dmin":2,"dmax":20,"num_bins":36}'
    """
    enc = get_encoder(encoder)

    # Apply encoder parameters
    _apply_params(enc, params, allow_unknown=allow_unknown_params)

    # Toggle cache
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
        art.data["node_emb"] = emb

    sid = sample_id or input_path.stem
    outdir = out / sid / encoder
    art.save(outdir)
    typer.echo(str(outdir))


@app.command()
def batch(
    manifest: Path = typer.Option(..., "--manifest", exists=True, readable=True),
    encoder: str = typer.Option(..., "--encoder"),
    out: Path = typer.Option(Path("out"), "--out"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable cache for this run"),
    n_jobs: int = typer.Option(1, "--n-jobs", min=1, help="Parallel workers (threads)"),
    failed: Path = typer.Option(Path("failed.csv"), "--failed", help="Write failures to CSV"),
    params: str = typer.Option("{}", "--params", help="JSON dict with encoder parameters"),
    allow_unknown_params: bool = typer.Option(
        False, "--allow-unknown-params", help="Ignore unknown keys in --params"
    ),
) -> None:
    """
    Batch-encode structures from a CSV manifest.

    Manifest columns:
      - id (required)
      - path (required)
      - chain (optional)
      - model (optional, default 0)
    """
    out.mkdir(parents=True, exist_ok=True)

    # Read manifest rows
    rows: list[dict] = []
    with manifest.open("r", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in ("id", "path") if (reader.fieldnames is None or c not in reader.fieldnames)]
        if missing:
            raise ValueError(f"Manifest missing columns: {missing}. Found: {reader.fieldnames}")

        for r in reader:
            sid = (r.get("id") or "").strip()
            spath = (r.get("path") or "").strip()
            chain = (r.get("chain") or "").strip() or None
            model_raw = (r.get("model") or "").strip()
            model = int(model_raw) if model_raw else 0

            if not sid or not spath:
                continue

            rows.append({"id": sid, "path": spath, "chain": chain, "model": model})

    if not rows:
        typer.echo("No valid rows found in manifest.")
        raise typer.Exit(code=1)

    # Worker
    def _run_one(item: dict) -> tuple[str, str, Optional[str]]:
        """
        Returns (id, status, error_message)
        """
        sid = item["id"]
        path = Path(item["path"])
        chain = item["chain"]
        model = item["model"]

        enc = get_encoder(encoder)

        # Apply encoder parameters
        _apply_params(enc, params, allow_unknown=allow_unknown_params)

        # Toggle cache
        if no_cache and hasattr(enc, "use_cache"):
            enc.use_cache = False

        outdir = out / sid / encoder
        data_npz = outdir / "data.npz"
        meta_json = outdir / "meta.json"

        if resume and data_npz.exists() and meta_json.exists():
            return sid, "skipped", None

        try:
            inp = StructureInput(source=str(path), chain_id=chain, model_index=model)
            art = enc.encode(inp)
            art.save(outdir)
            return sid, "ok", None
        except Exception as e:
            return sid, "failed", str(e)

    failures: list[dict] = []
    ok = 0
    skipped = 0

    if n_jobs == 1:
        for item in rows:
            sid, status, err = _run_one(item)
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                failures.append({"id": sid, "path": item["path"], "error": err or ""})
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            futs = {ex.submit(_run_one, item): item for item in rows}
            for fut in as_completed(futs):
                item = futs[fut]
                sid, status, err = fut.result()
                if status == "ok":
                    ok += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    failures.append({"id": sid, "path": item["path"], "error": err or ""})

    if failures:
        failed.parent.mkdir(parents=True, exist_ok=True)
        with failed.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "path", "error"])
            w.writeheader()
            w.writerows(failures)
        typer.echo(f"Done: ok={ok} skipped={skipped} failed={len(failures)} (see {failed})")
    else:
        typer.echo(f"Done: ok={ok} skipped={skipped} failed=0")
