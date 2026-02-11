from __future__ import annotations

import csv
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, List

import typer

from morpheo.core.cache import default_cache_dir
from morpheo.core.embeddings import load_node_embeddings
from morpheo.core.registry import get_encoder, list_encoders
from morpheo.core.types import StructureInput

from morpheo.cli.batch import batch_encode

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
def encoders(
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    from morpheo.core.registry import list_encoders, list_encoder_info

    if not verbose:
        for n in list_encoders():
            typer.echo(n)
        return

    # verbose table
    rows = list_encoder_info()
    header = f"{'name':28s} {'type':7s} {'level':7s} {'family':10s} desc"
    typer.echo(header)
    typer.echo("-" * len(header))
    for r in rows:
        typer.echo(f"{r.name:28s} {r.type:7s} {r.level:7s} {r.family:10s} {r.desc}")

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

@app.command()
def pack(
    root: Path = typer.Option(Path("out"), "--root", exists=True, readable=True),
    out_csv: Path = typer.Option(Path("index.csv"), "--out"),
    encoder: Optional[str] = typer.Option(None, "--encoder", help="Only include encoder(s), comma-separated"),
    prefer: str = typer.Option(
        "any",
        "--prefer",
        help="Filter by representation type: any|graph|graph_ns|grid|cloud|matrix",
    ),
    grid_c: Optional[int] = typer.Option(None, "--grid-c", help="For grid prefer: only include grids with this #channels"),
    grid_g: Optional[int] = typer.Option(None, "--grid-g", help="For grid prefer: only include grids with this grid size G"),
    x_dim: Optional[int] = typer.Option(None, "--x-dim", help="For graph/cloud: only include samples with this x feature dim"),
    include_meta: bool = typer.Option(False, "--include-meta", help="Store meta.json content as a JSON string column"),
) -> None:
    """
    Scan an output folder and build an index.csv for dataset loading.

    Expected structure:
      out/<sample_id>/<encoder>/meta.json
      out/<sample_id>/<encoder>/data.npz
    """
    import csv
    import json

    import numpy as np

    prefer = (prefer or "any").strip().lower()
    allowed = {"any", "graph", "graph_ns", "grid", "cloud", "matrix"}
    if prefer not in allowed:
        raise typer.BadParameter(f"prefer must be one of: {', '.join(sorted(allowed))}")

    enc_allow = None
    if encoder is not None:
        enc_allow = {e.strip() for e in encoder.split(",") if e.strip()}
        if not enc_allow:
            enc_allow = None

    def infer(npz_path: Path) -> dict:
        """
        Infer lightweight schema info from data.npz without loading full arrays.
        """
        info: dict = {
            "has_graph": False,
            "has_grid": False,
            "has_cloud": False,
            "has_matrix": False,
            "x_dim": "",
            "edge_attr_dim": "",
            "graph_attr_dim": "",
            "grid_c": "",
            "grid_g": "",
            "matrix_key": "",
            "matrix_n": "",
        }

        try:
            with np.load(npz_path, allow_pickle=False) as z:
                files = set(z.files)

                info["has_graph"] = "edge_index" in files
                info["has_grid"] = "grid" in files
                info["has_cloud"] = "pos" in files

                # x_dim inference (node_attr > node_emb > aa_index)
                if "node_attr" in files:
                    info["x_dim"] = int(z["node_attr"].shape[1])
                elif "node_emb" in files:
                    info["x_dim"] = int(z["node_emb"].shape[1])
                elif "aa_index" in files:
                    info["x_dim"] = 21  # baseline

                if "edge_attr" in files:
                    info["edge_attr_dim"] = int(z["edge_attr"].shape[1])

                if "graph_attr" in files:
                    info["graph_attr_dim"] = int(z["graph_attr"].reshape(-1).shape[0])

                # grid schema
                if "grid" in files:
                    g = z["grid"]
                    # expects (C,G,G,G)
                    if g.ndim == 4:
                        info["grid_c"] = int(g.shape[0])
                        info["grid_g"] = int(g.shape[-1])

                # matrix schema
                for k in ("dist", "dist_bins", "contact", "contact_bins"):
                    if k in files:
                        info["has_matrix"] = True
                        info["matrix_key"] = k
                        info["matrix_n"] = int(z[k].shape[-1])
                        break

        except Exception:
            pass

        return info

    def prefer_ok(enc: str, meta: dict, inf: dict) -> bool:
        shapes = meta.get("shapes", {})
        if not isinstance(shapes, dict):
            shapes = {}

        has_graph = bool(inf["has_graph"]) or ("edge_index" in shapes)
        has_grid = bool(inf["has_grid"]) or ("grid" in shapes)
        has_cloud = bool(inf["has_cloud"]) or ("pos" in shapes)
        has_matrix = bool(inf["has_matrix"]) or any(k in shapes for k in ("dist", "dist_bins", "contact", "contact_bins"))

        if prefer == "any":
            return True
        if prefer == "graph":
            return has_graph
        if prefer == "graph_ns":
            return has_graph and enc.endswith("_ns")
        if prefer == "grid":
            return has_grid
        if prefer == "cloud":
            return has_cloud and (not has_graph) and (not has_grid) and (not has_matrix)
        if prefer == "matrix":
            return has_matrix
        return True

    rows = []
    for sid_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        sid = sid_dir.name
        for enc_dir in sorted([p for p in sid_dir.iterdir() if p.is_dir()]):
            enc = enc_dir.name
            if enc_allow is not None and enc not in enc_allow:
                continue

            meta_path = enc_dir / "meta.json"
            data_path = enc_dir / "data.npz"
            if not meta_path.exists() or not data_path.exists():
                continue

            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = {}

            inf = infer(data_path)

            if not prefer_ok(enc, meta, inf):
                continue

            # --- extra schema filters (homogenize indices)
            if prefer == "grid":
                if grid_c is not None and inf["grid_c"] != grid_c:
                    continue
                if grid_g is not None and inf["grid_g"] != grid_g:
                    continue

            if x_dim is not None:
                # applies to graph/cloud primarily, but ok generally
                if inf["x_dim"] == "" or int(inf["x_dim"]) != int(x_dim):
                    continue

            row = {
                "id": sid,
                "encoder": enc,
                "artifact_dir": str(enc_dir),
                "data_path": str(data_path),
                "meta_path": str(meta_path),
                "n_residues": meta.get("n_residues", ""),
                "n_edges": meta.get("n_edges", ""),
                # schema columns
                "x_dim": inf["x_dim"],
                "edge_attr_dim": inf["edge_attr_dim"],
                "graph_attr_dim": inf["graph_attr_dim"],
                "grid_c": inf["grid_c"],
                "grid_g": inf["grid_g"],
                "matrix_key": inf["matrix_key"],
                "matrix_n": inf["matrix_n"],
            }

            if include_meta:
                row["meta_json"] = json.dumps(meta, ensure_ascii=False)

            rows.append(row)

    if not rows:
        typer.echo("No artifacts found (after filtering).")
        raise typer.Exit(code=1)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    typer.echo(f"Wrote {len(rows)} rows -> {out_csv}")
