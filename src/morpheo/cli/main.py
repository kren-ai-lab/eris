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
    encoder: Optional[str] = typer.Option(
        None,
        "--encoder",
        help="Only include encoder(s). Comma-separated allowed, e.g. residue_contact_graph_ns,residue_knn_graph_ns",
    ),
    prefer: str = typer.Option(
        "any",
        "--prefer",
        help="Filter by representation type: any|graph|graph_ns|grid|cloud|matrix",
    ),
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

    def _infer_from_npz(npz_path: Path) -> dict:
        """
        Cheap inference without loading full arrays.
        """
        info: dict = {}
        try:
            with np.load(npz_path, allow_pickle=False) as z:
                files = set(z.files)
                # dims
                if "node_attr" in files:
                    info["x_dim"] = int(z["node_attr"].shape[1])
                elif "node_emb" in files:
                    info["x_dim"] = int(z["node_emb"].shape[1])
                elif "aa_index" in files:
                    # onehot baseline (21) + optional plddt/bfactor often done later
                    info["x_dim"] = 21
                else:
                    info["x_dim"] = ""

                if "edge_attr" in files:
                    info["edge_attr_dim"] = int(z["edge_attr"].shape[1])
                else:
                    info["edge_attr_dim"] = ""

                if "graph_attr" in files:
                    info["graph_attr_dim"] = int(z["graph_attr"].reshape(-1).shape[0])
                else:
                    info["graph_attr_dim"] = ""

                # matrix key + size
                matrix_key = ""
                matrix_n = ""
                for k in ("dist", "dist_bins", "contact", "contact_bins"):
                    if k in files:
                        matrix_key = k
                        arr = z[k]
                        matrix_n = int(arr.shape[-1])
                        break
                info["matrix_key"] = matrix_key
                info["matrix_n"] = matrix_n

                # grid shape
                if "grid" in files:
                    info["grid_shape"] = "x".join(str(x) for x in z["grid"].shape)
                else:
                    info["grid_shape"] = ""

                # graph marker
                info["has_graph"] = "edge_index" in files
                info["has_grid"] = "grid" in files
                info["has_cloud"] = "pos" in files
                info["has_matrix"] = bool(matrix_key)

        except Exception:
            # best effort
            info.update(
                {
                    "x_dim": "",
                    "edge_attr_dim": "",
                    "graph_attr_dim": "",
                    "matrix_key": "",
                    "matrix_n": "",
                    "grid_shape": "",
                    "has_graph": False,
                    "has_grid": False,
                    "has_cloud": False,
                    "has_matrix": False,
                }
            )
        return info

    def _prefer_ok(enc: str, meta: dict, inferred: dict) -> bool:
        # Try registry-style meta if present
        enc_type = meta.get("type") or meta.get("encoder_type") or meta.get("repr_type")
        if isinstance(enc_type, str):
            enc_type = enc_type.strip().lower()
        else:
            enc_type = None

        family = meta.get("family")
        if isinstance(family, str):
            family = family.strip().lower()
        else:
            family = None

        # Fallback to shapes
        shapes = meta.get("shapes", {})
        if not isinstance(shapes, dict):
            shapes = {}

        has_graph = bool(inferred.get("has_graph")) or ("edge_index" in shapes)
        has_grid = bool(inferred.get("has_grid")) or ("grid" in shapes)
        has_cloud = bool(inferred.get("has_cloud")) or ("pos" in shapes)
        has_matrix = bool(inferred.get("has_matrix")) or any(k in shapes for k in ("dist", "dist_bins", "contact", "contact_bins"))

        # Resolve prefer
        if prefer == "any":
            return True

        if prefer == "graph":
            return (enc_type == "graph") or has_graph

        if prefer == "graph_ns":
            # rule: encoder name endswith _ns OR family == "residue" + presence of node_attr/edge_attr/graph_attr
            if enc.endswith("_ns"):
                return True
            # fallback: require graph + node_attr present
            return has_graph and bool(inferred.get("x_dim", "")) and ("node_attr" in shapes or meta.get("has_node_attr") is True)

        if prefer == "grid":
            return (enc_type == "grid") or has_grid

        if prefer == "cloud":
            # cloud should be pos-only (not graph/grid) in our worldview
            if enc_type == "cloud":
                return True
            return has_cloud and (not has_graph) and (not has_grid) and (not has_matrix)

        if prefer == "matrix":
            return (enc_type == "matrix") or has_matrix

        return True

    rows = []
    sid_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    for sid_dir in sid_dirs:
        sid = sid_dir.name
        enc_dirs = sorted([p for p in sid_dir.iterdir() if p.is_dir()])

        for enc_dir in enc_dirs:
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

            inferred = _infer_from_npz(data_path)

            if not _prefer_ok(enc, meta, inferred):
                continue

            shapes = meta.get("shapes", {}) if isinstance(meta.get("shapes", {}), dict) else {}
            n_res = meta.get("n_residues", "")
            n_edges = meta.get("n_edges", "")

            row = {
                "id": sid,
                "encoder": enc,
                "artifact_dir": str(enc_dir),
                "data_path": str(data_path),
                "meta_path": str(meta_path),
                "n_residues": n_res,
                "n_edges": n_edges,
                # dims to avoid collate surprises
                "x_dim": inferred.get("x_dim", ""),
                "edge_attr_dim": inferred.get("edge_attr_dim", ""),
                "graph_attr_dim": inferred.get("graph_attr_dim", ""),
                # matrix details (if any)
                "matrix_key": inferred.get("matrix_key", ""),
                "matrix_n": inferred.get("matrix_n", ""),
                # grid details (if any)
                "grid_shape": inferred.get("grid_shape", ""),
            }

            # optional meta
            if include_meta:
                row["meta_json"] = json.dumps(meta, ensure_ascii=False)

            rows.append(row)

    if not rows:
        typer.echo("No artifacts found (after filtering).")
        raise typer.Exit(code=1)

    # Stable field order (important)
    base_fields = [
        "id",
        "encoder",
        "artifact_dir",
        "data_path",
        "meta_path",
        "n_residues",
        "n_edges",
        "x_dim",
        "edge_attr_dim",
        "graph_attr_dim",
        "matrix_key",
        "matrix_n",
        "grid_shape",
    ]
    if include_meta:
        base_fields.append("meta_json")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=base_fields)
        w.writeheader()
        w.writerows(rows)

    typer.echo(f"Wrote {len(rows)} rows -> {out_csv}")
