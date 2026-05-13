from __future__ import annotations

import logging
logger = logging.getLogger(__name__)

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from morpheo.core.registry import get_encoder
from morpheo.core.types import StructureInput


@dataclass
class ManifestRow:
    id: str
    path: str
    chain: Optional[str] = None
    model: int = 0

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ManifestRow":
        path = str(d.get("path") or d.get("input") or "").strip()
        if not path:
            raise ValueError("Manifest row missing 'path' (or 'input').")

        sid = str(d.get("id") or d.get("sample_id") or "").strip()
        if not sid:
            sid = Path(path).stem

        chain = str(d.get("chain") or d.get("chain_id") or "").strip() or None

        model_raw = d.get("model") or d.get("model_index") or 0
        try:
            model = int(model_raw)
        except Exception:
            model = 0

        return ManifestRow(id=sid, path=path, chain=chain, model=model)


def read_manifest(path: Path) -> List[ManifestRow]:
    """
    Supported formats:
      - .csv with headers: id,path,chain,model (chain/model optional)
      - .jsonl where each line is a dict with keys like: id/path/chain/model
      - .json list[dict]
    """
    suf = path.suffix.lower()

    if suf == ".csv":
        rows: List[ManifestRow] = []
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for d in reader:
                rows.append(ManifestRow.from_dict(d))
        if not rows:
            raise ValueError(f"Empty CSV manifest: {path}")
        return rows

    if suf == ".jsonl":
        rows = []
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(ManifestRow.from_dict(json.loads(line)))
        if not rows:
            raise ValueError(f"Empty JSONL manifest: {path}")
        return rows

    if suf == ".json":
        obj = json.loads(path.read_text())
        if not isinstance(obj, list):
            raise ValueError("JSON manifest must be a list of objects.")
        rows = [ManifestRow.from_dict(d) for d in obj]
        if not rows:
            raise ValueError(f"Empty JSON manifest: {path}")
        return rows

    raise ValueError(f"Unsupported manifest format: {path.suffix} (use .csv, .jsonl, or .json)")


def _apply_params(enc: Any, params: Optional[Dict[str, Any]]) -> None:
    if not params:
        return
    # soft-apply: only set attributes that exist
    for k, v in params.items():
        if hasattr(enc, k):
            setattr(enc, k, v)


# def batch_encode(
#     manifest_path: Path,
#     encoders: Sequence[str],
#     out: Path,
#     *,
#     overwrite: bool = False,
#     no_cache: bool = False,
#     params: Optional[Dict[str, Dict[str, Any]]] = None,
#     fail_fast: bool = False,
# ) -> Dict[str, int]:
#     """
#     Run multiple encoders over a manifest and save artifacts under:
#       out/<id>/<encoder>/{data.npz,meta.json}

#     params:
#       dict like:
#         {
#           "voxel_grid_ca": {"grid_size": 64, "sigma": 1.5},
#           "residue_knn_graph_ns": {"k": 16}
#         }
#     """
#     rows = read_manifest(manifest_path)
#     out.mkdir(parents=True, exist_ok=True)

#     stats = {"processed": 0, "ok": 0, "failed": 0, "skipped": 0}

#     for r in rows:
#         inp = StructureInput(source=r.path, chain_id=r.chain, model_index=r.model)

#         for enc_name in encoders:
#             stats["processed"] += 1
#             outdir = out / r.id / enc_name
#             data_npz = outdir / "data.npz"

#             if (not overwrite) and data_npz.exists():
#                 stats["skipped"] += 1
#                 continue

#             try:
#                 enc = get_encoder(enc_name)

#                 if no_cache and hasattr(enc, "use_cache"):
#                     setattr(enc, "use_cache", False)

#                 _apply_params(enc, (params or {}).get(enc_name))

#                 outdir.mkdir(parents=True, exist_ok=True)
#                 art = enc.encode(inp)
#                 art.save(outdir)

#                 stats["ok"] += 1

#             except Exception:
#                 stats["failed"] += 1
#                 if fail_fast:
#                     raise

#     return stats


def batch_encode(
    manifest_path: Path,
    encoders: Sequence[str],
    out: Path,
    *,
    overwrite: bool = False,
    no_cache: bool = False,
    params: Optional[Dict[str, Dict[str, Any]]] = None,
    fail_fast: bool = False,
) -> Dict[str, int]:
    """
    Run multiple encoders over a manifest and save artifacts under:
      out/<id>/<encoder>/{data.npz,meta.json}

    params:
      dict like:
        {
          "voxel_grid_ca": {"grid_size": 64, "sigma": 1.5},
          "residue_knn_graph_ns": {"k": 16}
        }
    """
    rows = read_manifest(manifest_path)
    out.mkdir(parents=True, exist_ok=True)

    stats = {"tasks": 0, "ok": 0, "failed": 0, "skipped": 0, "encoder_failed": 0}
    built_encoders = {}
    
    for name in encoders:
        try:
            kwargs = (params or {}).get(name, {})
            enc = get_encoder(name, **kwargs)

            if no_cache and hasattr(enc, "use_cache"):
                enc.use_cache = False

            built_encoders[name] = enc

        except Exception:
            logger.exception(f"Failed to build encoder {name}")
            
            stats["encoder_failed"] += 1
            if fail_fast:
                raise


    for row in rows:
        inp = StructureInput(source=str(row.path), chain_id=row.chain, model_index=row.model)
        logger.debug("Processing input", 
                     extra={
                         "input_id": row.id,
                         "source": str(row.path)})
        
        for enc_name, enc in built_encoders.items():
            stats["tasks"] += 1
            outdir = out / row.id / enc_name
            data_npz = outdir / "data.npz"

            if not overwrite and data_npz.exists():
                stats["skipped"] += 1
                
                logger.debug(
                    "Skipping existing artifact",
                    extra={
                        "input_id": row.id,
                        "encoder": enc_name
                    }
                )

                continue

            try:
                art = enc.encode(inp)
                outdir.mkdir(parents=True, exist_ok=True)
                art.save(outdir)

                stats["ok"] += 1

            except Exception:
                
                logger.exception(
                    "Encoding failed",
                    extra={
                        "input_id": row.id,
                        "encoder": enc_name,
                        "source": str(row.path)
                    }
                )

                stats["failed"] += 1
                if fail_fast:
                    raise

    return stats


            

