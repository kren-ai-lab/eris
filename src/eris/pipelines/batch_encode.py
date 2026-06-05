from __future__ import annotations

import logging
logger = logging.getLogger(__name__)

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from eris.core.registry import get_encoder
from eris.core.types import StructureInput
from eris.pipelines.manifest import read_manifest, StructureManifestRow



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
    rows = read_manifest(manifest_path, StructureManifestRow)
    out.mkdir(parents=True, exist_ok=True)
    out_manifest_path = out / "artifact_manifest.csv"
    out_manifest_rows = []

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

                out_manifest_rows.append({
                    "id": row.id+"_"+enc_name,
                    "artifact_dir": str(outdir)
                    })

                continue

            try:
                art = enc.encode(inp)
                outdir.mkdir(parents=True, exist_ok=True)
                art.save(outdir)

                stats["ok"] += 1

                out_manifest_rows.append({
                    "id": row.id+"_"+enc_name,
                    "artifact_dir": str(outdir)
                    })

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


    with open(out_manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "artifact_dir"])
        writer.writeheader()

        for row in out_manifest_rows:
            writer.writerow(row)   

    return stats


            

