from __future__ import annotations

import logging
logger = logging.getLogger(__name__)

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from eris.core.registry import get_encoder
from eris.core.types import StructureInput
from eris.export import save_artifact_to_pyg, save_artifact_to_torch
from eris.pipelines.manifest import read_manifest, ArtifactManifestRow

def batch_encode(
    manifest_path: Path,
    out: Path,
    *,
    format: str = "torch",
    overwrite: bool = False,
    fail_fast: bool = False,
    prefer: str,
    device: Optional[str],
    matrix_as_channel: bool,
) -> Dict[str, int]:
    """

    """
    rows = read_manifest(manifest_path, ArtifactManifestRow)
    out.mkdir(parents=True, exist_ok=True)
    # out_manifest_path = out / "artifact_manifest.csv"
    # out_manifest_rows = []

    stats = {"tasks": 0, "ok": 0, "failed": 0, "skipped": 0}

    if format == "torch":
        exporter = save_artifact_to_torch
        out_name = "data.pt"
    elif format.lower() in ["pyg", "pytorchgeometric", "geometric"]:
        exporter = save_artifact_to_pyg
        out_name = "data_pyg.pt"

    for row in rows:
        outdir = out / row.id
        data_npz = outdir / out_name

        logger.debug("Processing input", 
                extra={
                    "artifact_id": row.id,
                    "source": str(row.artifact_dir)})

        stats["tasks"] += 1
        
        if not overwrite and data_npz.exists():
            stats["skipped"] += 1

            logger.debug(
                "Skipping existing torch/pyg data",
                extra={
                    "input_id": row.id,
                }
            )

            continue

        try:
            exporter(row.artifact_dir, out_path=out, prefer=prefer, device=device, matrix_as_channel=matrix_as_channel)

        except Exception:
            logger.exception(
                "Encoding failed",
                extra={
                    "input_id": row.id,
                    "source": str(row.artifact_dir)
                }
            )

            stats["failed"] += 1
            if fail_fast:
                raise



def batch_export(
    manifest_path: Path,
    out: Optional[Path] = None,
    *,
    format: str = "torch",
    overwrite: bool = False,
    fail_fast: bool = False,
    prefer: str = "grid_or_graph",
    device: Optional[str] = None,
    matrix_as_channel: bool = True,
) -> Dict[str, int]:
    """
    Batch export of previously encoded artifacts.

    Parameters
    ----------
    manifest_path
        Manifest of artifacts (id, artifact_dir).
    out
        Optional output root directory. If None, saves next to each artifact.
    format
        "torch" or "pyg".
    overwrite
        Whether to overwrite existing exported files.
    fail_fast
        If True, raise immediately on first failure.
    prefer, device, matrix_as_channel
        Forwarded to exporter.
    """
    rows = read_manifest(manifest_path, ArtifactManifestRow)

    if format.lower() == "torch":
        exporter = save_artifact_to_torch
        out_name = "data.pt"
    elif format.lower() in {"pyg", "pytorchgeometric", "geometric"}:
        exporter = save_artifact_to_pyg
        out_name = "data_pyg.pt"
    else:
        raise ValueError("format must be one of: 'torch', 'pyg'")

    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        out_manifest_path = out / "export_manifest.csv"
    else:
        out_manifest_path = None

    out_manifest_rows = []
    stats = {"tasks": 0, "ok": 0, "failed": 0, "skipped": 0}

    for row in rows:
        artifact_dir = Path(row.artifact_dir)

        if out is None:
            outdir = artifact_dir
        else:
            outdir = out / row.id

        out_path = outdir / out_name

        logger.debug(
            "Processing artifact export",
            extra={
                "artifact_id": row.id,
                "artifact_dir": str(artifact_dir),
                "out_path": str(out_path),
                "format": format,
            },
        )

        stats["tasks"] += 1

        if not overwrite and out_path.exists():
            stats["skipped"] += 1

            logger.debug(
                "Skipping existing exported artifact",
                extra={
                    "artifact_id": row.id,
                    "out_path": str(out_path),
                },
            )

            out_manifest_rows.append({
                "id": row.id,
                "artifact_dir": str(out_path),
            })
            continue

        try:
            outdir.mkdir(parents=True, exist_ok=True)

            exporter(
                artifact_dir,
                out_path=out_path,
                prefer=prefer,
                device=device,
                matrix_as_channel=matrix_as_channel,
            )

            stats["ok"] += 1

            out_manifest_rows.append({
                "id": row.id,
                "artifact_dir": str(out_path),
            })

        except Exception:
            logger.exception(
                "Artifact export failed",
                extra={
                    "artifact_id": row.id,
                    "artifact_dir": str(artifact_dir),
                    "out_path": str(out_path),
                    "format": format,
                },
            )

            stats["failed"] += 1
            if fail_fast:
                raise

    if out_manifest_path is not None:
        with open(out_manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "artifact_dir"])
            writer.writeheader()
            for row in out_manifest_rows:
                writer.writerow(row)

    return stats


