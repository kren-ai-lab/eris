from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from eris.core.artifacts import ArtifactValidationError, StructureArtifact
from eris.export import save_artifact_to_pyg, save_artifact_to_torch
from eris.pipelines.manifest import BatchResultRow, ManifestValidationError, read_manifest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _SuccessfulBatchArtifact:
    row: BatchResultRow
    artifact_dir: Path
    artifact: StructureArtifact


def _exporter_for(format: str):
    normalized = format.lower()
    if normalized == "torch":
        return save_artifact_to_torch, "data.pt"
    if normalized in {"pyg", "pytorchgeometric", "geometric"}:
        return save_artifact_to_pyg, "data_pyg.pt"
    raise ValueError("format must be one of: 'torch', 'pyg'")


def _row_prefix(index: int) -> str:
    return f"row {index}"


def _load_successful_artifacts(
    rows: list[BatchResultRow],
) -> tuple[list[_SuccessfulBatchArtifact], int]:
    successes: list[_SuccessfulBatchArtifact] = []
    failed_rows = 0
    seen_results: dict[tuple[str, str], int] = {}
    seen_artifact_dirs: dict[Path, int] = {}

    for index, row in enumerate(rows, start=1):
        result_key = (row.id, row.representation_name)
        if result_key in seen_results:
            first = seen_results[result_key]
            raise ManifestValidationError(
                f"{_row_prefix(index)}: duplicate result for id={row.id!r} "
                f"and representation_name={row.representation_name!r} "
                f"first seen at row {first}"
            )
        seen_results[result_key] = index

        if row.status == "failed":
            failed_rows += 1
            continue

        if row.artifact_dir is None:
            raise ManifestValidationError(
                f"{_row_prefix(index)}: success row requires artifact_dir"
            )

        artifact_dir = Path(row.artifact_dir)
        artifact_dir_key = artifact_dir.resolve(strict=False)
        if artifact_dir_key in seen_artifact_dirs:
            first = seen_artifact_dirs[artifact_dir_key]
            raise ManifestValidationError(
                f"{_row_prefix(index)}: duplicate artifact_dir {str(artifact_dir)!r} "
                f"first seen at row {first}"
            )
        seen_artifact_dirs[artifact_dir_key] = index

        try:
            artifact = StructureArtifact.load(artifact_dir, strict=True)
        except (ArtifactValidationError, OSError, ValueError) as error:
            raise ManifestValidationError(
                f"{_row_prefix(index)}: artifact_dir must contain a strict-valid "
                f"Eris artifact: {artifact_dir}"
            ) from error

        meta = artifact.meta
        actual_name = meta.get("representation_name")
        actual_family = meta.get("representation_family")
        if actual_name != row.representation_name:
            raise ManifestValidationError(
                f"{_row_prefix(index)}: result representation_name "
                f"{row.representation_name!r} does not match artifact "
                f"{actual_name!r}"
            )
        if actual_family != row.representation_family:
            raise ManifestValidationError(
                f"{_row_prefix(index)}: result representation_family "
                f"{row.representation_family!r} does not match artifact "
                f"{actual_family!r}"
            )

        successes.append(
            _SuccessfulBatchArtifact(row=row, artifact_dir=artifact_dir, artifact=artifact)
        )

    return successes, failed_rows


def batch_export(
    manifest_path: Path,
    out: Path | None = None,
    *,
    format: str = "torch",
    overwrite: bool = False,
    fail_fast: bool = False,
    prefer: str = "grid_or_graph",
    device: str | None = None,
    matrix_as_channel: bool = True,
) -> dict[str, int]:
    """
    Export successful artifacts from a canonical batch-result manifest.

    ``manifest_path`` must use ``BatchResultRow`` fields. Failed rows are counted explicitly
    and never treated as artifacts. Successful rows must point to strict-valid Artifact Schema
    directories whose representation identity matches the result row.
    """
    exporter, out_name = _exporter_for(format)
    rows = read_manifest(manifest_path, BatchResultRow)
    successful_artifacts, failed_rows = _load_successful_artifacts(rows)

    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    stats = {"tasks": len(rows), "ok": 0, "failed": failed_rows, "skipped": 0}

    for item in successful_artifacts:
        row = item.row
        artifact_dir = item.artifact_dir
        outdir = artifact_dir if out is None else out / row.id / row.representation_name
        out_path = outdir / out_name

        logger.debug(
            "Processing artifact export",
            extra={
                "artifact_id": row.id,
                "representation_name": row.representation_name,
                "representation_family": row.representation_family,
                "artifact_dir": str(artifact_dir),
                "out_path": str(out_path),
                "format": format,
            },
        )

        if not overwrite and out_path.exists():
            stats["skipped"] += 1
            logger.debug(
                "Skipping existing exported artifact",
                extra={
                    "artifact_id": row.id,
                    "representation_name": row.representation_name,
                    "out_path": str(out_path),
                },
            )
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
        except Exception:
            logger.exception(
                "Artifact export failed",
                extra={
                    "artifact_id": row.id,
                    "representation_name": row.representation_name,
                    "artifact_dir": str(artifact_dir),
                    "out_path": str(out_path),
                    "format": format,
                },
            )
            stats["failed"] += 1
            if fail_fast:
                raise

    return stats


__all__ = ["batch_export"]
