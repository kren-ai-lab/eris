from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eris.core.artifacts import ArtifactValidationError, StructureArtifact

PACKED_INDEX_FIELDS: tuple[str, ...] = (
    "id",
    "artifact_dir",
    "meta_path",
    "representation_name",
    "representation_family",
    "representation_type",
    "granularity",
    "n_nodes",
    "n_edges",
    "shape_summary",
)


class PackIndexError(ValueError):
    """Raised when artifact packing/index generation cannot complete safely."""


@dataclass(frozen=True)
class PackedIndexRow:
    id: str
    artifact_dir: str
    meta_path: str
    representation_name: str
    representation_family: str
    representation_type: str
    granularity: str
    n_nodes: int | None
    n_edges: int | None
    shape_summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "artifact_dir": self.artifact_dir,
            "meta_path": self.meta_path,
            "representation_name": self.representation_name,
            "representation_family": self.representation_family,
            "representation_type": self.representation_type,
            "granularity": self.granularity,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "shape_summary": self.shape_summary,
        }


def _ensure_root(path: str | Path) -> Path:
    root = Path(path)
    if not root.exists():
        raise PackIndexError(f"Artifact root does not exist: {root}")
    if not root.is_dir():
        raise PackIndexError(f"Artifact root is not a directory: {root}")
    return root


def _posix_relative(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    if not relative.parts:
        return "."
    return relative.as_posix()


def _posix_locator(path: Path, anchor: Path) -> str:
    relative = os.path.relpath(
        path.resolve(strict=False),
        start=anchor.resolve(strict=False),
    )
    return Path(relative).as_posix()


def _meta_path_for(relative_artifact_dir: str) -> str:
    if relative_artifact_dir == ".":
        return "meta.json"
    return f"{relative_artifact_dir}/meta.json"


def _artifact_id(
    artifact_dir: Path,
    root: Path,
    *,
    representation_name: str,
) -> str:
    relative = artifact_dir.relative_to(root)
    if not relative.parts:
        return artifact_dir.name
    if len(relative.parts) >= 2 and relative.parts[-1] == representation_name:
        return Path(*relative.parts[:-1]).as_posix()
    return relative.as_posix()


def _shape_summary(meta: dict[str, Any]) -> str:
    shapes = meta.get("shapes")
    if not isinstance(shapes, dict):
        shapes = {}
    return json.dumps(shapes, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _optional_count(meta: dict[str, Any], key: str) -> int | None:
    counts = meta.get("counts")
    if not isinstance(counts, dict) or key not in counts:
        return None
    value = counts[key]
    if type(value) is not int:
        raise PackIndexError(f"meta.counts.{key} must be an integer when present")
    return value


def _candidate_artifact_dirs(root: Path) -> list[Path]:
    candidates: list[Path] = []
    incomplete: list[Path] = []

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        dirnames[:] = [dirname for dirname in dirnames if not (current / dirname).is_symlink()]

        names = set(filenames)
        has_meta = "meta.json" in names
        has_data = "data.npz" in names
        if has_meta and has_data:
            candidates.append(current)
        elif has_meta or has_data:
            incomplete.append(current)

    if incomplete:
        ordered_incomplete = sorted(incomplete, key=lambda item: item.as_posix())
        relative = ", ".join(_posix_relative(path, root) for path in ordered_incomplete)
        raise PackIndexError(
            "Incomplete artifact directories found; each candidate needs meta.json "
            f"and data.npz: {relative}"
        )

    return sorted(candidates, key=lambda item: _posix_relative(item, root))


def _row_from_artifact(root: Path, locator_root: Path, artifact_dir: Path) -> PackedIndexRow:
    try:
        artifact = StructureArtifact.load(artifact_dir, strict=True)
    except ArtifactValidationError as error:
        raise PackIndexError(
            f"Invalid Artifact Schema v2 directory: {_posix_relative(artifact_dir, root)}"
        ) from error

    meta = artifact.meta
    representation_name = meta["representation_name"]
    relative_artifact_dir = _posix_locator(artifact_dir, locator_root)
    return PackedIndexRow(
        id=_artifact_id(
            artifact_dir,
            root,
            representation_name=representation_name,
        ),
        artifact_dir=relative_artifact_dir,
        meta_path=_meta_path_for(relative_artifact_dir),
        representation_name=representation_name,
        representation_family=meta["representation_family"],
        representation_type=meta["representation_type"],
        granularity=meta["granularity"],
        n_nodes=_optional_count(meta, "n_nodes"),
        n_edges=_optional_count(meta, "n_edges"),
        shape_summary=_shape_summary(meta),
    )


def _validate_index_rows(rows: list[PackedIndexRow], root: Path) -> None:
    seen_id_representation: dict[tuple[str, str], int] = {}
    seen_artifact_dirs: dict[Path, int] = {}

    for index, row in enumerate(rows, start=1):
        identity = (row.id, row.representation_name)
        if identity in seen_id_representation:
            first = seen_id_representation[identity]
            raise PackIndexError(
                f"row {index}: duplicate artifact for id={row.id!r} and "
                f"representation_name={row.representation_name!r} first seen at row {first}"
            )
        seen_id_representation[identity] = index

        artifact_dir = root / row.artifact_dir
        resolved = artifact_dir.resolve(strict=False)
        if resolved in seen_artifact_dirs:
            first = seen_artifact_dirs[resolved]
            raise PackIndexError(
                f"row {index}: duplicate artifact_dir {row.artifact_dir!r} "
                f"first seen at row {first}"
            )
        seen_artifact_dirs[resolved] = index


def _csv_row(row: PackedIndexRow) -> dict[str, Any]:
    serialized = row.to_dict()
    return {
        field: "" if serialized[field] is None else serialized[field]
        for field in PACKED_INDEX_FIELDS
    }


def _write_index_safely(path: Path, rows: list[PackedIndexRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{path.stem}.tmp-",
        suffix=path.suffix,
        dir=path.parent,
    )
    os.close(descriptor)
    staging = Path(staging_name)
    try:
        with staging.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=PACKED_INDEX_FIELDS)
            writer.writeheader()
            writer.writerows(_csv_row(row) for row in rows)
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def pack_artifacts(
    artifact_root: str | Path,
    output_path: str | Path,
) -> list[PackedIndexRow]:
    """
    Build a canonical metadata-only packed index for strict-valid Artifact Schema v2 dirs.

    The function never rewrites artifact ``meta.json`` or ``data.npz`` files. ``output_path``
    is replaced atomically only after discovery, validation, duplicate checks, and full CSV
    serialization succeed.
    """
    root = _ensure_root(artifact_root)
    output = Path(output_path)
    locator_root = output.parent
    candidates = _candidate_artifact_dirs(root)
    rows = [_row_from_artifact(root, locator_root, candidate) for candidate in candidates]
    _validate_index_rows(rows, locator_root)
    _write_index_safely(output, rows)
    return rows


__all__ = [
    "PACKED_INDEX_FIELDS",
    "PackIndexError",
    "PackedIndexRow",
    "pack_artifacts",
]
