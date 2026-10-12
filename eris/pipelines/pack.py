from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from eris.core.artifacts import ArtifactValidationError, StructureArtifact
from eris.pipelines.manifest import row_location, write_atomically

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

    @classmethod
    def from_dict(
        cls,
        row: dict[str, Any],
        *,
        row_index: int | None = None,
    ) -> PackedIndexRow:
        return cls(
            id=_required_index_string(row.get("id"), "id", row_index=row_index),
            artifact_dir=_required_locator(
                row.get("artifact_dir"),
                "artifact_dir",
                row_index=row_index,
            ),
            meta_path=_required_locator(row.get("meta_path"), "meta_path", row_index=row_index),
            representation_name=_required_index_string(
                row.get("representation_name"),
                "representation_name",
                row_index=row_index,
            ),
            representation_family=_literal_index_value(
                row.get("representation_family"),
                "representation_family",
                row_index=row_index,
                choices={"explicit", "learned"},
            ),
            representation_type=_literal_index_value(
                row.get("representation_type"),
                "representation_type",
                row_index=row_index,
                choices={"cloud", "embedding", "graph", "grid", "matrix"},
            ),
            granularity=_literal_index_value(
                row.get("granularity"),
                "granularity",
                row_index=row_index,
                choices={"atom", "mixed", "residue", "structure"},
            ),
            n_nodes=_optional_index_count(row.get("n_nodes"), "n_nodes", row_index=row_index),
            n_edges=_optional_index_count(row.get("n_edges"), "n_edges", row_index=row_index),
            shape_summary=_required_shape_summary(
                row.get("shape_summary"),
                "shape_summary",
                row_index=row_index,
            ),
        )


def _ensure_root(path: str | Path) -> Path:
    root = Path(path)
    if not root.exists():
        raise PackIndexError(f"Artifact root does not exist: {root}")
    if not root.is_dir():
        raise PackIndexError(f"Artifact root is not a directory: {root}")
    return root


def _row_location(row_index: int | None) -> str:
    return row_location(row_index, "packed index row")


def _required_index_string(value: Any, field: str, *, row_index: int | None) -> str:
    if not isinstance(value, str):
        raise PackIndexError(f"{_row_location(row_index)}: field {field!r} must be a string")
    text = value.strip()
    if not text:
        raise PackIndexError(f"{_row_location(row_index)}: field {field!r} must be non-empty")
    return text


def _is_absolute_locator(value: str) -> bool:
    return PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _required_locator(value: Any, field: str, *, row_index: int | None) -> str:
    text = _required_index_string(value, field, row_index=row_index)
    if _is_absolute_locator(text):
        raise PackIndexError(f"{_row_location(row_index)}: field {field!r} must be relative")
    if "\\" in text:
        raise PackIndexError(f"{_row_location(row_index)}: field {field!r} must use POSIX '/' separators")
    return text


def _literal_index_value(
    value: Any,
    field: str,
    *,
    row_index: int | None,
    choices: set[str],
) -> str:
    text = _required_index_string(value, field, row_index=row_index)
    if text not in choices:
        expected = ", ".join(sorted(choices))
        raise PackIndexError(f"{_row_location(row_index)}: field {field!r} must be one of: {expected}")
    return text


def _optional_index_count(value: Any, field: str, *, row_index: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.isdecimal():
            return int(text)
    if type(value) is int and value >= 0:
        return value
    raise PackIndexError(f"{_row_location(row_index)}: field {field!r} must be empty or an integer >= 0")


def parse_shape_summary(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("shape_summary must decode to an object")
    return parsed


def _required_shape_summary(value: Any, field: str, *, row_index: int | None) -> str:
    text = _required_index_string(value, field, row_index=row_index)
    try:
        parse_shape_summary(text)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise PackIndexError(
            f"{_row_location(row_index)}: field {field!r} must be valid JSON object text"
        ) from error
    return text


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


def resolve_packed_locator(index_path: str | Path, locator: str) -> Path:
    parts = PurePosixPath(locator).parts
    return Path(index_path).parent.joinpath(*parts)


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


def shape_summary_from_metadata(meta: dict[str, Any]) -> str:
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
            f"Incomplete artifact directories found; each candidate needs meta.json and data.npz: {relative}"
        )

    return sorted(candidates, key=lambda item: _posix_relative(item, root))


def _row_from_artifact(root: Path, locator_root: Path, artifact_dir: Path) -> PackedIndexRow:
    try:
        artifact = StructureArtifact.load(artifact_dir, strict=True)
    except ArtifactValidationError as error:
        raise PackIndexError(
            f"Invalid Artifact Schema directory: {_posix_relative(artifact_dir, root)}"
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
        shape_summary=shape_summary_from_metadata(meta),
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
                f"row {index}: duplicate artifact_dir {row.artifact_dir!r} first seen at row {first}"
            )
        seen_artifact_dirs[resolved] = index


def validate_packed_index_rows(rows: list[PackedIndexRow], locator_root: Path) -> None:
    """Validate duplicate/ambiguity constraints for parsed packed index rows."""
    _validate_index_rows(rows, locator_root)


def _csv_row(row: PackedIndexRow) -> dict[str, Any]:
    serialized = row.to_dict()
    return {field: "" if serialized[field] is None else serialized[field] for field in PACKED_INDEX_FIELDS}


def _write_index_safely(path: Path, rows: list[PackedIndexRow]) -> None:
    def write(staging: Path) -> None:
        with staging.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=PACKED_INDEX_FIELDS)
            writer.writeheader()
            writer.writerows(_csv_row(row) for row in rows)

    write_atomically(path, write)


def _validate_packed_index_header(fieldnames: list[str] | None) -> None:
    if fieldnames is None:
        raise PackIndexError("Packed index CSV is missing a header")
    missing = [field for field in PACKED_INDEX_FIELDS if field not in fieldnames]
    if missing:
        joined = ", ".join(missing)
        raise PackIndexError(f"Packed index CSV is missing required columns: {joined}")
    if tuple(fieldnames[: len(PACKED_INDEX_FIELDS)]) != PACKED_INDEX_FIELDS:
        expected = ",".join(PACKED_INDEX_FIELDS)
        actual = ",".join(fieldnames)
        raise PackIndexError(
            f"Packed index CSV canonical columns must appear first in exact order: {expected}; got: {actual}"
        )


def read_packed_index(path: str | Path) -> list[PackedIndexRow]:
    index_path = Path(path)
    if not index_path.exists():
        raise PackIndexError(f"Packed index does not exist: {index_path}")
    if not index_path.is_file():
        raise PackIndexError(f"Packed index is not a file: {index_path}")

    with index_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _validate_packed_index_header(reader.fieldnames)
        rows = [
            PackedIndexRow.from_dict(dict(row), row_index=index) for index, row in enumerate(reader, start=1)
        ]
    validate_packed_index_rows(rows, index_path.parent)
    return rows


def pack_artifacts(
    artifact_root: str | Path,
    output_path: str | Path,
) -> list[PackedIndexRow]:
    """Build a canonical metadata-only packed index for strict-valid Artifact Schema directories.

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
    "parse_shape_summary",
    "read_packed_index",
    "resolve_packed_locator",
    "shape_summary_from_metadata",
    "validate_packed_index_rows",
]
