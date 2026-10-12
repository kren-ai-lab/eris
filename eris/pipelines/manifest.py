"""Batch manifest reading, validation and writing."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from eris.exceptions import ErisError


class ManifestValidationError(ErisError, ValueError):
    """Raised when a manifest row or manifest file violates the canonical contract."""


def row_location(row_index: int | None, none_label: str = "manifest row") -> str:
    if row_index is None:
        return none_label
    return f"row {row_index}"


def write_atomically(path: Path, write: Callable[[Path], None]) -> None:
    """Write via a sibling temp file and ``os.replace`` so readers never see partial output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{path.stem}.tmp-",
        suffix=path.suffix,
        dir=path.parent,
    )
    os.close(descriptor)
    staging = Path(staging_name)
    try:
        write(staging)
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)


def _required_string(value: Any, *, field: str, row_index: int | None = None) -> str:
    if not isinstance(value, str):
        location = row_location(row_index)
        raise ManifestValidationError(f"{location}: field '{field}' must be a string")

    text = value.strip()
    if not text:
        location = row_location(row_index)
        raise ManifestValidationError(f"{location}: field '{field}' must be non-empty")

    return text


def _required_text_path(value: Any, *, field: str, row_index: int | None = None) -> str:
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    return _required_string(value, field=field, row_index=row_index)


def _chain_selection(value: Any, *, field: str, row_index: int | None = None) -> tuple[str, ...] | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                value = json.loads(text)
            except json.JSONDecodeError as error:
                location = row_location(row_index)
                raise ManifestValidationError(
                    f"{location}: field '{field}' contains invalid JSON chain list"
                ) from error
        else:
            value = (text,)
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        location = row_location(row_index)
        raise ManifestValidationError(
            f"{location}: field '{field}' must be a string or non-empty list of strings"
        )
    chains = tuple(value)
    location = row_location(row_index)
    if not chains or any(not isinstance(chain, str) for chain in chains):
        raise ManifestValidationError(
            f"{location}: field '{field}' must be a string or non-empty list of strings"
        )
    if len(chains) != len(set(chains)):
        raise ManifestValidationError(f"{location}: field '{field}' contains duplicate chains")
    return chains


def _model(value: Any, *, field: str, row_index: int | None = None) -> int:
    if value is None:
        return 0
    if isinstance(value, str) and not value.strip():
        return 0
    if isinstance(value, bool):
        location = row_location(row_index)
        raise ManifestValidationError(f"{location}: field '{field}' must be an integer >= 0")
    if isinstance(value, int):
        model = value
    elif isinstance(value, str) and value.strip().isdecimal():
        model = int(value.strip())
    else:
        location = row_location(row_index)
        raise ManifestValidationError(f"{location}: field '{field}' must be an integer >= 0")

    if model < 0:
        location = row_location(row_index)
        raise ManifestValidationError(f"{location}: field '{field}' must be an integer >= 0")
    return model


STRUCTURE_MANIFEST_FIELDS = ("id", "path", "chain", "model")
BATCH_RESULT_SCHEMA = pl.Schema(
    {
        "id": pl.String,
        "representation_name": pl.String,
        "representation_family": pl.String,
        "artifact_dir": pl.String,
        "status": pl.String,
        "error_type": pl.String,
        "error_message": pl.String,
    }
)
_FORMATS = "use .csv or .jsonl"


@dataclass(frozen=True)
class StructureManifestRow:
    """One validated input structure in a batch manifest."""

    id: str
    path: str
    chain: tuple[str, ...] | None = None
    model: int = 0

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], *, row_index: int | None = None) -> StructureManifestRow:
        """Build a row from a mapping, rejecting unexpected fields."""
        unexpected = sorted(set(d) - set(STRUCTURE_MANIFEST_FIELDS))
        if unexpected:
            location = row_location(row_index)
            fields = ", ".join(STRUCTURE_MANIFEST_FIELDS)
            raise ManifestValidationError(
                f"{location}: unexpected field(s) {', '.join(unexpected)!r}; use canonical fields: {fields}"
            )
        return cls(
            id=_required_string(d.get("id"), field="id", row_index=row_index),
            path=_required_text_path(d.get("path"), field="path", row_index=row_index),
            chain=_chain_selection(d.get("chain"), field="chain", row_index=row_index),
            model=_model(d.get("model"), field="model", row_index=row_index),
        )


def read_csv_as_text(path: Path) -> pl.DataFrame:
    """Read a CSV keeping every cell as text, with empty cells as empty strings."""
    return pl.read_csv(path, infer_schema=False, empty_string_is_null=False, raise_if_empty=False)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ManifestValidationError(f"line {line_number}: JSONL manifest entries must be objects")
            rows.append(obj)
    return rows


def read_manifest(manifest: str | Path | pl.DataFrame) -> list[StructureManifestRow]:
    """Validate a structure manifest given as a CSV/JSONL path or a DataFrame.

    Canonical fields are ``id,path,chain,model``. CSV cells are read as text, so no
    value is type-inferred or coerced before validation.
    """
    if isinstance(manifest, pl.DataFrame):
        raw_rows = manifest.to_dicts()
        source: object = "DataFrame"
    else:
        path = Path(manifest)
        source = path
        suffix = path.suffix.lower()
        if suffix == ".csv":
            try:
                raw_rows = read_csv_as_text(path).to_dicts()
            except pl.exceptions.PolarsError as error:
                raise ManifestValidationError(f"Invalid CSV manifest {path}: {error}") from error
        elif suffix == ".jsonl":
            raw_rows = _read_jsonl(path)
        else:
            raise ManifestValidationError(f"Unsupported manifest format: {path.suffix} ({_FORMATS})")

    if not raw_rows:
        raise ManifestValidationError(f"Empty manifest: {source}")

    rows = [StructureManifestRow.from_dict(raw, row_index=index) for index, raw in enumerate(raw_rows, 1)]
    seen: dict[str, int] = {}
    for index, row in enumerate(rows, 1):
        if row.id in seen:
            raise ManifestValidationError(
                f"row {index}: duplicate id '{row.id}' first seen at row {seen[row.id]}"
            )
        seen[row.id] = index
    return rows


def write_results(path: Path, results: pl.DataFrame) -> None:
    """Write a batch-result frame as CSV or JSONL, chosen by the file suffix."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        results.write_csv(path)
    elif suffix == ".jsonl":
        results.write_ndjson(path)
    else:
        raise ManifestValidationError(f"Unsupported results format: {path.suffix} ({_FORMATS})")


__all__ = [
    "BATCH_RESULT_SCHEMA",
    "ManifestValidationError",
    "StructureManifestRow",
    "read_manifest",
    "write_results",
]
