from __future__ import annotations

import csv
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, TypeVar


class ManifestValidationError(ValueError):
    """Raised when a manifest row or manifest file violates the canonical contract."""


def _row_location(row_index: int | None) -> str:
    if row_index is None:
        return "manifest row"
    return f"row {row_index}"


def _raw_field(
    row: Mapping[str, Any],
    field: str,
    *,
    aliases: tuple[str, ...] = (),
    row_index: int | None = None,
    normalizer,
) -> Any:
    values = [(field, row[field])] if field in row else []
    values.extend((alias, row[alias]) for alias in aliases if alias in row)

    if not values:
        return None

    normalized = [
        (name, normalizer(value, field=field, row_index=row_index)) for name, value in values
    ]
    _, first_value = normalized[0]

    for alias_name, alias_value in normalized[1:]:
        if alias_value != first_value:
            location = _row_location(row_index)
            raise ManifestValidationError(
                f"{location}: field '{field}' conflicts with alias '{alias_name}'"
            )

    return first_value


def _required_string(value: Any, *, field: str, row_index: int | None = None) -> str:
    if not isinstance(value, str):
        location = _row_location(row_index)
        raise ManifestValidationError(f"{location}: field '{field}' must be a string")

    text = value.strip()
    if not text:
        location = _row_location(row_index)
        raise ManifestValidationError(f"{location}: field '{field}' must be non-empty")

    return text


def _required_text_path(value: Any, *, field: str, row_index: int | None = None) -> str:
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    return _required_string(value, field=field, row_index=row_index)


def _optional_string(value: Any, *, field: str, row_index: int | None = None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        location = _row_location(row_index)
        raise ManifestValidationError(f"{location}: field '{field}' must be a string")
    return value.strip() or None


def _optional_text_path(value: Any, *, field: str, row_index: int | None = None) -> str | None:
    if value is None:
        return None
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    return _optional_string(value, field=field, row_index=row_index)


def _model(value: Any, *, field: str, row_index: int | None = None) -> int:
    if value is None:
        return 0
    if isinstance(value, str) and not value.strip():
        return 0
    if isinstance(value, bool):
        location = _row_location(row_index)
        raise ManifestValidationError(f"{location}: field '{field}' must be an integer >= 0")
    if isinstance(value, int):
        model = value
    elif isinstance(value, str) and value.strip().isdecimal():
        model = int(value.strip())
    else:
        location = _row_location(row_index)
        raise ManifestValidationError(f"{location}: field '{field}' must be an integer >= 0")

    if model < 0:
        location = _row_location(row_index)
        raise ManifestValidationError(f"{location}: field '{field}' must be an integer >= 0")
    return model


def _literal(
    value: Any,
    *,
    field: str,
    row_index: int | None = None,
    choices: set[str],
) -> str:
    text = _required_string(value, field=field, row_index=row_index)
    if text not in choices:
        location = _row_location(row_index)
        expected = ", ".join(sorted(choices))
        raise ManifestValidationError(f"{location}: field '{field}' must be one of: {expected}")
    return text


@dataclass
class BaseManifestRow(ABC):
    id: str

    field_order: ClassVar[tuple[str, ...]]

    @classmethod
    @abstractmethod
    def from_dict(cls, d: Mapping[str, Any], *, row_index: int | None = None) -> BaseManifestRow:
        raise NotImplementedError("Manifest from_dict not implemented.")

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError("Manifest to_dict not implemented.")


@dataclass
class StructureManifestRow(BaseManifestRow):
    path: str
    chain: str | None = None
    model: int = 0

    field_order: ClassVar[tuple[str, ...]] = ("id", "path", "chain", "model")

    def __post_init__(self) -> None:
        self.id = _required_string(self.id, field="id")
        self.path = _required_text_path(self.path, field="path")
        self.chain = _optional_string(self.chain, field="chain")
        self.model = _model(self.model, field="model")

    @classmethod
    def from_dict(
        cls, d: Mapping[str, Any], *, row_index: int | None = None
    ) -> StructureManifestRow:
        sid = _raw_field(
            d,
            "id",
            aliases=("sample_id",),
            row_index=row_index,
            normalizer=_required_string,
        )
        path = _raw_field(
            d,
            "path",
            aliases=("input",),
            row_index=row_index,
            normalizer=_required_text_path,
        )
        chain = _raw_field(
            d,
            "chain",
            aliases=("chain_id",),
            row_index=row_index,
            normalizer=_optional_string,
        )
        model = _raw_field(
            d,
            "model",
            aliases=("model_index",),
            row_index=row_index,
            normalizer=_model,
        )

        if sid is None:
            location = _row_location(row_index)
            raise ManifestValidationError(f"{location}: field 'id' is required")
        if path is None:
            location = _row_location(row_index)
            raise ManifestValidationError(f"{location}: field 'path' is required")

        return cls(id=sid, path=path, chain=chain, model=0 if model is None else model)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "chain": self.chain,
            "model": self.model,
        }


@dataclass
class BatchResultRow(BaseManifestRow):
    representation_name: str
    representation_family: str
    artifact_dir: str | None
    status: str
    error_type: str | None = None
    error_message: str | None = None

    field_order: ClassVar[tuple[str, ...]] = (
        "id",
        "representation_name",
        "representation_family",
        "artifact_dir",
        "status",
        "error_type",
        "error_message",
    )

    def __post_init__(self) -> None:
        self.id = _required_string(self.id, field="id")
        self.representation_name = _required_string(
            self.representation_name, field="representation_name"
        )
        self.representation_family = _literal(
            self.representation_family,
            field="representation_family",
            choices={"explicit", "learned"},
        )
        self.artifact_dir = _optional_text_path(self.artifact_dir, field="artifact_dir")
        self.status = _literal(self.status, field="status", choices={"success", "failed"})
        self.error_type = _optional_string(self.error_type, field="error_type")
        self.error_message = _optional_string(self.error_message, field="error_message")

        if self.status == "success":
            if self.artifact_dir is None:
                raise ManifestValidationError(
                    "manifest row: field 'artifact_dir' must be non-empty for success"
                )
            if self.error_type is not None or self.error_message is not None:
                raise ManifestValidationError(
                    "manifest row: success rows must not include error metadata"
                )
        else:
            if self.error_type is None:
                raise ManifestValidationError(
                    "manifest row: field 'error_type' must be non-empty for failed"
                )
            if self.error_message is None:
                raise ManifestValidationError(
                    "manifest row: field 'error_message' must be non-empty for failed"
                )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], *, row_index: int | None = None) -> BatchResultRow:
        try:
            return cls(
                id=_required_string(d.get("id"), field="id", row_index=row_index),
                representation_name=_required_string(
                    d.get("representation_name"),
                    field="representation_name",
                    row_index=row_index,
                ),
                representation_family=_literal(
                    d.get("representation_family"),
                    field="representation_family",
                    row_index=row_index,
                    choices={"explicit", "learned"},
                ),
                artifact_dir=_optional_text_path(
                    d.get("artifact_dir"), field="artifact_dir", row_index=row_index
                ),
                status=_literal(
                    d.get("status"),
                    field="status",
                    row_index=row_index,
                    choices={"success", "failed"},
                ),
                error_type=_optional_string(
                    d.get("error_type"), field="error_type", row_index=row_index
                ),
                error_message=_optional_string(
                    d.get("error_message"), field="error_message", row_index=row_index
                ),
            )
        except ManifestValidationError as error:
            message = str(error)
            if message.startswith("manifest row:") and row_index is not None:
                message = message.replace("manifest row:", _row_location(row_index) + ":", 1)
                raise ManifestValidationError(message) from error
            raise

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "representation_name": self.representation_name,
            "representation_family": self.representation_family,
            "artifact_dir": self.artifact_dir,
            "status": self.status,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


@dataclass
class ArtifactManifestRow(BaseManifestRow):
    """Legacy internal artifact manifest row retained until M7.3 migration."""

    artifact_dir: str

    field_order: ClassVar[tuple[str, ...]] = ("id", "artifact_dir")

    def __post_init__(self) -> None:
        self.id = _required_string(self.id, field="id")
        self.artifact_dir = _required_text_path(self.artifact_dir, field="artifact_dir")

    @classmethod
    def from_dict(
        cls, d: Mapping[str, Any], *, row_index: int | None = None
    ) -> ArtifactManifestRow:
        sid = _raw_field(
            d,
            "id",
            aliases=("sample_id",),
            row_index=row_index,
            normalizer=_required_string,
        )
        artifact_dir = _raw_field(
            d,
            "artifact_dir",
            aliases=("path", "input"),
            row_index=row_index,
            normalizer=_required_text_path,
        )
        if sid is None:
            location = _row_location(row_index)
            raise ManifestValidationError(f"{location}: field 'id' is required")
        if artifact_dir is None:
            location = _row_location(row_index)
            raise ManifestValidationError(f"{location}: field 'artifact_dir' is required")
        return cls(id=sid, artifact_dir=artifact_dir)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "artifact_dir": self.artifact_dir}


T = TypeVar("T", bound=BaseManifestRow)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ManifestValidationError(
                    f"line {line_number}: JSONL manifest entries must be objects"
                )
            rows.append(obj)
    return rows


def _read_json(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ManifestValidationError("JSON manifest must be a list of objects")
    for index, item in enumerate(obj, start=1):
        if not isinstance(item, dict):
            raise ManifestValidationError(f"row {index}: JSON manifest entries must be objects")
    return obj


def read_manifest(path: Path, row_cls: type[T]) -> list[T]:
    """
    Read a canonical manifest.

    Supported containers are CSV, JSONL, and JSON list-of-objects. Structure manifests
    use canonical fields ``id,path,chain,model``. Batch result manifests use canonical
    fields ``id,representation_name,representation_family,artifact_dir,status,error_type,
    error_message``.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        raw_rows = _read_csv(path)
    elif suffix == ".jsonl":
        raw_rows = _read_jsonl(path)
    elif suffix == ".json":
        raw_rows = _read_json(path)
    else:
        raise ManifestValidationError(
            f"Unsupported manifest format: {path.suffix} (use .csv, .jsonl, or .json)"
        )

    if not raw_rows:
        raise ManifestValidationError(f"Empty manifest: {path}")

    rows = [row_cls.from_dict(raw, row_index=index) for index, raw in enumerate(raw_rows, 1)]
    seen: dict[str, int] = {}
    for index, row in enumerate(rows, 1):
        if row.id in seen:
            raise ManifestValidationError(
                f"row {index}: duplicate id '{row.id}' first seen at row {seen[row.id]}"
            )
        seen[row.id] = index
    return rows


def _csv_row(row: BaseManifestRow) -> dict[str, Any]:
    serialized = row.to_dict()
    return {
        field: "" if serialized[field] is None else serialized[field] for field in row.field_order
    }


def write_manifest(path: Path, rows: list[BaseManifestRow]) -> None:
    """Write manifest rows with deterministic canonical fields for CSV, JSONL, or JSON."""
    path = Path(path)
    if not rows:
        raise ManifestValidationError("Cannot write an empty manifest")

    row_cls = type(rows[0])
    if any(type(row) is not row_cls for row in rows):
        raise ManifestValidationError("Cannot mix manifest row types")

    suffix = path.suffix.lower()
    field_order = rows[0].field_order

    if suffix == ".csv":
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=field_order)
            writer.writeheader()
            writer.writerows(_csv_row(row) for row in rows)
        return

    if suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row.to_dict(), allow_nan=False))
                handle.write("\n")
        return

    if suffix == ".json":
        payload = [row.to_dict() for row in rows]
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return

    raise ManifestValidationError(
        f"Unsupported manifest format: {path.suffix} (use .csv, .jsonl, or .json)"
    )


__all__ = [
    "ArtifactManifestRow",
    "BaseManifestRow",
    "BatchResultRow",
    "ManifestValidationError",
    "StructureManifestRow",
    "read_manifest",
    "write_manifest",
]
