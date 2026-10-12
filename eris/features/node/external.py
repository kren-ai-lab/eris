from __future__ import annotations

import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from zipfile import BadZipFile

import numpy as np

from eris.core.residue_identity import format_residue_id

ALIGNMENT_POLICY = "strict_residue_id"
_HASH_DOMAIN = b"eris.external-residue-features.v2\x00"
_REQUIRED_FIELDS = {"values", "residue_ids", "source_name", "chain_ids", "model_index"}
_OPTIONAL_FIELDS = {"feature_keys", "source_version", "source_model"}
_ALLOWED_FIELDS = _REQUIRED_FIELDS | _OPTIONAL_FIELDS
_DESCRIPTOR_FIELDS = {
    "path",
    "content_sha256",
    "source_name",
    "source_version",
    "source_model",
    "chain_ids",
    "model_index",
    "feature_dim",
    "feature_keys",
    "alignment",
}
_KEY_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_PREVIEW_LIMIT = 5


class ExternalFeatureError(ValueError):
    """Raised when an external residue feature payload violates its contract."""


@dataclass(frozen=True)
class ExternalResiduePayload:
    path: str
    values: np.ndarray
    residue_ids: tuple[str, ...]
    source_name: str
    source_version: str | None
    source_model: str | None
    chain_ids: tuple[str, ...]
    model_index: int
    feature_keys: tuple[str, ...]
    content_sha256: str

    @property
    def feature_dim(self) -> int:
        return int(self.values.shape[1])

    def descriptor(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "content_sha256": self.content_sha256,
            "source_name": self.source_name,
            "source_version": self.source_version,
            "source_model": self.source_model,
            "chain_ids": list(self.chain_ids),
            "model_index": self.model_index,
            "feature_dim": self.feature_dim,
            "feature_keys": list(self.feature_keys),
            "alignment": ALIGNMENT_POLICY,
        }


def _resolved_npz_path(path: Any) -> Path:
    if not isinstance(path, (str, Path)):
        raise TypeError("external parameter 'path' must be a filesystem path string")
    raw_path = Path(path).expanduser()
    if raw_path.suffix.lower() != ".npz":
        raise ExternalFeatureError("external feature path must use the .npz format")
    try:
        resolved = raw_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ExternalFeatureError(f"External feature file does not exist: {raw_path}") from exc
    if not resolved.is_file():
        raise ExternalFeatureError(f"External feature path is not a file: {resolved}")
    return resolved


def _load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    archive: Any = None
    try:
        archive = np.load(path, allow_pickle=False)
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise ExternalFeatureError(f"External feature file is not an .npz archive: {path}")

        field_names = list(archive.files)
        duplicate_fields = sorted({name for name in field_names if field_names.count(name) > 1})
        if duplicate_fields:
            raise ExternalFeatureError(
                f"External feature archive contains duplicate fields: {duplicate_fields}"
            )

        fields = set(field_names)
        missing = sorted(_REQUIRED_FIELDS - fields)
        if missing:
            raise ExternalFeatureError(f"External feature archive is missing fields: {missing}")
        unknown = sorted(fields - _ALLOWED_FIELDS)
        if unknown:
            raise ExternalFeatureError(f"External feature archive has unknown fields: {unknown}")

        try:
            return {name: np.asarray(archive[name]) for name in field_names}
        except ValueError as exc:
            raise ExternalFeatureError(
                "External feature archive contains an unsafe object array or malformed field"
            ) from exc
    except ExternalFeatureError:
        raise
    except (EOFError, OSError, ValueError, BadZipFile) as exc:
        raise ExternalFeatureError(f"Could not read external feature archive: {path}") from exc
    finally:
        if isinstance(archive, np.lib.npyio.NpzFile):
            archive.close()


def _decode_text(value: np.generic, *, field: str) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        try:
            return bytes(value).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ExternalFeatureError(f"External field '{field}' is not valid UTF-8") from exc
    return str(value)


def _text_scalar(
    array: np.ndarray,
    *,
    field: str,
    allow_empty: bool = False,
) -> str:
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ExternalFeatureError(
            f"External field '{field}' must be a scalar Unicode/string value"
        )
    value = _decode_text(array[()], field=field)
    if not allow_empty and not value:
        raise ExternalFeatureError(f"External field '{field}' must not be empty")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ExternalFeatureError(f"External field '{field}' must not contain control characters")
    return value


def _optional_text_scalar(arrays: Mapping[str, np.ndarray], field: str) -> str | None:
    if field not in arrays:
        return None
    return _text_scalar(arrays[field], field=field)


def _text_vector(array: np.ndarray, *, field: str) -> list[str]:
    if array.ndim != 1 or array.dtype.kind not in {"U", "S"}:
        raise ExternalFeatureError(
            f"External field '{field}' must be a one-dimensional Unicode/string array"
        )
    return [_decode_text(item, field=field) for item in array]


def _integer_scalar(array: np.ndarray, *, field: str) -> int:
    if array.shape != () or array.dtype.kind not in {"i", "u"}:
        raise ExternalFeatureError(f"External field '{field}' must be a scalar integer")
    return int(array[()])


def _validate_key_component(value: str, *, field: str) -> str:
    if _KEY_COMPONENT.fullmatch(value) is None:
        raise ExternalFeatureError(
            f"External field '{field}' value {value!r} must use only letters, digits, "
            "underscores, or hyphens and must start with a letter or digit"
        )
    return value


def _canonical_residue_parts(residue_id: str) -> tuple[str, int, str, str]:
    if not residue_id or residue_id != residue_id.strip():
        raise ExternalFeatureError(f"Malformed canonical residue ID: {residue_id!r}")
    parts = residue_id.split(":")
    if len(parts) != 4:
        raise ExternalFeatureError(f"Malformed canonical residue ID: {residue_id!r}")
    chain_id, number_text, insertion_code, residue_name = parts
    if not number_text or not residue_name or any(part != part.strip() for part in parts):
        raise ExternalFeatureError(f"Malformed canonical residue ID: {residue_id!r}")
    try:
        residue_number = int(number_text)
    except ValueError as exc:
        raise ExternalFeatureError(f"Malformed canonical residue ID: {residue_id!r}") from exc
    canonical = format_residue_id(
        chain_id=chain_id,
        residue_number=residue_number,
        insertion_code=insertion_code,
        residue_name=residue_name,
    )
    if canonical != residue_id:
        raise ExternalFeatureError(f"Malformed canonical residue ID: {residue_id!r}")
    return chain_id, residue_number, insertion_code, residue_name


def _duplicates(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _preview(values: Sequence[str]) -> str:
    ordered = sorted(values)
    preview = ordered[:_PREVIEW_LIMIT]
    suffix = "" if len(ordered) <= _PREVIEW_LIMIT else ", ..."
    return f"[{', '.join(repr(value) for value in preview)}{suffix}]"


def _canonical_values(array: np.ndarray) -> np.ndarray:
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ExternalFeatureError(
            f"External field 'values' must have shape (N, D) with N,D >= 1; got {array.shape}"
        )
    if array.dtype.kind not in {"i", "u", "f"}:
        raise ExternalFeatureError(
            "External field 'values' must be a real numeric matrix, excluding boolean and complex"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.array(array, dtype=np.float32, order="C", copy=True)
    overflow = np.isfinite(array) & ~np.isfinite(values)
    if np.any(overflow):
        raise ExternalFeatureError("External field 'values' contains values outside float32 range")
    values[np.isnan(values)] = np.float32(np.nan)
    values[values == 0.0] = np.float32(0.0)
    return np.ascontiguousarray(values, dtype=np.float32)


def _content_sha256(
    *,
    values: np.ndarray,
    residue_ids: Sequence[str],
    feature_keys: Sequence[str],
    source_name: str,
    source_version: str | None,
    source_model: str | None,
    chain_ids: Sequence[str],
    model_index: int,
) -> str:
    header = {
        "alignment": ALIGNMENT_POLICY,
        "chain_ids": list(chain_ids),
        "feature_keys": list(feature_keys),
        "model_index": model_index,
        "residue_ids": list(residue_ids),
        "source_model": source_model,
        "source_name": source_name,
        "source_version": source_version,
        "values_dtype": "float32-le",
        "values_shape": list(values.shape),
    }
    header_bytes = json.dumps(
        header,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    value_bytes = np.ascontiguousarray(values, dtype="<f4").tobytes(order="C")

    digest = hashlib.sha256()
    digest.update(_HASH_DOMAIN)
    digest.update(struct.pack(">Q", len(header_bytes)))
    digest.update(header_bytes)
    digest.update(struct.pack(">Q", len(value_bytes)))
    digest.update(value_bytes)
    return digest.hexdigest()


def load_external_payload(path: str | Path) -> ExternalResiduePayload:
    resolved_path = _resolved_npz_path(path)
    arrays = _load_npz_arrays(resolved_path)

    values = _canonical_values(arrays["values"])
    residue_ids = _text_vector(arrays["residue_ids"], field="residue_ids")
    if len(residue_ids) != values.shape[0]:
        raise ExternalFeatureError(
            "External residue row mismatch: "
            f"values has {values.shape[0]} rows but residue_ids has {len(residue_ids)}"
        )
    duplicate_ids = _duplicates(residue_ids)
    if duplicate_ids:
        raise ExternalFeatureError(
            f"External residue_ids contains {len(duplicate_ids)} duplicate IDs; "
            f"preview={_preview(duplicate_ids)}"
        )
    residue_chains = []
    for residue_id in residue_ids:
        residue_chains.append(_canonical_residue_parts(residue_id)[0])

    source_name = _validate_key_component(
        _text_scalar(arrays["source_name"], field="source_name"),
        field="source_name",
    )
    chain_ids = _text_vector(arrays["chain_ids"], field="chain_ids")
    if not chain_ids:
        raise ExternalFeatureError("External field 'chain_ids' must not be empty")
    if any(chain != chain.strip() or ":" in chain for chain in chain_ids):
        raise ExternalFeatureError(
            "External field 'chain_ids' entries must not contain surrounding whitespace or ':'"
        )
    duplicate_chains = _duplicates(chain_ids)
    if duplicate_chains:
        raise ExternalFeatureError(
            f"External field 'chain_ids' contains duplicates: {_preview(duplicate_chains)}"
        )
    observed_chains = set(residue_chains)
    declared_chains = set(chain_ids)
    undeclared_chains = observed_chains - declared_chains
    if undeclared_chains:
        raise ExternalFeatureError(
            "External residue_ids contain chains outside payload chain_ids: "
            f"declared={_preview(chain_ids)}, undeclared={_preview(undeclared_chains)}"
        )

    model_index = _integer_scalar(arrays["model_index"], field="model_index")
    if model_index < 0:
        raise ExternalFeatureError("External field 'model_index' must be >= 0")
    source_version = _optional_text_scalar(arrays, "source_version")
    source_model = _optional_text_scalar(arrays, "source_model")

    if "feature_keys" in arrays:
        feature_keys = _text_vector(arrays["feature_keys"], field="feature_keys")
        if len(feature_keys) != values.shape[1]:
            raise ExternalFeatureError(
                "External feature-key mismatch: "
                f"values has {values.shape[1]} columns but feature_keys has {len(feature_keys)}"
            )
    else:
        feature_keys = [str(index) for index in range(values.shape[1])]
    for feature_key in feature_keys:
        _validate_key_component(feature_key, field="feature_keys")
    duplicate_keys = _duplicates(feature_keys)
    if duplicate_keys:
        raise ExternalFeatureError(
            f"External feature_keys contains {len(duplicate_keys)} duplicates; "
            f"preview={_preview(duplicate_keys)}"
        )

    content_sha256 = _content_sha256(
        values=values,
        residue_ids=residue_ids,
        feature_keys=feature_keys,
        source_name=source_name,
        source_version=source_version,
        source_model=source_model,
        chain_ids=chain_ids,
        model_index=model_index,
    )
    return ExternalResiduePayload(
        path=str(resolved_path),
        values=values,
        residue_ids=tuple(residue_ids),
        source_name=source_name,
        source_version=source_version,
        source_model=source_model,
        chain_ids=tuple(chain_ids),
        model_index=model_index,
        feature_keys=tuple(feature_keys),
        content_sha256=content_sha256,
    )


def normalize_external_params(params: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(params) - _DESCRIPTOR_FIELDS)
    if unknown:
        raise ExternalFeatureError(f"Unknown external parameters: {unknown}")
    if "path" not in params:
        raise ExternalFeatureError("external parameter 'path' is required")

    payload = load_external_payload(params["path"])
    descriptor = payload.descriptor()
    for field in sorted(_DESCRIPTOR_FIELDS - {"path"}):
        if field in params and params[field] != descriptor[field]:
            raise ExternalFeatureError(
                f"external parameter '{field}' does not match the resolved payload"
            )
    return descriptor


def project_external_fingerprint_params(params: dict[str, Any]) -> dict[str, Any]:
    """Exclude only invocation location from canonical scientific identity."""
    if set(params) != _DESCRIPTOR_FIELDS:
        missing = sorted(_DESCRIPTOR_FIELDS - set(params))
        extra = sorted(set(params) - _DESCRIPTOR_FIELDS)
        raise ExternalFeatureError(
            f"Canonical external parameters are incomplete: missing={missing}, extra={extra}"
        )
    projected = dict(params)
    del projected["path"]
    return projected


def verify_external_params(params: Mapping[str, Any]) -> ExternalResiduePayload:
    if set(params) != _DESCRIPTOR_FIELDS:
        missing = sorted(_DESCRIPTOR_FIELDS - set(params))
        extra = sorted(set(params) - _DESCRIPTOR_FIELDS)
        raise ExternalFeatureError(
            f"Canonical external parameters are incomplete: missing={missing}, extra={extra}"
        )
    payload = load_external_payload(params["path"])
    observed = payload.descriptor()
    expected = dict(params)
    expected["feature_keys"] = list(expected["feature_keys"])
    if observed != expected:
        raise ExternalFeatureError(
            "External residue payload changed after configuration was resolved: "
            f"expected content_sha256={params['content_sha256']}, "
            f"observed content_sha256={payload.content_sha256}"
        )
    return payload


def strict_align_external_payload(
    payload: ExternalResiduePayload,
    target_residue_ids: np.ndarray,
    *,
    chain_ids: Sequence[str],
    model_index: int,
) -> np.ndarray:
    selected_chain_ids = tuple(chain_ids)
    if payload.chain_ids != selected_chain_ids:
        raise ExternalFeatureError(
            f"External chain-set mismatch: expected selected chains {selected_chain_ids!r}, "
            f"observed payload chains {payload.chain_ids!r}"
        )
    if payload.model_index != model_index:
        raise ExternalFeatureError(
            f"External model-index mismatch: expected {model_index}, observed {payload.model_index}"
        )

    target_array = np.asarray(target_residue_ids)
    if target_array.ndim != 1 or target_array.dtype.kind not in {"U", "S"}:
        raise ExternalFeatureError("Eris residue_ids must be a one-dimensional string array")
    target_ids = [_decode_text(item, field="Eris residue_ids") for item in target_array]
    target_duplicates = _duplicates(target_ids)
    if target_duplicates:
        raise ExternalFeatureError(
            f"Eris residue_ids contains duplicate IDs; preview={_preview(target_duplicates)}"
        )
    for residue_id in target_ids:
        _canonical_residue_parts(residue_id)

    external_set = set(payload.residue_ids)
    target_set = set(target_ids)
    missing = sorted(target_set - external_set)
    extra = sorted(external_set - target_set)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={len(missing)} preview={_preview(missing)}")
        if extra:
            details.append(f"extra={len(extra)} preview={_preview(extra)}")
        raise ExternalFeatureError(
            "External strict_residue_id alignment failed: " + "; ".join(details)
        )

    row_by_id = {residue_id: row for row, residue_id in enumerate(payload.residue_ids)}
    aligned = payload.values[[row_by_id[residue_id] for residue_id in target_ids]]
    return np.ascontiguousarray(aligned, dtype=np.float32)


def external_feature_values(
    data: dict[str, Any],
    ctx: dict[str, Any],
    params: dict[str, Any],
) -> np.ndarray:
    payload = verify_external_params(params)
    return strict_align_external_payload(
        payload,
        data["residue_ids"],
        chain_ids=ctx["chain_ids"],
        model_index=ctx["model_index"],
    )


def external_feature_keys(
    data: dict[str, Any],
    ctx: dict[str, Any],
    params: dict[str, Any],
) -> list[str]:
    return [
        f"external.{params['source_name']}.{feature_key}" for feature_key in params["feature_keys"]
    ]


__all__ = [
    "ExternalFeatureError",
    "ExternalResiduePayload",
    "external_feature_keys",
    "external_feature_values",
    "load_external_payload",
    "normalize_external_params",
    "project_external_fingerprint_params",
    "strict_align_external_payload",
    "verify_external_params",
]
