from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from platformdirs import user_cache_dir

from eris.core.artifact_schema import (
    _IMPLEMENTATION_PATH,
    ARTIFACT_SCHEMA_VERSION,
    ArtifactValidationError,
    _require_non_empty_string,
    _require_sha256,
    _validated_chain_ids,
    canonical_json,
    normalize_metadata,
)
from eris.core.hashing import sha256_text

if TYPE_CHECKING:
    from eris.core.artifacts import StructureArtifact


def _validate_cache_key(key: str) -> str:
    try:
        return _require_sha256(key, "cache key")
    except ArtifactValidationError as error:
        raise ValueError(str(error)) from error


def parser_hint_from_source(source: str | Path) -> str:
    """Return a path-independent parser selector from the source suffix."""
    path = Path(source)
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if not suffixes:
        return "<no-suffix>"
    if suffixes[-1] == ".gz" and len(suffixes) > 1:
        return f"{suffixes[-2]}.gz"
    return suffixes[-1]


def parser_hint_matches_format(parser_hint: str, input_format: str) -> bool:
    """Check known Gemmi suffix selectors against a confirmed input format."""
    expected_formats = {
        ".pdb": "pdb",
        ".ent": "pdb",
        ".pdb.gz": "pdb",
        ".ent.gz": "pdb",
        ".cif": "mmcif",
        ".mmcif": "mmcif",
        ".cif.gz": "mmcif",
        ".mmcif.gz": "mmcif",
    }
    expected = expected_formats.get(parser_hint)
    return expected is None or input_format == expected


def compute_cache_key(
    *,
    representation_name: str,
    encoder_implementation: str,
    encoder_version: str,
    config_fingerprint: str,
    source_sha256: str,
    parser_hint: str,
    model_index: int,
    chain_ids: Sequence[str],
    model: Mapping[str, Any] | None = None,
    execution: Mapping[str, Any] | None = None,
) -> str:
    try:
        _require_non_empty_string(representation_name, "representation_name")
        implementation = _require_non_empty_string(
            encoder_implementation,
            "encoder_implementation",
        )
        if _IMPLEMENTATION_PATH.fullmatch(implementation) is None:
            raise ValueError("encoder_implementation must be a valid dotted implementation path")
        _require_non_empty_string(encoder_version, "encoder_version")
        _require_sha256(config_fingerprint, "config_fingerprint")
        _require_sha256(source_sha256, "source_sha256")
        hint = _require_non_empty_string(parser_hint, "parser_hint")
        if type(model_index) is not int or model_index < 0:
            raise ValueError("model_index must be a plain integer >= 0")
        normalized_chain_ids = _validated_chain_ids(chain_ids, "chain_ids")
    except ArtifactValidationError as error:
        raise ValueError(str(error)) from error
    if model is not None and not isinstance(model, Mapping):
        raise ValueError("model must be a mapping or None")
    if execution is not None and not isinstance(execution, Mapping):
        raise ValueError("execution must be a mapping or None")

    payload: dict[str, Any] = {
        "cache_contract": 1,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "representation_name": representation_name,
        "encoder": {
            "implementation": implementation,
            "version": encoder_version,
        },
        "config_fingerprint": config_fingerprint,
        "source_sha256": source_sha256,
        "parser_hint": hint,
        "input_selection": {
            "model_index": model_index,
            "chain_ids": normalized_chain_ids,
        },
    }
    if model is not None:
        payload["model"] = normalize_metadata(model, _path="model")
    if execution is not None:
        payload["execution"] = normalize_metadata(execution, _path="execution")
    return sha256_text(canonical_json(payload))


def default_cache_dir() -> Path:
    p = Path(user_cache_dir("eris"))
    p.mkdir(parents=True, exist_ok=True)
    return p


class DiskCache:
    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = cache_dir or default_cache_dir()

    def _entry_dir(self, key: str) -> Path:
        return self.cache_dir / "artifacts" / _validate_cache_key(key)

    def get(self, key: str) -> StructureArtifact | None:
        from eris.core.artifacts import ArtifactValidationError, StructureArtifact

        entry_dir = self._entry_dir(key)
        if not entry_dir.is_dir():
            return None
        try:
            return StructureArtifact.load(entry_dir)
        except (ArtifactValidationError, OSError):
            return None

    def set(self, key: str, artifact: StructureArtifact) -> None:
        artifact.save(self._entry_dir(key))
