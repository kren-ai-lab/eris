from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import json
import numpy as np
from platformdirs import user_cache_dir

from eris.core.hashing import sha256_text

if TYPE_CHECKING:
    from eris.core.artifacts import StructureArtifact


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IMPLEMENTATION_PATTERN = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$"
)


def _require_non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 hex string")
    return value


def _validate_cache_key_v2(key: str) -> str:
    return _require_sha256(key, "v2 cache key")


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


def compute_cache_key_v2(
    *,
    representation_name: str,
    encoder_implementation: str,
    encoder_version: str,
    config_fingerprint: str,
    source_sha256: str,
    parser_hint: str,
    model_index: int,
    chain_id: str | None,
    model: Mapping[str, Any] | None = None,
) -> str:
    from eris.core.artifact_schema_v2 import canonical_json, normalize_metadata

    _require_non_empty_string(representation_name, "representation_name")
    implementation = _require_non_empty_string(
        encoder_implementation,
        "encoder_implementation",
    )
    if _IMPLEMENTATION_PATTERN.fullmatch(implementation) is None:
        raise ValueError("encoder_implementation must be a valid dotted implementation path")
    _require_non_empty_string(encoder_version, "encoder_version")
    _require_sha256(config_fingerprint, "config_fingerprint")
    _require_sha256(source_sha256, "source_sha256")
    hint = _require_non_empty_string(parser_hint, "parser_hint")
    if type(model_index) is not int or model_index < 0:
        raise ValueError("model_index must be a plain integer >= 0")
    if chain_id is not None and not isinstance(chain_id, str):
        raise ValueError("chain_id must be a string or None")
    if model is not None and not isinstance(model, Mapping):
        raise ValueError("model must be a mapping or None")

    payload: dict[str, Any] = {
        "cache_contract": 2,
        "artifact_schema_version": 2,
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
            "chain_id": chain_id,
        },
    }
    if model is not None:
        payload["model"] = normalize_metadata(model, _path="model")
    return sha256_text(canonical_json(payload))


def default_cache_dir() -> Path:
    p = Path(user_cache_dir("eris"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_cache_key(*parts: str) -> str:
    return sha256_text("|".join(parts))


class DiskCache:
    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = cache_dir or default_cache_dir()

    def _paths(self, key: str) -> Tuple[Path, Path]:
        base = self.cache_dir / key
        return base.with_suffix(".npz"), base.with_suffix(".json")

    def _v2_entry_dir(self, key: str) -> Path:
        return self.cache_dir / "v2" / _validate_cache_key_v2(key)

    def get(self, key: str) -> Optional[Tuple[Dict[str, np.ndarray], Dict[str, Any]]]:
        npz_path, meta_path = self._paths(key)
        if not npz_path.exists() or not meta_path.exists():
            return None
        npz = np.load(npz_path)
        data = {k: npz[k] for k in npz.files}
        meta = json.loads(meta_path.read_text())
        return data, meta

    def set(self, key: str, data: Dict[str, np.ndarray], meta: Dict[str, Any]) -> None:
        npz_path, meta_path = self._paths(key)
        np.savez_compressed(npz_path, **data)
        meta_path.write_text(json.dumps(meta, indent=2))

    def get_v2(self, key: str) -> StructureArtifact | None:
        from eris.core.artifacts import ArtifactValidationError, StructureArtifact

        entry_dir = self._v2_entry_dir(key)
        if not entry_dir.is_dir():
            return None
        try:
            artifact = StructureArtifact.load(entry_dir)
        except (ArtifactValidationError, OSError):
            return None
        if type(artifact.meta.get("schema_version")) is not int:
            return None
        if artifact.meta["schema_version"] != 2:
            return None
        return artifact

    def set_v2(self, key: str, artifact: StructureArtifact) -> None:
        from eris.core.artifacts import ArtifactValidationError

        entry_dir = self._v2_entry_dir(key)
        version = artifact.meta.get("schema_version")
        if type(version) is not int or version != 2:
            raise ArtifactValidationError("V2 cache entries require schema_version=2.")
        artifact.validate(strict=True)
        artifact.save(entry_dir)
