from __future__ import annotations

import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from eris.core.artifact_schema import ArtifactValidationError, validate_schema


def _validate_artifact(
    meta: dict[str, Any],
    data: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    return validate_schema(meta, data, strict=True)


def _write_staged_npz(path: Path, data: Mapping[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **data)
        handle.flush()
        os.fsync(handle.fileno())


def _write_staged_json(path: Path, serialized_meta: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(serialized_meta)
        handle.flush()
        os.fsync(handle.fileno())


def _temporary_path(outdir: Path, *, prefix: str, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(dir=outdir, prefix=prefix, suffix=suffix)
    os.close(descriptor)
    return Path(name)


def _save_staged(
    outdir: Path,
    *,
    data: Mapping[str, np.ndarray],
    serialized_meta: str,
) -> None:
    data_tmp: Path | None = None
    meta_tmp: Path | None = None
    try:
        data_tmp = _temporary_path(outdir, prefix=".eris-data-", suffix=".npz")
        meta_tmp = _temporary_path(outdir, prefix=".eris-meta-", suffix=".json")
        _write_staged_npz(data_tmp, data)
        _write_staged_json(meta_tmp, serialized_meta)
        os.replace(data_tmp, outdir / "data.npz")
        os.replace(meta_tmp, outdir / "meta.json")
    finally:
        if data_tmp is not None:
            data_tmp.unlink(missing_ok=True)
        if meta_tmp is not None:
            meta_tmp.unlink(missing_ok=True)


def _read_artifact_metadata(meta_path: Path) -> dict[str, Any]:
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ArtifactValidationError(f"Invalid artifact metadata JSON: {meta_path}") from error
    if not isinstance(meta, dict):
        raise ArtifactValidationError("Artifact metadata must be a JSON object.")
    return meta


def _output_names(meta: dict[str, Any]) -> list[str]:
    outputs = meta.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ArtifactValidationError("Artifact meta.outputs must be a non-empty list.")

    names: list[str] = []
    for index, output in enumerate(outputs):
        if not isinstance(output, dict):
            raise ArtifactValidationError(f"Artifact meta.outputs[{index}] must be an object.")
        name = output.get("name")
        if not isinstance(name, str) or not name:
            raise ArtifactValidationError(
                f"Artifact meta.outputs[{index}].name must be a non-empty string."
            )
        names.append(name)

    if len(names) != len(set(names)):
        raise ArtifactValidationError("Artifact meta.outputs names must be unique.")
    return names


def _load_npz_data(data_path: Path, output_names: list[str]) -> dict[str, np.ndarray]:
    try:
        loaded = np.load(data_path, allow_pickle=False)
        if not isinstance(loaded, np.lib.npyio.NpzFile):
            raise ArtifactValidationError(f"Artifact payload is not an NPZ archive: {data_path}")
        with loaded as npz:
            member_names = list(npz.files)
            if len(member_names) != len(set(member_names)):
                raise ArtifactValidationError("Artifact data.npz contains duplicate member names.")
            if set(member_names) != set(output_names):
                raise ArtifactValidationError(
                    "Artifact data.npz members must exactly match meta.outputs names."
                )
            return {name: np.array(npz[name], copy=True) for name in output_names}
    except ArtifactValidationError:
        raise
    except (EOFError, KeyError, ValueError, zipfile.BadZipFile) as error:
        raise ArtifactValidationError(f"Invalid or unsafe artifact NPZ: {data_path}") from error


@dataclass
class StructureArtifact:
    """
    Canonical artifact with enforced schema contract.

    Always writes:
      - data.npz
      - meta.json (schema-validated)
    """

    data: Dict[str, np.ndarray]
    meta: Dict[str, Any]

    def finalize(self) -> None:
        canonical_meta = _validate_artifact(self.meta, self.data)
        if canonical_meta != self.meta:
            self.meta = canonical_meta

    def validate(self, strict=False):
        # Validation is always strict; `strict` is kept for existing callers.
        _validate_artifact(self.meta, self.data)

    def save(self, outdir: Path) -> None:
        self.finalize()
        output_names = [output["name"] for output in self.meta["outputs"]]
        serialized_data = {name: self.data[name] for name in output_names}
        serialized_meta = json.dumps(
            self.meta,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )

        outdir.mkdir(parents=True, exist_ok=True)
        _save_staged(
            outdir,
            data=serialized_data,
            serialized_meta=serialized_meta,
        )

    @staticmethod
    def load(indir: Path, *, strict: bool = False) -> "StructureArtifact":
        # Loading always validates strictly; `strict` is kept for existing callers.
        if not indir.exists():
            raise ArtifactValidationError(f"Artifact directory does not exist: {indir}")
        if not indir.is_dir():
            raise ArtifactValidationError(f"Artifact path is not a directory: {indir}")

        meta_path = indir / "meta.json"
        data_path = indir / "data.npz"

        if not meta_path.exists() or not data_path.exists():
            raise ArtifactValidationError(
                f"Missing artifact files in {indir} (need meta.json + data.npz)"
            )

        meta = _read_artifact_metadata(meta_path)
        data = _load_npz_data(data_path, _output_names(meta))
        canonical_meta = _validate_artifact(meta, data)
        return StructureArtifact(data=data, meta=canonical_meta)
