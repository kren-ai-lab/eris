from __future__ import annotations

import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from eris.exceptions import ErisError


ARTIFACT_SCHEMA_VERSION = 1

ALLOWED_TYPES = {"graph", "matrix", "cloud", "grid", "mesh"}
ALLOWED_LEVELS = {"atom", "residue", "structure", "mixed"}


class ArtifactValidationError(ErisError):
    pass


_PROVISIONAL_V1_REQUIRED = {
    "encoder_name",
    "encoder_version",
    "encoder_fingerprint",
    "type",
    "level",
    "params",
    "input",
    "feature_keys",
}
_V2_IDENTITY_FIELDS = {
    "representation_name",
    "representation_family",
    "representation_type",
    "coordinate_source",
    "encoder",
    "model",
    "parameters",
    "provenance",
    "outputs",
}


def _detect_artifact_schema(meta: dict[str, Any]) -> str:
    if not isinstance(meta, dict):
        raise ArtifactValidationError("Artifact metadata must be a dict.")

    if "schema_version" not in meta:
        missing = sorted(_PROVISIONAL_V1_REQUIRED - meta.keys())
        mixed = sorted(_V2_IDENTITY_FIELDS & meta.keys())
        if not missing and not mixed:
            return "provisional_v1"
        details = []
        if missing:
            details.append(f"missing legacy fields: {missing}")
        if mixed:
            details.append(f"contains Schema v2 fields: {mixed}")
        raise ArtifactValidationError(
            "Versionless metadata is not a complete provisional Schema v1 artifact"
            f" ({'; '.join(details)})."
        )

    version = meta["schema_version"]
    if type(version) is not int:
        raise ArtifactValidationError("schema_version must be an exact plain Python int.")
    if version == 1:
        return "v1"
    if version == 2:
        return "v2"
    raise ArtifactValidationError(f"Unsupported schema_version={version}.")


def _validate_artifact_v2(
    meta: dict[str, Any],
    data: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    from eris.core.artifact_schema_v2 import (
        ArtifactSchemaV2ValidationError,
        validate_schema_v2,
    )

    try:
        return validate_schema_v2(meta, data, strict=True)
    except ArtifactSchemaV2ValidationError as error:
        raise ArtifactValidationError(str(error)) from error


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


def _v2_output_names(meta: dict[str, Any]) -> list[str]:
    outputs = meta.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ArtifactValidationError("Schema v2 meta.outputs must be a non-empty list.")

    names: list[str] = []
    for index, output in enumerate(outputs):
        if not isinstance(output, dict):
            raise ArtifactValidationError(
                f"Schema v2 meta.outputs[{index}] must be an object."
            )
        name = output.get("name")
        if not isinstance(name, str) or not name:
            raise ArtifactValidationError(
                f"Schema v2 meta.outputs[{index}].name must be a non-empty string."
            )
        names.append(name)

    if len(names) != len(set(names)):
        raise ArtifactValidationError("Schema v2 meta.outputs names must be unique.")
    return names


def _load_npz_data(
    data_path: Path,
    *,
    output_names: list[str] | None = None,
) -> dict[str, np.ndarray]:
    try:
        loaded = np.load(data_path, allow_pickle=False)
        if not isinstance(loaded, np.lib.npyio.NpzFile):
            raise ArtifactValidationError(f"Artifact payload is not an NPZ archive: {data_path}")
        with loaded as npz:
            member_names = list(npz.files)
            if output_names is not None:
                if len(member_names) != len(set(member_names)):
                    raise ArtifactValidationError(
                        "Schema v2 data.npz contains duplicate member names."
                    )
                if set(member_names) != set(output_names):
                    raise ArtifactValidationError(
                        "Schema v2 data.npz members must exactly match meta.outputs names."
                    )
                names = output_names
            else:
                names = member_names
            return {name: np.array(npz[name], copy=True) for name in names}
    except ArtifactValidationError:
        raise
    except (EOFError, KeyError, ValueError, zipfile.BadZipFile) as error:
        raise ArtifactValidationError(f"Invalid or unsafe artifact NPZ: {data_path}") from error


def _safe_shape(x: Any) -> Optional[List[int]]:
    if isinstance(x, np.ndarray):
        return list(x.shape)
    return None


def _safe_dtype(x: Any) -> Optional[str]:
    if isinstance(x, np.ndarray):
        return str(x.dtype)
    return None


def infer_shapes_dtypes(data: Mapping[str, np.ndarray]) -> Tuple[Dict[str, List[int]], Dict[str, str]]:
    shapes: Dict[str, List[int]] = {}
    dtypes: Dict[str, str] = {}
    for k, v in data.items():
        if not isinstance(v, np.ndarray):
            continue
        shapes[k] = list(v.shape)
        dtypes[k] = str(v.dtype)
    return shapes, dtypes


def infer_counts(data: Mapping[str, np.ndarray]) -> Dict[str, int]:
    counts: Dict[str, int] = {}

    # Graph
    if "edge_index" in data and isinstance(data["edge_index"], np.ndarray) and data["edge_index"].ndim == 2:
        counts["n_edges"] = int(data["edge_index"].shape[1])
    if "pos" in data and isinstance(data["pos"], np.ndarray) and data["pos"].ndim == 2:
        counts["n_nodes"] = int(data["pos"].shape[0])

    # Matrix
    if "x" in data and isinstance(data["x"], np.ndarray) and data["x"].ndim >= 2:
        counts.setdefault("n_nodes", int(data["x"].shape[0]))

    # Grid
    if "grid" in data and isinstance(data["grid"], np.ndarray):
        # common: (C,G,G,G) or (G,G,G) – we store both if possible
        g = data["grid"]
        if g.ndim == 4:
            counts["grid_c"] = int(g.shape[0])
            counts["grid_g"] = int(g.shape[-1])
        elif g.ndim == 3:
            counts["grid_g"] = int(g.shape[-1])

    # Cloud
    if "pos" in data and isinstance(data["pos"], np.ndarray) and data["pos"].ndim == 2:
        counts.setdefault("n_points", int(data["pos"].shape[0]))

    return counts


_FEATURE_KEY_LIST_FIELDS = (
    "node_attr_keys",
    "edge_attr_keys",
    "graph_attr_keys",
    "matrix_attr_keys",
    "grid_attr_keys",
    "mesh_attr_keys",
    "grid_channels",
)


def _validate_feature_keys(
    feature_keys: Dict[str, Any],
    data: Mapping[str, np.ndarray],
) -> None:
    for field in _FEATURE_KEY_LIST_FIELDS:
        if field not in feature_keys:
            continue
        keys = feature_keys[field]
        if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
            raise ArtifactValidationError(
                f"meta.feature_keys.{field} must be a list[str]"
            )

    grid_fields = {"grid_key", "grid_layout", "grid_channels"}
    if grid_fields.intersection(feature_keys):
        missing = sorted(grid_fields.difference(feature_keys))
        if missing:
            raise ArtifactValidationError(
                f"meta.feature_keys grid metadata is missing required fields: {missing}"
            )
        if feature_keys["grid_key"] != "grid":
            raise ArtifactValidationError(
                "meta.feature_keys.grid_key must be 'grid' for Schema v1 voxel artifacts"
            )
        if feature_keys["grid_layout"] != "CXYZ":
            raise ArtifactValidationError(
                "meta.feature_keys.grid_layout must be 'CXYZ' for Schema v1 voxel artifacts"
            )

        grid = data.get("grid")
        if not isinstance(grid, np.ndarray):
            raise ArtifactValidationError(
                "data.grid must exist as a numpy array when grid feature metadata is provided"
            )
        if grid.ndim != 4:
            raise ArtifactValidationError(
                f"data.grid must be 4D with CXYZ layout, got shape {grid.shape}"
            )
        if len(feature_keys["grid_channels"]) != grid.shape[0]:
            raise ArtifactValidationError(
                "meta.feature_keys.grid_channels has "
                f"{len(feature_keys['grid_channels'])} keys for data.grid with "
                f"{grid.shape[0]} channels"
            )

    for keys_field, data_field in (
        ("node_attr_keys", "node_attr"),
        ("edge_attr_keys", "edge_attr"),
    ):
        if keys_field not in feature_keys or data_field not in data:
            continue
        array = data[data_field]
        if not isinstance(array, np.ndarray):
            continue
        if array.ndim != 2:
            raise ArtifactValidationError(
                f"data.{data_field} must be 2D when {keys_field} are provided, "
                f"got shape {array.shape}"
            )
        if len(feature_keys[keys_field]) != array.shape[1]:
            raise ArtifactValidationError(
                f"meta.feature_keys.{keys_field} has {len(feature_keys[keys_field])} "
                f"keys for data.{data_field} with {array.shape[1]} columns"
            )

    if "graph_attr_keys" in feature_keys and "graph_attr" in data:
        graph_attr = data["graph_attr"]
        if isinstance(graph_attr, np.ndarray):
            if graph_attr.ndim != 1:
                raise ArtifactValidationError(
                    "data.graph_attr must be 1D when graph_attr_keys are provided, "
                    f"got shape {graph_attr.shape}"
                )
            if len(feature_keys["graph_attr_keys"]) != graph_attr.shape[0]:
                raise ArtifactValidationError(
                    "meta.feature_keys.graph_attr_keys has "
                    f"{len(feature_keys['graph_attr_keys'])} keys for data.graph_attr "
                    f"with length {graph_attr.shape[0]}"
                )


def infer_feature_keys(meta: Dict[str, Any], data: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    """
    Canonical place for feature key lists.
    - Preserve an explicit, non-empty feature_keys dict as authoritative.
    - Otherwise use the existing Schema v1 legacy fallback.
    """
    if "feature_keys" in meta:
        explicit = meta["feature_keys"]
        if not isinstance(explicit, dict):
            raise ArtifactValidationError("meta.feature_keys must be a dict")
        if explicit:
            return explicit

    out: Dict[str, List[str]] = {}

    for k in ("node_attr_keys", "edge_attr_keys", "graph_attr_keys", "matrix_attr_keys", "grid_attr_keys", "mesh_attr_keys"):
        if k in meta and isinstance(meta[k], list) and all(isinstance(x, str) for x in meta[k]):
            out[k] = list(meta[k])

    # allow storing keys arrays inside data.npz (strings)
    if "graph_attr_keys" not in out and "graph_attr_keys" in data:
        arr = data["graph_attr_keys"]
        if isinstance(arr, np.ndarray) and arr.dtype.kind in {"U", "S"}:
            out["graph_attr_keys"] = [str(x) for x in arr.tolist()]

    return out


def normalize_input_meta(inp: Any) -> Dict[str, Any]:
    """
    Ensure meta['input'] is always a dict with stable keys.
    Backward compatible with legacy string "source=...|chain=...|model=..."
    """
    if isinstance(inp, dict):
        # minimal stability: ensure keys exist (may be None)
        return {
            "source": inp.get("source"),
            "chain_id": inp.get("chain_id"),
            "model_index": inp.get("model_index"),
            "format": inp.get("format"),
        }

    if isinstance(inp, str):
        # parse legacy StructureInput.to_string()
        parts = {}
        for tok in inp.split("|"):
            if "=" in tok:
                k, v = tok.split("=", 1)
                parts[k.strip()] = v.strip()
        return {
            "source": parts.get("source"),
            "chain_id": (None if parts.get("chain") in (None, "None", "") else parts.get("chain")),
            "model_index": int(parts.get("model", 0)) if parts.get("model") not in (None, "") else 0,
            "format": None,
        }

    return {"source": None, "chain_id": None, "model_index": 0, "format": None}


def validate_artifact(meta: Dict[str, Any], data: Mapping[str, np.ndarray], *, strict: bool = True) -> None:
    """
    Validate artifact schema contract.
    Raises ArtifactValidationError with actionable messages.
    """
    required = [
        "schema_version",
        "encoder_name",
        "encoder_version",
        "encoder_fingerprint",
        "type",
        "level",
        "params",
        "input",
        "shapes",
        "dtypes",
        "counts",
        "feature_keys",
    ]
    missing = [k for k in required if k not in meta]
    if missing:
        raise ArtifactValidationError(f"meta.json missing required keys: {missing}")

    if meta["schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactValidationError(
            f"Unsupported schema_version={meta['schema_version']} (expected {ARTIFACT_SCHEMA_VERSION})."
        )

    t = str(meta["type"])
    if t not in ALLOWED_TYPES:
        raise ArtifactValidationError(f"Invalid meta.type={t!r}. Allowed: {sorted(ALLOWED_TYPES)}")

    lvl = str(meta["level"])
    if lvl not in ALLOWED_LEVELS:
        raise ArtifactValidationError(f"Invalid meta.level={lvl!r}. Allowed: {sorted(ALLOWED_LEVELS)}")


    if not isinstance(meta["params"], dict):
        raise ArtifactValidationError("meta.params must be a dict")
    if not isinstance(meta["shapes"], dict) or not isinstance(meta["dtypes"], dict):
        raise ArtifactValidationError("meta.shapes and meta.dtypes must be dicts")
    if not isinstance(meta["counts"], dict):
        raise ArtifactValidationError("meta.counts must be a dict")
    if not isinstance(meta["feature_keys"], dict):
        raise ArtifactValidationError("meta.feature_keys must be a dict")
    _validate_feature_keys(meta["feature_keys"], data)

    # Ensure input is normalized dict
    if not isinstance(meta["input"], dict):
        raise ArtifactValidationError("meta.input must be a dict (source/chain_id/model_index/format)")

    # Data must be numpy arrays only (avoid pickles)
    bad = [k for k, v in data.items() if not isinstance(v, np.ndarray)]
    if bad:
        raise ArtifactValidationError(f"data.npz contains non-numpy entries: {bad}")

    # Basic consistency checks (strict)
    if strict:
        shapes, dtypes = infer_shapes_dtypes(data)
        if meta["shapes"] != shapes:
            raise ArtifactValidationError("meta.shapes does not match data.npz arrays (re-save artifact).")
        if meta["dtypes"] != dtypes:
            raise ArtifactValidationError("meta.dtypes does not match data.npz arrays (re-save artifact).")


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
        schema = _detect_artifact_schema(self.meta)
        if schema == "v2":
            canonical_meta = _validate_artifact_v2(self.meta, self.data)
            if canonical_meta != self.meta:
                self.meta = canonical_meta
            return

        self._finalize_v1()

    def _finalize_v1(self) -> None:
        # Normalize/upgrade meta in-place and (re)compute derived fields
        self.meta["schema_version"] = ARTIFACT_SCHEMA_VERSION

        # normalize input
        self.meta["input"] = normalize_input_meta(self.meta.get("input"))

        # derived
        shapes, dtypes = infer_shapes_dtypes(self.data)
        self.meta["shapes"] = shapes
        self.meta["dtypes"] = dtypes
        self.meta["counts"] = infer_counts(self.data)
        feature_keys = infer_feature_keys(self.meta, self.data)
        _validate_feature_keys(feature_keys, self.data)
        self.meta["feature_keys"] = feature_keys


    def validate(self, strict = False):
        schema = _detect_artifact_schema(self.meta)
        if schema == "provisional_v1":
            raise ArtifactValidationError(
                "Provisional Schema v1 artifacts must be finalized before validation."
            )
        if schema == "v2":
            _validate_artifact_v2(self.meta, self.data)
            return
        validate_artifact(self.meta, self.data, strict=strict)


    def save(self, outdir: Path, *, validate: bool = True) -> None:
        self.finalize()
        schema = _detect_artifact_schema(self.meta)

        if schema == "v2":
            self.validate(strict=True)
            output_names = [output["name"] for output in self.meta["outputs"]]
            serialized_data = {name: self.data[name] for name in output_names}
            serialized_meta = json.dumps(
                self.meta,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
        else:
            if validate:
                self.validate(strict=True)
            serialized_data = self.data
            serialized_meta = json.dumps(self.meta, indent=2, sort_keys=True)

        outdir.mkdir(parents=True, exist_ok=True)
        _save_staged(
            outdir,
            data=serialized_data,
            serialized_meta=serialized_meta,
        )

    @staticmethod
    def load(indir: Path, *, validate: bool = True, strict: bool = False) -> "StructureArtifact":
        if not indir.exists():
            raise ArtifactValidationError(f"Artifact directory does not exist: {indir}")
        if not indir.is_dir():
            raise ArtifactValidationError(f"Artifact path is not a directory: {indir}")

        meta_path = indir / "meta.json"
        data_path = indir / "data.npz"

        if not meta_path.exists() or not data_path.exists():
            raise ArtifactValidationError(f"Missing artifact files in {indir} (need meta.json + data.npz)")

        meta = _read_artifact_metadata(meta_path)
        schema = _detect_artifact_schema(meta)
        if schema == "provisional_v1":
            raise ArtifactValidationError(
                "Stored Schema v1 artifacts must contain schema_version=1."
            )

        if schema == "v2":
            output_names = _v2_output_names(meta)
            data = _load_npz_data(data_path, output_names=output_names)
            canonical_meta = _validate_artifact_v2(meta, data)
            return StructureArtifact(data=data, meta=canonical_meta)

        data = _load_npz_data(data_path)

        art = StructureArtifact(data=data, meta=meta)
        # do not auto-finalize on load; validate the stored content
        if validate:
            art.validate(strict=strict)
        return art
