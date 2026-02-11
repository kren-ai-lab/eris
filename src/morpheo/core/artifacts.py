from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from morpheo.exceptions import MorpheoError


ARTIFACT_SCHEMA_VERSION = 1

ALLOWED_TYPES = {"graph", "matrix", "cloud", "grid", "mesh"}
ALLOWED_LEVELS = {"residue", "ca", "atom"}
ALLOWED_FAMILIES = {"residue_graph", "atom_graph", "matrix", "cloud", "grid", "mesh"}


class ArtifactValidationError(MorpheoError):
    pass


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


def infer_feature_keys(meta: Dict[str, Any], data: Mapping[str, np.ndarray]) -> Dict[str, List[str]]:
    """
    Canonical place for feature key lists.
    - Prefer explicit lists already present in meta.
    - Otherwise infer only from standard *_keys arrays if present in data.
    """
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
        "family",
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

    fam = str(meta["family"])
    if fam not in ALLOWED_FAMILIES:
        raise ArtifactValidationError(f"Invalid meta.family={fam!r}. Allowed: {sorted(ALLOWED_FAMILIES)}")

    if not isinstance(meta["params"], dict):
        raise ArtifactValidationError("meta.params must be a dict")
    if not isinstance(meta["shapes"], dict) or not isinstance(meta["dtypes"], dict):
        raise ArtifactValidationError("meta.shapes and meta.dtypes must be dicts")
    if not isinstance(meta["counts"], dict):
        raise ArtifactValidationError("meta.counts must be a dict")
    if not isinstance(meta["feature_keys"], dict):
        raise ArtifactValidationError("meta.feature_keys must be a dict")

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
        # Normalize/upgrade meta in-place and (re)compute derived fields
        self.meta["schema_version"] = ARTIFACT_SCHEMA_VERSION

        # normalize input
        self.meta["input"] = normalize_input_meta(self.meta.get("input"))

        # derived
        shapes, dtypes = infer_shapes_dtypes(self.data)
        self.meta["shapes"] = shapes
        self.meta["dtypes"] = dtypes
        self.meta["counts"] = infer_counts(self.data)
        self.meta["feature_keys"] = infer_feature_keys(self.meta, self.data)

    def save(self, outdir: Path, *, validate: bool = True) -> None:
        outdir.mkdir(parents=True, exist_ok=True)

        self.finalize()

        if validate:
            validate_artifact(self.meta, self.data, strict=True)

        np.savez_compressed(outdir / "data.npz", **self.data)
        (outdir / "meta.json").write_text(json.dumps(self.meta, indent=2, sort_keys=True), encoding="utf-8")

    @staticmethod
    def load(indir: Path, *, validate: bool = True, strict: bool = False) -> "StructureArtifact":
        meta_path = indir / "meta.json"
        data_path = indir / "data.npz"

        if not meta_path.exists() or not data_path.exists():
            raise ArtifactValidationError(f"Missing artifact files in {indir} (need meta.json + data.npz)")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        with np.load(data_path, allow_pickle=False) as npz:
            data = {k: npz[k] for k in npz.files}

        art = StructureArtifact(data=data, meta=meta)
        # do not auto-finalize on load; validate the stored content
        if validate:
            validate_artifact(meta, data, strict=strict)
        return art
