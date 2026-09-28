from __future__ import annotations

from importlib import metadata
from numbers import Real
from typing import Any, Dict, List

import numpy as np

from eris.features.node.constants import (
    AA_VOCAB,
    UNKNOWN_INDEX,
    PHYSCHEM_LOOKUPS,
    PHYSchem_KEYS,
)

from eris.features.utils import onehot, safe_float
from eris.features.feature_registry import FeatureSpec, register_residue_node_feature

# --- canonical ordering (NO CAMBIAR en releases)
_AA_ONEHOT_KEYS = [f"aa_onehot.{aa}" for aa in AA_VOCAB]
_AA_INDEX_KEY = ["aa_index"]


def aa_index_to_letters(aa_index: np.ndarray) -> np.ndarray:
    aa_index = np.asarray(aa_index, dtype=np.int64)
    invalid = (aa_index < 0) | (aa_index >= len(AA_VOCAB))
    if np.any(invalid):
        value = int(aa_index.flat[np.flatnonzero(invalid)[0]])
        raise ValueError(f"aa_index values must be within [0, {len(AA_VOCAB) - 1}], got {value}")
    return np.array(AA_VOCAB)[aa_index]


# -------------------------
# aa_onehot
# -------------------------
def aa_onehot_fn(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
    aa_index = np.asarray(data["aa_index"], dtype=np.int64).reshape(-1)

    oh = onehot(
        np.clip(aa_index, 0, len(AA_VOCAB) - 1),
        num_classes=len(AA_VOCAB),
    )

    return oh.astype(np.float32)


def aa_onehot_keys(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
    return _AA_ONEHOT_KEYS


register_residue_node_feature(
    FeatureSpec(
        token="onehot",
        order=10,
        requires={"aa_index"},
        fn=aa_onehot_fn,
        keys_fn=aa_onehot_keys,
    )
)


# -------------------------
# aa_index
# -------------------------
def aa_index_fn(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
    aa_index = np.asarray(data["aa_index"], dtype=np.int64).reshape(-1, 1)

    return aa_index.astype(np.float32)


def aa_index_keys(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
    return _AA_INDEX_KEY


register_residue_node_feature(
    FeatureSpec(
        token="aa_index",
        order=20,
        requires={"aa_index"},
        fn=aa_index_fn,
        keys_fn=aa_index_keys,
    )
)


# -------------------------
# physchem
# -------------------------
def physchem_fn(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
    aa_index = np.asarray(data["aa_index"], dtype=np.int64)

    feats = [lookup[aa_index] for _, lookup in PHYSCHEM_LOOKUPS]

    return np.stack(feats, axis=1).astype(np.float32)


def physchem_keys(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
    return PHYSchem_KEYS


register_residue_node_feature(
    FeatureSpec(
        token="physchem",
        order=30,
        requires={"aa_index"},
        fn=physchem_fn,
        keys_fn=physchem_keys,
    )
)


# -------------------------
# aaindex
# -------------------------

_AAINDEX_PROVIDER = "aaindex"
_AAINDEX_SOURCE = "AAIndex1"


def _choice(name: str, value: Any, choices: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        raise TypeError(f"aaindex parameter '{name}' must be a string")
    normalized = value.strip().lower()
    if normalized not in choices:
        raise ValueError(f"aaindex parameter '{name}' must be one of: {', '.join(choices)}")
    return normalized


def normalize_aaindex_params(params: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {
        "ids",
        "normalization",
        "unknown",
        "missing",
        "provider",
        "source",
        "provider_version",
    }
    unknown_params = sorted(set(params) - allowed)
    if unknown_params:
        raise ValueError(f"Unknown aaindex parameters: {unknown_params}")

    if "ids" not in params:
        raise ValueError("aaindex parameter 'ids' is required")
    ids_value = params["ids"]
    if not isinstance(ids_value, list):
        raise TypeError("aaindex parameter 'ids' must be a list of accession strings")
    if not ids_value:
        raise ValueError("aaindex parameter 'ids' must not be empty")

    accessions: list[str] = []
    for accession in ids_value:
        if not isinstance(accession, str):
            raise TypeError("aaindex parameter 'ids' must contain only strings")
        canonical = accession.strip().upper()
        if not canonical:
            raise ValueError("aaindex accession identifiers must not be empty")
        accessions.append(canonical)
    duplicates = sorted({item for item in accessions if accessions.count(item) > 1})
    if duplicates:
        raise ValueError(f"Duplicate AAIndex1 accession identifiers: {duplicates}")

    try:
        provider_version = metadata.version(_AAINDEX_PROVIDER)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("AAIndex features require the 'aaindex>=1.3.2,<2' package") from exc

    supplied_identity = {
        "provider": _AAINDEX_PROVIDER,
        "source": _AAINDEX_SOURCE,
        "provider_version": provider_version,
    }
    for name, expected in supplied_identity.items():
        if name in params and params[name] != expected:
            raise ValueError(
                f"aaindex parameter '{name}' must match the active provider value {expected!r}"
            )

    from aaindex import aaindex1

    invalid = [accession for accession in accessions if accession not in aaindex1]
    if invalid:
        raise ValueError(f"Unknown AAIndex1 accession identifiers: {invalid}")

    return {
        "ids": accessions,
        "normalization": _choice("normalization", params.get("normalization", "none"), ("none",)),
        "unknown": _choice("unknown", params.get("unknown", "nan"), ("nan", "zero", "error")),
        "missing": _choice("missing", params.get("missing", "nan"), ("nan", "zero", "error")),
        **supplied_identity,
    }


def _policy_value(policy: str, *, message: str) -> float:
    if policy == "nan":
        return float("nan")
    if policy == "zero":
        return 0.0
    raise ValueError(message)


def aaindex_fn(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
    from aaindex import aaindex1

    letters = aa_index_to_letters(data["aa_index"])
    columns: list[np.ndarray] = []
    for accession in params["ids"]:
        record_values = aaindex1.values(accession)
        column: list[float] = []
        for row, letter in enumerate(letters.tolist()):
            if letter == "X":
                column.append(
                    _policy_value(
                        params["unknown"],
                        message=(
                            f"AAIndex1 feature '{accession}' cannot encode unknown residue X "
                            f"at row {row} with unknown='error'"
                        ),
                    )
                )
                continue

            value = record_values.get(letter)
            if (
                value is None
                or isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(float(value))
            ):
                column.append(
                    _policy_value(
                        params["missing"],
                        message=(
                            f"AAIndex1 record '{accession}' has no finite value for residue "
                            f"{letter} at row {row} with missing='error'"
                        ),
                    )
                )
            else:
                column.append(float(value))
        columns.append(np.asarray(column, dtype=np.float32))

    return np.stack(columns, axis=1).astype(np.float32)


def aaindex_keys(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
    return [f"aaindex.{accession}" for accession in params["ids"]]


register_residue_node_feature(
    FeatureSpec(
        token="aaindex",
        order=40,
        requires={"aa_index"},
        fn=aaindex_fn,
        keys_fn=aaindex_keys,
        params_fn=normalize_aaindex_params,
    )
)

# -------------------------
# plddt_ca
# -------------------------


def normalize_plddt_params(params: Dict[str, Any]) -> Dict[str, Any]:
    unknown_params = sorted(set(params) - {"normalization"})
    if unknown_params:
        raise ValueError(f"Unknown plddt_ca parameters: {unknown_params}")
    normalization = params.get("normalization", "none")
    if not isinstance(normalization, str):
        raise TypeError("plddt_ca parameter 'normalization' must be a string")
    normalization = normalization.strip().lower()
    if normalization not in {"none", "unit_interval"}:
        raise ValueError("plddt_ca parameter 'normalization' must be one of: none, unit_interval")
    return {"normalization": normalization}


def plddt_ca_fn(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
    n = data["n"]
    missing = ctx["missing"]

    if data["plddt_ca"] is None:
        v = np.full((n, 1), np.nan if missing == "nan" else 0.0, dtype=np.float32)
    else:
        v = safe_float(data["plddt_ca"], missing=missing).reshape(-1, 1)
        finite = np.isfinite(v)
        invalid = finite & ((v < 0.0) | (v > 100.0))
        if np.any(invalid):
            value = float(v[np.flatnonzero(invalid)[0], 0])
            raise ValueError(f"plddt_ca values must be within [0, 100], got {value}")
        if params["normalization"] == "unit_interval":
            v = v / 100.0
    return v.astype(np.float32)


def plddt_ca_keys(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
    return ["plddt_ca"]


register_residue_node_feature(
    FeatureSpec(
        token="plddt_ca",
        order=50,
        requires={"n", "plddt_ca", "missing"},
        fn=plddt_ca_fn,
        keys_fn=plddt_ca_keys,
        params_fn=normalize_plddt_params,
    )
)


# -------------------------
# bfactor_ca
# -------------------------


def bfactor_ca_fn(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
    n = data["n"]
    missing = ctx["missing"]

    if data["bfactor_ca"] is None:
        v = np.full((n, 1), np.nan if missing == "nan" else 0.0, dtype=np.float32)
    else:
        v = safe_float(data["bfactor_ca"], missing=missing).reshape(-1, 1)
    return v.astype(np.float32)


def bfactor_ca_keys(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
    return ["bfactor_ca"]


register_residue_node_feature(
    FeatureSpec(
        token="bfactor_ca",
        order=60,
        requires={"n", "bfactor_ca", "missing"},
        fn=bfactor_ca_fn,
        keys_fn=bfactor_ca_keys,
    )
)


# -------------------------
# flags
# -------------------------


def flags_fn(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
    aa_index = np.asarray(data["aa_index"], dtype=np.int64).reshape(-1)
    has_ca = data["has_ca"]

    is_gly = (aa_index == int(list(AA_VOCAB).index("G"))).astype(np.float32).reshape(-1, 1)
    is_pro = (aa_index == int(list(AA_VOCAB).index("P"))).astype(np.float32).reshape(-1, 1)
    is_unk = (aa_index == UNKNOWN_INDEX).astype(np.float32).reshape(-1, 1)
    has_ca_f = has_ca.astype(np.float32).reshape(-1, 1)

    return np.concatenate([is_gly, is_pro, is_unk, has_ca_f], axis=1).astype(np.float32)


def flags_keys(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
    return ["flag.is_gly", "flag.is_pro", "flag.is_unk", "flag.has_ca"]


register_residue_node_feature(
    FeatureSpec(
        token="flags",
        order=70,
        requires={"aa_index", "has_ca"},
        fn=flags_fn,
        keys_fn=flags_keys,
    )
)
