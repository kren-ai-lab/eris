from __future__ import annotations

from typing import Dict, Any, List
import numpy as np

from eris.features.node.constants import (
    AA_VOCAB,
    UNKNOWN_INDEX,
    PHYSCHEM_LOOKUPS,
    PHYSchem_KEYS,
    AAINDEX_BASIC_KEYS,
    AAINDEX_BASIC_SUBIDX
)

from eris.features.utils import parse_mode, onehot, safe_float, ensure_1d
from eris.features.feature_registry import register_residue_node_feature, FeatureSpec

# --- canonical ordering (NO CAMBIAR en releases)
_AA_ONEHOT_KEYS = [f"aa_onehot.{aa}" for aa in AA_VOCAB]
_AA_INDEX_KEY = ["aa_index"]



def aa_index_to_letters(aa_index: np.ndarray) -> np.ndarray:
    aa_index = np.asarray(aa_index, dtype=np.int64)
    aa_index = np.clip(aa_index, 0, len(AA_VOCAB) - 1)
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
    ))

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
    ))

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
    ))


# -------------------------
# aaindex_basic
# -------------------------
def aaindex_basic_fn(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
    aa_index = np.asarray(data["aa_index"], dtype=np.int64)
    feats = [lookup[aa_index] for _, lookup in PHYSCHEM_LOOKUPS]

    subset_idx = AAINDEX_BASIC_SUBIDX
    physchem_full = np.stack(feats, axis=1)

    return physchem_full[:, subset_idx]



def aaindex_basic_keys(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
    return AAINDEX_BASIC_KEYS


register_residue_node_feature(
    FeatureSpec(
        token="aaindex_basic",
        order=40,
        requires={"aa_index"},
        fn=aaindex_basic_fn,
        keys_fn=aaindex_basic_keys,
    ))

# -------------------------
# plddt_ca
# -------------------------

def plddt_ca_fn(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
    n = data["n"]
    missing=ctx["missing"]

    if data["plddt_ca"] is None:
        v = np.full((n, 1), np.nan if missing == "nan" else 0.0, dtype=np.float32)
    else:
        v = safe_float(data["plddt_ca"], missing=missing).reshape(-1, 1)
        # normalize heuristic (0..100 -> 0..1)
        if missing != "nan" and float(np.nanmax(v)) > 1.5:
            v = v / 100.0
        if missing != "nan":
            v = np.clip(v, 0.0, 1.0)
    return v.astype(np.float32)


def plddt_ca_keys(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
    return ["plddt_ca"]


register_residue_node_feature(
    FeatureSpec(
        token="plddt_ca",
        order=50,
        requires={"n", "plddt_ca"},
        fn=plddt_ca_fn,
        keys_fn=plddt_ca_keys,
    ))


# -------------------------
# bfactor_ca
# -------------------------

def bfactor_ca_fn(data: Dict[str, Any], ctx: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
    n = data["n"]
    missing=ctx["missing"]

    if data["bfactor_ca"] is None:
        v = np.full((n, 1), np.nan if missing == "nan" else 0.0, dtype=np.float32)
    else:
        v = safe_float(data["bfactor_ca"], missing=missing).reshape(-1, 1)
        # normalize heuristic (0..100 -> 0..1)
        if missing != "nan" and float(np.nanmax(v)) > 1.5:
            v = v / 100.0
        if missing != "nan":
            v = np.clip(v, 0.0, 1.0)
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
    ))


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
    ))




