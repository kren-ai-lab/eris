import numpy as np

from typing import Dict, Any, List

from eris.features.utils import parse_mode, onehot, safe_float, ensure_1d
from eris.features.feature_registry import register_graph_edge_feature, FeatureSpec

from eris.features.rbf import compute_rbf as rbf_encode

# -------------------------
# dist
# -------------------------
def dist_fn(data: Dict[str, Any], ctx: Dict[str, Any]) -> np.ndarray:
    dist = data["dist"]
    return dist.reshape(-1, 1).astype(np.float32)


def dist_keys(data: Dict[str, Any], ctx: Dict[str, Any]) -> List[str]:
    return ["dist"]


register_graph_edge_feature(
    FeatureSpec(
        token="dist",
        order=10,
        requires={"dist"},
        fn=dist_fn,
        keys_fn=dist_keys,
    ))


# -------------------------
# rbf
# -------------------------
def rbf_fn(data: Dict[str, Any], ctx: Dict[str, Any]) -> np.ndarray:
    dist = data["dist"]
    E = data["E"]
    rbf_k = ctx["rbf_k"]
    rbf_dmax = ctx["rbf_dmax"]

    if E == 0:
        r = np.zeros((0, int(rbf_k)), dtype=np.float32)
    else:
        r = rbf_encode(dist, num_kernels=int(rbf_k), dmin=0.0, dmax=float(rbf_dmax))

    return r.astype(np.float32)


def rbf_keys(data: Dict[str, Any], ctx: Dict[str, Any]) -> List[str]:
    rbf_k = ctx["rbf_k"]
    return [f"rbf.{i}" for i in range(int(rbf_k))]


register_graph_edge_feature(
    FeatureSpec(
        token="rbf",
        order=20,
        requires={"dist", "E", "rbf_k", "rbf_dmax"},
        fn=rbf_fn,
        keys_fn=rbf_keys,
    ))


# -------------------------
# seq_sep
# -------------------------
def seq_sep_fn(data: Dict[str, Any], ctx: Dict[str, Any]) -> np.ndarray:
    seq_sep = ctx["seq_sep"]
    return seq_sep.reshape(-1, 1).astype(np.float32)


def seq_sep_keys(data: Dict[str, Any], ctx: Dict[str, Any]) -> List[str]:
    return ["seq_sep"]


register_graph_edge_feature(
    FeatureSpec(
        token="seq",
        order=30,
        requires={"seq_sep"},
        fn=seq_sep_fn,
        keys_fn=seq_sep_keys,
    ))


# -------------------------
# local
# -------------------------
def local_fn(data: Dict[str, Any], ctx: Dict[str, Any]) -> np.ndarray:
    seq_sep = ctx["seq_sep"]
    local_cutoff = ctx["local_cutoff"]

    is_local = (seq_sep <= float(local_cutoff)).astype(np.float32).reshape(-1, 1)
    return is_local.astype(np.float32)


def local_keys(data: Dict[str, Any], ctx: Dict[str, Any]) -> List[str]:
    local_cutoff = ctx["local_cutoff"]

    return [f"flag.is_local_le_{int(local_cutoff)}"]


register_graph_edge_feature(
    FeatureSpec(
        token="local",
        order=40,
        requires={"seq_sep", "local_cutoff"},
        fn=local_fn,
        keys_fn=local_keys,
    ))
