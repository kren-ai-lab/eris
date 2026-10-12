"""Graph edge feature implementations and their registration."""

from numbers import Real
from typing import Any

import numpy as np

from eris.features.feature_registry import FeatureSpec, register_graph_edge_feature
from eris.features.geometry import FRAME_EPSILON, local_frames, relative_directions
from eris.features.interface import normalize_interface_params
from eris.features.rbf import compute_rbf as rbf_encode


# -------------------------
# dist
# -------------------------
def dist_fn(data: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]) -> np.ndarray:
    """Compute the ``dist`` feature values."""
    dist = data["dist"]
    return dist.reshape(-1, 1).astype(np.float32)


def dist_keys(data: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]) -> list[str]:
    """Return the ``dist`` column keys."""
    return ["dist"]


register_graph_edge_feature(
    FeatureSpec(
        token="dist",
        order=10,
        requires={"dist"},
        fn=dist_fn,
        keys_fn=dist_keys,
    )
)


# -------------------------
# chain relationship
# -------------------------
def same_chain_fn(
    data: dict[str, Any],
    ctx: dict[str, Any],
    params: dict[str, Any],
) -> np.ndarray:
    """Compute the ``same_chain`` feature values."""
    return np.asarray(ctx["same_chain"], dtype=np.float32).reshape(-1, 1)


def same_chain_keys(
    data: dict[str, Any],
    ctx: dict[str, Any],
    params: dict[str, Any],
) -> list[str]:
    """Return the ``same_chain`` column keys."""
    return ["edge.same_chain"]


register_graph_edge_feature(
    FeatureSpec(
        token="same_chain",
        order=50,
        requires={"same_chain"},
        fn=same_chain_fn,
        keys_fn=same_chain_keys,
    )
)


def inter_chain_fn(
    data: dict[str, Any],
    ctx: dict[str, Any],
    params: dict[str, Any],
) -> np.ndarray:
    """Compute the ``inter_chain`` feature values."""
    return np.asarray(~ctx["same_chain"], dtype=np.float32).reshape(-1, 1)


def inter_chain_keys(
    data: dict[str, Any],
    ctx: dict[str, Any],
    params: dict[str, Any],
) -> list[str]:
    """Return the ``inter_chain`` column keys."""
    return ["edge.inter_chain"]


register_graph_edge_feature(
    FeatureSpec(
        token="inter_chain",
        order=60,
        requires={"same_chain"},
        fn=inter_chain_fn,
        keys_fn=inter_chain_keys,
    )
)


# -------------------------
# source-frame relative direction
# -------------------------
_FRAME_ATOMS = ["N", "CA", "C"]
_FRAME_ORIGIN = "CA"
_FRAME_DEFINITION = "x=C-CA;y=orthogonalized(N-CA);z=x_cross_y"
_FRAME_HANDEDNESS = "right"
_FRAME_MATRIX_LAYOUT = "columns_xyz_global"
_DIRECTION_CONVENTION = "source_frame_transpose_unit_ca_displacement"
_MISSING_FRAME_POLICY = "nan"


def normalize_relative_direction_params(params: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize ``relative_direction`` parameters."""
    fixed = {
        "frame_atoms": _FRAME_ATOMS,
        "frame_origin": _FRAME_ORIGIN,
        "frame_definition": _FRAME_DEFINITION,
        "frame_handedness": _FRAME_HANDEDNESS,
        "frame_matrix_layout": _FRAME_MATRIX_LAYOUT,
        "direction_convention": _DIRECTION_CONVENTION,
        "missing_frame_policy": _MISSING_FRAME_POLICY,
    }
    allowed = {*fixed, "frame_epsilon"}
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(f"Unknown relative_direction parameters: {unknown}")

    epsilon = params.get("frame_epsilon", FRAME_EPSILON)
    if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
        raise TypeError("relative_direction parameter 'frame_epsilon' must be a real number")
    epsilon = float(epsilon)
    if epsilon != FRAME_EPSILON:
        raise ValueError(f"relative_direction parameter 'frame_epsilon' must equal {FRAME_EPSILON}")

    for name, expected in fixed.items():
        if name in params and params[name] != expected:
            raise ValueError(f"relative_direction parameter '{name}' must be {expected!r}")
    return {**fixed, "frame_epsilon": epsilon}


def relative_direction_fn(
    data: dict[str, Any],
    ctx: dict[str, Any],
    params: dict[str, Any],
) -> np.ndarray:
    """Compute the ``relative_direction`` feature values."""
    frames, frame_valid = local_frames(
        data["backbone_n"],
        data["backbone_ca"],
        data["backbone_c"],
        data["backbone_n_mask"],
        data["backbone_ca_mask"],
        data["backbone_c_mask"],
        frame_epsilon=params["frame_epsilon"],
    )
    return relative_directions(
        data["ei"],
        data["backbone_ca"],
        data["backbone_ca_mask"],
        frames,
        frame_valid,
    )


def relative_direction_keys(
    data: dict[str, Any],
    ctx: dict[str, Any],
    params: dict[str, Any],
) -> list[str]:
    """Return the ``relative_direction`` column keys."""
    return [
        "edge.relative_direction_x",
        "edge.relative_direction_y",
        "edge.relative_direction_z",
    ]


register_graph_edge_feature(
    FeatureSpec(
        token="relative_direction",
        order=70,
        requires={
            "ei",
            "backbone_n",
            "backbone_ca",
            "backbone_c",
            "backbone_n_mask",
            "backbone_ca_mask",
            "backbone_c_mask",
        },
        fn=relative_direction_fn,
        keys_fn=relative_direction_keys,
        params_fn=normalize_relative_direction_params,
    )
)


# -------------------------
# inter-chain interface contact
# -------------------------
def interface_contact_fn(
    data: dict[str, Any],
    ctx: dict[str, Any],
    params: dict[str, Any],
) -> np.ndarray:
    """Compute the ``interface_contact`` feature values."""
    return np.asarray(ctx["interface_contact"], dtype=np.float32).reshape(-1, 1)


def interface_contact_keys(
    data: dict[str, Any],
    ctx: dict[str, Any],
    params: dict[str, Any],
) -> list[str]:
    """Return the ``interface_contact`` column keys."""
    return ["edge.interface_contact"]


register_graph_edge_feature(
    FeatureSpec(
        token="interface_contact",
        order=80,
        requires={"interface_contact"},
        fn=interface_contact_fn,
        keys_fn=interface_contact_keys,
        params_fn=normalize_interface_params,
    )
)


# -------------------------
# rbf
# -------------------------
def rbf_fn(data: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]) -> np.ndarray:
    """Compute the ``rbf`` feature values."""
    dist = data["dist"]
    E = data["E"]
    rbf_k = ctx["rbf_k"]
    rbf_dmax = ctx["rbf_dmax"]

    if E == 0:
        r = np.zeros((0, int(rbf_k)), dtype=np.float32)
    else:
        r = rbf_encode(dist, num_kernels=int(rbf_k), dmin=0.0, dmax=float(rbf_dmax))

    return r.astype(np.float32)


def rbf_keys(data: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]) -> list[str]:
    """Return the ``rbf`` column keys."""
    rbf_k = ctx["rbf_k"]
    return [f"rbf.{i}" for i in range(int(rbf_k))]


register_graph_edge_feature(
    FeatureSpec(
        token="rbf",
        order=20,
        requires={"dist", "E", "rbf_k", "rbf_dmax"},
        fn=rbf_fn,
        keys_fn=rbf_keys,
    )
)


# -------------------------
# seq_sep
# -------------------------
def seq_sep_fn(data: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]) -> np.ndarray:
    """Compute the ``seq_sep`` feature values."""
    seq_sep = ctx["seq_sep"]
    return seq_sep.reshape(-1, 1).astype(np.float32)


def seq_sep_keys(data: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]) -> list[str]:
    """Return the ``seq_sep`` column keys."""
    return ["seq_sep"]


register_graph_edge_feature(
    FeatureSpec(
        token="seq",
        order=30,
        requires={"seq_sep"},
        fn=seq_sep_fn,
        keys_fn=seq_sep_keys,
    )
)


# -------------------------
# local
# -------------------------
def local_fn(data: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]) -> np.ndarray:
    """Compute the ``local`` feature values."""
    seq_sep = ctx["seq_sep"]
    local_cutoff = ctx["local_cutoff"]

    is_local = ((seq_sep >= 0.0) & (seq_sep <= float(local_cutoff))).astype(np.float32)
    is_local = is_local.reshape(-1, 1)
    return is_local.astype(np.float32)


def local_keys(data: dict[str, Any], ctx: dict[str, Any], params: dict[str, Any]) -> list[str]:
    """Return the ``local`` column keys."""
    local_cutoff = ctx["local_cutoff"]

    return [f"flag.is_local_le_{int(local_cutoff)}"]


register_graph_edge_feature(
    FeatureSpec(
        token="local",
        order=40,
        requires={"seq_sep", "local_cutoff"},
        fn=local_fn,
        keys_fn=local_keys,
    )
)
