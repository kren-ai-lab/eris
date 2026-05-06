from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, ClassVar
from typing import TYPE_CHECKING

import numpy as np

from morpheo.core.types import StructureArtifact, StructureInput
from morpheo.core.base_encoder import BaseAtomEncoder, EncodingContext
from morpheo.core.contracts import AtomContract
from morpheo.core.representations import GraphRepresentation

if TYPE_CHECKING:
    from morpheo.core.contracts.base_contract import GranularityContract
    from morpheo.core.representations.base_rep import BaseRepresentation


# -----------------------------
# Fixed dims (must be stable!)
# -----------------------------
ELEM_DIM = 17   # must match io.py _ELEM_VOCAB length
ROLE_DIM = 6    # roles 0..5 in io.py _atom_role


def _onehot_fixed(idx: np.ndarray, dim: int) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    out = np.zeros((idx.shape[0], dim), dtype=np.float32)
    m = (idx >= 0) & (idx < dim)
    out[np.arange(idx.shape[0])[m], idx[m]] = 1.0
    return out


def _rbf(dist: np.ndarray, *, k: int = 16, dmax: float = 20.0) -> np.ndarray:
    """
    Gaussian RBF features over distance.
    Returns (E,k).
    """
    d = np.asarray(dist, dtype=np.float32).reshape(-1, 1)
    # centers in [0, dmax]
    centers = np.linspace(0.0, float(dmax), int(k), dtype=np.float32).reshape(1, -1)
    # sigma: spacing
    if k <= 1:
        sigma = 1.0
    else:
        sigma = float(dmax) / float(k - 1)
        sigma = max(sigma, 1e-6)
    rbf = np.exp(-((d - centers) ** 2) / (2.0 * (sigma ** 2))).astype(np.float32)
    return rbf


def _knn_edge_index(pos: np.ndarray, k: int) -> np.ndarray:
    """
    Naive O(N^2) kNN (directed).
    edge_index shape (2,E)
    """
    pos = np.asarray(pos, dtype=np.float32)
    n = int(pos.shape[0])
    if n <= 1:
        return np.zeros((2, 0), dtype=np.int64)

    k_eff = int(max(1, min(k, n - 1)))

    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)
    np.fill_diagonal(dist, np.inf)

    nn = np.argpartition(dist, kth=k_eff - 1, axis=1)[:, :k_eff]  # (N,k)
    src = np.repeat(np.arange(n, dtype=np.int64), k_eff)
    dst = nn.reshape(-1).astype(np.int64)
    return np.stack([src, dst], axis=0)


def _edge_dist(pos: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    if edge_index.shape[1] == 0:
        return np.zeros((0,), dtype=np.float32)
    src = edge_index[0]
    dst = edge_index[1]
    d = pos[src] - pos[dst]
    return np.sqrt((d * d).sum(axis=1)).astype(np.float32)

#Repeated?
def _build_node_attr(
    *,
    elem_index: np.ndarray,
    atom_role: np.ndarray,
    bfactor: np.ndarray,
    occ: np.ndarray,
    bfactor_norm: bool = True,
) -> np.ndarray:
    """
    node_attr: [onehot(elem,17), onehot(role,6), bfactor(1), occ(1)] => (N,25)
    """
    e = _onehot_fixed(elem_index, ELEM_DIM)            # (N,17)
    r = _onehot_fixed(atom_role, ROLE_DIM)            # (N,6)

    bf = np.nan_to_num(np.asarray(bfactor, dtype=np.float32), nan=0.0).reshape(-1, 1)
    oc = np.nan_to_num(np.asarray(occ, dtype=np.float32), nan=0.0).reshape(-1, 1)

    if bfactor_norm:
        # if it looks like AF2 pLDDT or typical bfactor scale, normalize to 0..1 via /100
        if float(np.nanmax(bf)) > 1.5:
            bf = bf / 100.0

    return np.concatenate([e, r, bf.astype(np.float32), oc.astype(np.float32)], axis=1).astype(np.float32)


def _build_edge_attr(
    *,
    edge_dist: np.ndarray,
    edge_index: np.ndarray,
    seq_pos_residue: np.ndarray,
    rbf_k: int = 16,
    rbf_dmax: float = 20.0,
    local_cutoff: int = 4,
) -> np.ndarray:
    """
    edge_attr: [dist(1), rbf(16), seq_sep(1), local(1)] => (E,19)
    """
    d = np.asarray(edge_dist, dtype=np.float32).reshape(-1, 1)  # (E,1)
    rbf = _rbf(edge_dist, k=rbf_k, dmax=rbf_dmax)               # (E,K)

    if edge_index.shape[1] == 0:
        seq_sep = np.zeros((0, 1), dtype=np.float32)
        local = np.zeros((0, 1), dtype=np.float32)
    else:
        sp = np.asarray(seq_pos_residue, dtype=np.int64).reshape(-1)
        src = edge_index[0]
        dst = edge_index[1]
        sep = np.abs(sp[src] - sp[dst]).astype(np.float32).reshape(-1, 1)
        seq_sep = sep
        local = (sep <= float(local_cutoff)).astype(np.float32).reshape(-1, 1)

    return np.concatenate([d, rbf, seq_sep, local], axis=1).astype(np.float32)


def _graph_stats(edge_index: np.ndarray, edge_dist: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      graph_attr (G,) float32
      graph_attr_keys (G,) str
    """
    e = int(edge_index.shape[1])
    # degree (directed)
    deg = np.zeros((n,), dtype=np.float32)
    if e > 0:
        src = edge_index[0]
        np.add.at(deg, src, 1.0)

    mean_deg = float(np.mean(deg)) if n > 0 else 0.0
    std_deg = float(np.std(deg)) if n > 0 else 0.0

    if e > 0:
        md = float(np.mean(edge_dist))
        sd = float(np.std(edge_dist))
    else:
        md, sd = 0.0, 0.0

    graph_attr = np.asarray(
        [float(n), float(e), mean_deg, std_deg, md, sd],
        dtype=np.float32,
    )
    graph_attr_keys = np.asarray(
        ["n_nodes", "n_edges", "deg_mean", "deg_std", "edge_dist_mean", "edge_dist_std"],
        dtype=np.str_,
    )
    return graph_attr, graph_attr_keys


@dataclass
class AtomKNNGraph(BaseAtomEncoder):
    """
    Atom-level kNN graph (CA-independent) with deterministic node/edge features baseline.

    Outputs:
      - pos, edge_index, edge_dist
      - node_attr (fixed dim)
      - edge_attr (dist+rbf+seq+local)
      - graph_attr (+ keys)
      - plus raw per-atom fields: elem_index, atom_role, bfactor, occ, residue_ids, atom_name, etc.
    """
    # Identity
    name: ClassVar[str] = "atom_knn_graph"
    representation: ClassVar[BaseRepresentation] = GraphRepresentation()

    # Instance 
    k: int = 16
    include_self_loops: bool = False

    include_hetero: bool = False
    include_h: bool = False
    altloc: str = "first"
    amino_acids_only: bool = False

    # edge feature params
    rbf_k: int = 16
    rbf_dmax: float = 20.0
    local_cutoff: int = 4

    def _create_data(self, ctx: EncodingContext) -> Dict[str, Any]:
        strc = ctx.raw_structure
        centroid = ctx.centroid
        pos_out = ctx.pos

        pos = pos_out.astype(np.float32)
        n = int(pos.shape[0])

        edge_index = _knn_edge_index(pos, k=int(self.k))

        if (not self.include_self_loops) and edge_index.shape[1] > 0:
            src = edge_index[0]
            dst = edge_index[1]
            keep = src != dst
            edge_index = edge_index[:, keep]

        edge_dist = _edge_dist(pos, edge_index)

        # deterministic baseline attributes
        node_attr = _build_node_attr(
            elem_index=strc["elem_index"],
            atom_role=strc["atom_role"],
            bfactor=strc["bfactor"],
            occ=strc["occ"],
            bfactor_norm=True,
        )

        edge_attr = _build_edge_attr(
            edge_dist=edge_dist,
            edge_index=edge_index,
            seq_pos_residue=strc["seq_pos_residue"],
            rbf_k=int(self.rbf_k),
            rbf_dmax=float(self.rbf_dmax),
            local_cutoff=int(self.local_cutoff),
        )

        graph_attr, graph_attr_keys = _graph_stats(edge_index, edge_dist, n=n)

        data: Dict[str, Any] = {
            "pos": pos.astype(np.float32),
            "centroid": centroid.astype(np.float32),
            "edge_index": edge_index.astype(np.int64),
            "edge_dist": edge_dist.astype(np.float32),
            "node_attr": node_attr.astype(np.float32),
            "edge_attr": edge_attr.astype(np.float32),
            "graph_attr": graph_attr.astype(np.float32),
            "graph_attr_keys": graph_attr_keys.astype(np.str_),

            # raw atom fields (useful + deterministic)
            "elem_index": strc["elem_index"].astype(np.int16),
            "atom_role": strc["atom_role"].astype(np.int8),
            "bfactor": strc["bfactor"].astype(np.float32),
            "occ": strc["occ"].astype(np.float32),
            "residue_ids": strc["residue_ids"].astype(np.str_),
            "resname": strc["resname"].astype(np.str_),
            "atom_name": strc["atom_name"].astype(np.str_),
            "seq_pos_residue": strc["seq_pos_residue"].astype(np.int32),
            "aa_index_residue": strc["aa_index_residue"].astype(np.int16),
        }

        return data


    def _update_params_metadata(self, meta: Dict[str, Any], ctx: EncodingContext) -> Dict[str, Any]:
        meta["params"].update({
                "k": int(self.k),
                "include_self_loops": bool(self.include_self_loops),
                "include_hetero": bool(self.include_hetero),
                "include_h": bool(self.include_h),
                "altloc": str(self.altloc),
                "amino_acids_only": bool(self.amino_acids_only),
                "rbf_k": int(self.rbf_k),
                "rbf_dmax": float(self.rbf_dmax),
                "local_cutoff": int(self.local_cutoff),
                "bfactor_norm": True,
        })

            # "feature_keys": {
            # "node_attr_keys": node_attr_keys,
            # "edge_attr_keys": edge_attr_keys,
            # "graph_attr_keys": graph_keys_list,

        return meta
