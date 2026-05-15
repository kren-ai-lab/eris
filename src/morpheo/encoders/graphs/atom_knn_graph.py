from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, ClassVar
from typing import TYPE_CHECKING

import numpy as np

from morpheo.core.base_encoder import BaseAtomEncoder, EncodingContext
from morpheo.core.representations import GraphRepresentation
from morpheo.features.node.atom import build_node_attr
from morpheo.features.edge.edge import build_edge_attr
from morpheo.features.graph_stats import compute_graph_stats, pack_graph_stats
from morpheo.features.spatial.knn import knn_edges

if TYPE_CHECKING:
    from morpheo.core.representations.base_rep import BaseRepresentation


# def _knn_edge_index(pos: np.ndarray, k: int) -> np.ndarray:
#     """
#     Naive O(N^2) kNN (directed).
#     edge_index shape (2,E)
#     """
#     pos = np.asarray(pos, dtype=np.float32)
#     n = int(pos.shape[0])
#     if n <= 1:
#         return np.zeros((2, 0), dtype=np.int64)

#     k_eff = int(max(1, min(k, n - 1)))

#     diff = pos[:, None, :] - pos[None, :, :]
#     dist = np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)
#     np.fill_diagonal(dist, np.inf)

#     nn = np.argpartition(dist, kth=k_eff - 1, axis=1)[:, :k_eff]  # (N,k)
#     src = np.repeat(np.arange(n, dtype=np.int64), k_eff)
#     dst = nn.reshape(-1).astype(np.int64)
#     return np.stack([src, dst], axis=0)


# def _edge_dist(pos: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
#     if edge_index.shape[1] == 0:
#         return np.zeros((0,), dtype=np.float32)
#     src = edge_index[0]
#     dst = edge_index[1]
#     d = pos[src] - pos[dst]
#     return np.sqrt((d * d).sum(axis=1)).astype(np.float32)


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
    cutoff: float | None = 12.0 # max search radius for candidates (must be >= typical neighbor distances) None = Global mode
    symmetric: bool = True
    include_self_loops: bool = False

    # include_hetero: bool = False
    # include_h: bool = False
    # altloc: str = "first"
    # amino_acids_only: bool = False

    node_attr_mode: str = "elem+role+bfactor+occ"
    
    # edge feature params
    edge_mode = "dist+rbf+seq+local"
    rbf_k: int = 16
    rbf_dmax: float = 20.0
    local_cutoff: int = 4

    def _create_data(self, ctx: EncodingContext) -> Dict[str, Any]:
        strc = ctx.raw_structure
        centroid = ctx.centroid
        pos_out = ctx.pos

        pos = pos_out.astype(np.float32)
        n = int(pos.shape[0])

        # edge_index = _knn_edge_index(pos, k=int(self.k))

        # if (not self.include_self_loops) and edge_index.shape[1] > 0:
        #     src = edge_index[0]
        #     dst = edge_index[1]
        #     keep = src != dst
        #     edge_index = edge_index[:, keep]

        # edge_dist = _edge_dist(pos, edge_index)

        edge_index, edge_dist = knn_edges(
            pos=pos_out,
            k=int(self.k),
            cutoff= self.cutoff,
            symmetric=bool(self.symmetric)
        )


        # deterministic baseline attributes
        node_attr = build_node_attr(
            elem_index=strc["elem_index"],
            atom_role=strc["atom_role"],
            bfactor=strc["bfactor"],
            occ=strc["occ"],
            bfactor_norm=True,
            mode=self.node_attr_mode
        )

        edge_attr = build_edge_attr(
            edge_dist=edge_dist,
            edge_index=edge_index,
            mode = self.edge_mode,
            seq_pos=strc["seq_pos_residue"],
            rbf_k=int(self.rbf_k),
            rbf_dmax=float(self.rbf_dmax),
            local_cutoff=int(self.local_cutoff),
        )

        stats = compute_graph_stats(edge_index, edge_dist, n_nodes=n)
        graph_attr, graph_attr_keys = pack_graph_stats(stats)

        data: Dict[str, Any] = {
            "pos": pos.astype(np.float32),
            "centroid": centroid.astype(np.float32),
            "edge_index": edge_index.astype(np.int64),
            "edge_dist": edge_dist.astype(np.float32),
            "node_attr": node_attr.astype(np.float32),
            "edge_attr": edge_attr.astype(np.float32),
            "graph_attr": graph_attr.astype(np.float32),
            "graph_attr_keys": np.asarray(graph_attr_keys, dtype=np.str_),

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
                # "include_hetero": bool(self.include_hetero),
                # "include_h": bool(self.include_h),
                # "altloc": str(self.altloc),
                # "amino_acids_only": bool(self.amino_acids_only),
                "rbf_k": int(self.rbf_k),
                "rbf_dmax": float(self.rbf_dmax),
                "local_cutoff": int(self.local_cutoff),
                "bfactor_norm": True,
        })

        return meta
