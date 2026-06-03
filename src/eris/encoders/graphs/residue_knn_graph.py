from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, ClassVar
from typing import TYPE_CHECKING

import numpy as np

from eris.features.node.residue import build_node_attr
from eris.features.edge.edge import build_edge_attr
from eris.features.graph_stats import compute_graph_stats, pack_graph_stats

from eris.core.base_encoder import BaseResidueEncoder, EncodingContext
from eris.features.node.residue import build_node_attr
from eris.core.representations import GraphRepresentation
from eris.spatial.knn import knn_edges

if TYPE_CHECKING:
    from eris.core.representations.base_rep import BaseRepresentation


@dataclass
class ResidueKNNGraph(BaseResidueEncoder):
    #Identity
    name: str = "residue_knn_graph"
    representation: ClassVar[BaseRepresentation] = GraphRepresentation()

    #Instance
    knn_mode: str = "local" #or global
    k: int = 16
    cutoff: float | None = 12.0  # max search radius for candidates (must be >= typical neighbor distances)
    symmetric: bool = True
    include_self_loops: bool = False
    edge_attr_mode:str = "dist+rbf+seq+local"
    node_attr_mode:str = "onehot+physchem+plddt_ca+bfactor_ca"
    rbf_k:int = 16
    rbf_dmax:float = 20.0
    local_cutoff:float = 4


    #Overrides post_init to normalize cutoff before fingerprint calculation
    def __post_init__(self):
        super().__post_init__()
        if self.knn_mode == "global":
            self.cutoff = None


    def _create_data(self, ctx: EncodingContext) -> Dict[str, Any]:
        strc = ctx.raw_structure
        centroid = ctx.centroid
        pos_out = ctx.pos
        n = int(pos_out.shape[0])

        # # Candidates
        # if self.knn_mode.lower() == "local":
        #     candidates = _collect_local_neighbor_candidates(pos_out, cutoff=float(self.cutoff))
        # elif self.knn_mode.lower() == "global":
        #     candidates = _collect_global_neighbor_candidates(pos_out)
        # else:
        #     raise ValueError(
        #         f"Invalid knn_mode='{self.knn_mode}'. "
        #         "Expected 'local' or 'global'."
        #     )

        # Graph building    
        edge_index, edge_dist = knn_edges(
            pos=pos_out,
            k=int(self.k),
            cutoff= self.cutoff,
            symmetric=bool(self.symmetric)
        )

        seq_pos = np.arange(n, dtype=np.int64)

        stats = compute_graph_stats(edge_index=edge_index, edge_dist=edge_dist, n_nodes=n)
        graph_attr, graph_attr_keys = pack_graph_stats(stats)

        node_attr, node_attr_keys = build_node_attr(
            strc["aa_index"],
            plddt=strc["plddt"],
            bfactor_ca=strc["bfactor_ca"],
            mode=self.node_attr_mode,
            return_keys=True
        )

        if self.include_self_loops:
            loops = np.arange(n, dtype=np.int64)
            loop_edges = np.stack([loops, loops], axis=0)
            loop_dist = np.zeros((n,), dtype=np.float32)

            edge_index = np.concatenate([edge_index, loop_edges], axis=1)
            edge_dist = np.concatenate([edge_dist, loop_dist], axis=0)

        edge_attr, edge_attr_keys = build_edge_attr(
            edge_index=edge_index,
            edge_dist=edge_dist,
            mode=self.edge_attr_mode,
            rbf_k=self.rbf_k,
            rbf_dmax=self.rbf_dmax,
            seq_pos=seq_pos,
            local_cutoff=self.local_cutoff,
            return_keys=True
        )

        ctx.runtime["node_attr_keys"] = node_attr_keys
        ctx.runtime["edge_attr_keys"] = edge_attr_keys
        ctx.runtime["graph_attr_keys"] = graph_attr_keys

        data: Dict[str, Any] = {
            "pos": pos_out.astype(np.float32),
            "centroid": centroid.astype(np.float32),
            "residue_ids": strc["residue_ids"].astype(np.str_),
            "resname": strc["resname"].astype(np.str_),
            "aa_index": strc["aa_index"].astype(np.int16),
            "edge_index": edge_index.astype(np.int64),
            "edge_dist": edge_dist.astype(np.float32),
            "node_attr": node_attr.astype(np.float32),
            "edge_attr": edge_attr.astype(np.float32),
            "graph_attr": np.asarray(graph_attr, dtype=np.float32),
        }

        if strc["plddt"] is not None:
            data["plddt"] = strc["plddt"].astype(np.float32)
        if strc["bfactor_ca"] is not None:
            data["bfactor_ca"] = strc["bfactor_ca"].astype(np.float32)

        return data


    def _update_params_metadata(self, meta: Dict[str, Any], ctx: EncodingContext) -> Dict[str, Any]:
        meta["params"].update({
                "k": int(self.k),
                "knn_mode": self.knn_mode,
                **({"cutoff": self.cutoff} if self.knn_mode == "local" else {}),
                "symmetric": bool(self.symmetric),
                "include_self_loops": bool(self.include_self_loops),
                "node_attr_mode": self.node_attr_mode,
                "edge_attr_mode": self.edge_attr_mode,
                "rbf_k": int(self.rbf_k),
                "rbf_dmax": float(self.rbf_dmax),
                "local_cutoff": int(self.local_cutoff),
        })
        
        return meta
