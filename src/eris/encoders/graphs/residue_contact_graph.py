from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from eris.core.base_encoder import BaseResidueEncoder, EncodingContext
from eris.core.representations import GraphRepresentation
from eris.features.edge.edge import build_edge_attr
from eris.features.graph_stats import compute_graph_stats, pack_graph_stats
from eris.features.node.residue import build_node_attr
from eris.spatial.contact_graph import neighbor_search_edges

if TYPE_CHECKING:
    from eris.core.representations.base_rep import BaseRepresentation


@dataclass
class ResidueContactGraph(BaseResidueEncoder):
    #Identity
    name: ClassVar[str] = "residue_contact_graph"
    representation: BaseRepresentation = GraphRepresentation()

    #Instance
    include_self_loops: bool = False    
    cutoff: float = 8.0
    node_attr_mode:str = "onehot+physchem+plddt_ca+bfactor_ca"
    edge_attr_mode:str = "dist+rbf+seq+local"
    rbf_k:int = 16
    rbf_dmax:float = 20.0
    local_cutoff:float = 4
    

    def _create_data(self, ctx: EncodingContext) -> dict[str, Any]:
        strc = ctx.raw_structure
        centroid = ctx.centroid
        pos_out = ctx.pos

        n = int(pos_out.shape[0])

        edge_index, edge_dist = neighbor_search_edges(pos_out, cutoff=self.cutoff)

        seq_pos = np.arange(n, dtype=np.int64)

        
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

        stats = compute_graph_stats(edge_index=edge_index, edge_dist=edge_dist, n_nodes=n)
        graph_attr, graph_attr_keys = pack_graph_stats(stats)

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
        
        data: dict[str, Any] = {
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
    

    def _update_params_metadata(
        self,
        meta: dict[str, Any],
        ctx: EncodingContext,
    ) -> dict[str, Any]:

        meta["params"].update({
            "cutoff": float(self.cutoff),
            "include_self_loops": bool(self.include_self_loops),
            "node_attr_mode": self.node_attr_mode,
            "edge_attr_mode": self.edge_attr_mode,
            "rbf_k": int(self.rbf_k),
            "rbf_dmax": float(self.rbf_dmax),
            "local_cutoff": int(self.local_cutoff),}
        )

        return meta
