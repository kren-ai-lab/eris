from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, ClassVar, Tuple
from typing import TYPE_CHECKING

import numpy as np

from morpheo.core.base_encoder import BaseResidueEncoder, EncodingContext
from morpheo.core.representations import GraphRepresentation
from morpheo.features.node.residue import build_node_attr
from morpheo.features.edge.edge import build_edge_attr
from morpheo.features.graph_stats import compute_graph_stats, pack_graph_stats
from morpheo.features.spatial.contact_graph import neighbor_search_edges


if TYPE_CHECKING:
    from morpheo.core.representations.base_rep import BaseRepresentation


@dataclass
class ResidueContactGraph(BaseResidueEncoder):
    #Identity
    name: ClassVar[str] = "residue_contact_graph"
    representation: BaseRepresentation = GraphRepresentation()

    #Instance
    include_self_loops: bool = False    
    cutoff: float = 8.0
    node_attr_mode = "onehot+physchem+plddt+bfactor"
    edge_attr_mode = "dist+rbf+seq+local"
    rbf_k = 16
    rbf_dmax = 20.0
    local_cutoff = 4
    

    def _create_data(self, ctx: EncodingContext) -> Dict[str, Any]:
        strc = ctx.raw_structure
        centroid = ctx.centroid
        pos_out = ctx.pos

        n = int(pos_out.shape[0])

        edge_index, edge_dist = neighbor_search_edges(pos_out, cutoff=self.cutoff)

        stats = compute_graph_stats(edge_index=edge_index, edge_dist=edge_dist, n_nodes=n)
        graph_attr, graph_attr_keys = pack_graph_stats(stats)

        seq_pos = np.arange(n, dtype=np.int64)

        
        node_attr = build_node_attr(
            strc["aa_index"],
            plddt=strc["plddt"],
            bfactor_ca=strc["bfactor_ca"],
            mode=self.node_attr_mode,
        )


        if self.include_self_loops:
            loops = np.arange(n, dtype=np.int64)
            loop_edges = np.stack([loops, loops], axis=0)
            loop_dist = np.zeros((n,), dtype=np.float32)

            edge_index = np.concatenate([edge_index, loop_edges], axis=1)
            edge_dist = np.concatenate([edge_dist, loop_dist], axis=0)

            # NOTE: edge_attr currently computed before loops; if loops are enabled,
            # you may want to recompute edge_attr OR append loop edge_attr rows.
            # Here we leave it as-is to match your original logic.

        edge_attr = build_edge_attr(
            edge_index=edge_index,
            edge_dist=edge_dist,
            mode=self.edge_attr_mode,
            rbf_k=self.rbf_k,
            rbf_dmax=self.rbf_dmax,
            seq_pos=seq_pos,
            local_cutoff=self.local_cutoff,
        )

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
            "graph_attr_keys": np.asarray(graph_attr_keys, dtype=np.str_),
        }

        if strc["plddt"] is not None:
            data["plddt"] = strc["plddt"].astype(np.float32)
        if strc["bfactor_ca"] is not None:
            data["bfactor_ca"] = strc["bfactor_ca"].astype(np.float32)

        return data
    

    def _update_params_metadata(self, meta: Dict[str, Any], ctx: EncodingContext) -> Dict[str, Any]:

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