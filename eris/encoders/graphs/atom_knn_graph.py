from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from eris.core.base_encoder import BaseAtomEncoder, EncodingContext
from eris.core.config import AtomKNNGraphConfig
from eris.core.representations import GraphRepresentation
from eris.core.representations.output_declarations import ATOM_GRAPH_OUTPUTS
from eris.features.edge.edge import build_edge_attr
from eris.features.graph_stats import compute_graph_stats, pack_graph_stats
from eris.features.node.atom import build_node_attr
from eris.spatial.knn import knn_edges

if TYPE_CHECKING:
    from eris.core.representations.base_rep import BaseRepresentation


class AtomKNNGraph(BaseAtomEncoder):
    """
    Atom-level kNN graph (CA-independent) with deterministic node/edge features baseline.

    Outputs:
      - pos, edge_index, edge_dist
      - node_attr (fixed dim)
      - edge_attr_mode (dist+rbf+seq+local)
      - graph_attr (+ keys)
      - plus raw per-atom fields: elem_index, atom_role, bfactor, occ, residue_ids, atom_name, etc.
    """

    # Identity
    name: ClassVar[str] = "atom_knn_graph"
    version: ClassVar[str] = "0.1.2"
    representation: ClassVar[BaseRepresentation] = GraphRepresentation()
    output_declarations_v2: ClassVar = ATOM_GRAPH_OUTPUTS

    config_type: ClassVar = AtomKNNGraphConfig
    config: AtomKNNGraphConfig

    def _create_data(self, ctx: EncodingContext) -> dict[str, Any]:
        strc = ctx.raw_structure
        centroid = ctx.centroid
        pos_out = ctx.pos

        pos = pos_out.astype(np.float32)
        n = int(pos.shape[0])

        edge_index, edge_dist = knn_edges(
            pos=pos_out,
            k=self.config.k,
            cutoff=self.config.cutoff,
            symmetric=self.config.symmetric,
        )

        # deterministic baseline attributes
        node_attr, node_attr_keys = build_node_attr(
            elem_index=strc["elem_index"],
            atom_role=strc["atom_role"],
            bfactor=strc["bfactor"],
            occ=strc["occ"],
            mode=self.config.node_attr_mode.to_list(),
            return_keys=True,
        )

        edge_attr, edge_attr_keys = build_edge_attr(
            edge_dist=edge_dist,
            edge_index=edge_index,
            mode=self.config.edge_attr_mode.to_list(),
            seq_pos=strc["seq_pos_residue"],
            chain_id=strc["chain_id"],
            rbf_k=self.config.rbf_k,
            rbf_dmax=self.config.rbf_dmax,
            local_cutoff=self.config.local_cutoff,
            return_keys=True,
        )

        stats = compute_graph_stats(edge_index, edge_dist, n_nodes=n)
        graph_attr, graph_attr_keys = pack_graph_stats(stats)

        ctx.runtime["node_attr_keys"] = node_attr_keys
        ctx.runtime["edge_attr_keys"] = edge_attr_keys
        ctx.runtime["graph_attr_keys"] = graph_attr_keys

        data: dict[str, Any] = {
            "pos": pos.astype(np.float32),
            "centroid": centroid.astype(np.float32),
            "edge_index": edge_index.astype(np.int64),
            "edge_dist": edge_dist.astype(np.float32),
            "node_attr": node_attr.astype(np.float32),
            "edge_attr": edge_attr.astype(np.float32),
            "graph_attr": graph_attr.astype(np.float32),
            # raw atom fields (useful + deterministic)
            "elem_index": strc["elem_index"].astype(np.int16),
            "chain_id": strc["chain_id"].astype(np.str_),
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
