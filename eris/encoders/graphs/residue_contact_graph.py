from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from eris.core.base_encoder import BaseResidueEncoder, EncodingContext
from eris.core.config import ResidueContactGraphConfig
from eris.core.representations import GraphRepresentation
from eris.core.representations.output_declarations import (
    RESIDUE_GRAPH_OUTPUTS,
    residue_graph_output_declarations,
)
from eris.features.edge.edge import build_edge_attr
from eris.features.graph_stats import compute_graph_stats, pack_graph_stats
from eris.features.interface import (
    compute_interface_contacts,
    interface_edge_flags,
    resolve_interface_cutoff,
)
from eris.features.node.residue import build_node_attr
from eris.spatial.contact_graph import neighbor_search_edges


class ResidueContactGraph(BaseResidueEncoder):
    # Identity
    name: ClassVar[str] = "residue_contact_graph"
    version: ClassVar[str] = "0.1.4"
    representation: ClassVar = GraphRepresentation()
    output_declarations_v2: ClassVar = RESIDUE_GRAPH_OUTPUTS

    config_type: ClassVar = ResidueContactGraphConfig
    config: ResidueContactGraphConfig

    def _create_data(self, ctx: EncodingContext) -> dict[str, Any]:
        strc = ctx.raw_structure
        centroid = ctx.centroid
        pos_out = ctx.pos

        n = int(pos_out.shape[0])

        edge_index, edge_dist = neighbor_search_edges(
            pos_out,
            cutoff=self.config.cutoff,
        )

        interface_cutoff = resolve_interface_cutoff(
            self.config.node_attr_mode.to_list(),
            self.config.edge_attr_mode.to_list(),
        )
        interface_flags = None
        interface_pairs = None
        if interface_cutoff is not None:
            interface_flags, interface_pairs = compute_interface_contacts(
                strc["polymer_heavy_atom_coordinates"],
                strc["polymer_heavy_atom_residue_index"],
                strc["chain_id"],
                cutoff=interface_cutoff,
            )

        seq_pos = np.asarray(strc["biological_sequence_positions"], dtype=np.int64)

        node_attr, node_attr_keys = build_node_attr(
            strc["aa_index"],
            plddt_ca=strc["plddt"],
            bfactor_ca=strc["bfactor_ca"],
            residue_ids=strc["residue_ids"],
            chain_id=strc["chain_id"],
            chain_ids=strc.chain_ids,
            model_index=strc.model_index,
            structural_residue_index=strc["structural_residue_index"],
            backbone_n=strc["backbone_n"],
            backbone_ca=strc["backbone_ca"],
            backbone_c=strc["backbone_c"],
            backbone_n_mask=strc["backbone_n_mask"],
            backbone_ca_mask=strc["backbone_ca_mask"],
            backbone_c_mask=strc["backbone_c_mask"],
            resname=strc["resname"],
            polymer_heavy_atom_coordinates=strc["polymer_heavy_atom_coordinates"],
            polymer_heavy_atom_residue_index=strc["polymer_heavy_atom_residue_index"],
            polymer_heavy_atom_name=strc["polymer_heavy_atom_name"],
            polymer_heavy_atom_element=strc["polymer_heavy_atom_element"],
            interface_flags=interface_flags,
            mode=self.config.node_attr_mode.to_list(),
            return_keys=True,
        )

        if self.config.include_self_loops:
            loops = np.arange(n, dtype=np.int64)
            loop_edges = np.stack([loops, loops], axis=0)
            loop_dist = np.zeros((n,), dtype=np.float32)

            edge_index = np.concatenate([edge_index, loop_edges], axis=1)
            edge_dist = np.concatenate([edge_dist, loop_dist], axis=0)

        interface_edges = (
            interface_edge_flags(edge_index, interface_pairs)
            if interface_pairs is not None
            else None
        )

        edge_attr, edge_attr_keys = build_edge_attr(
            edge_index=edge_index,
            edge_dist=edge_dist,
            mode=self.config.edge_attr_mode.to_list(),
            rbf_k=self.config.rbf_k,
            rbf_dmax=self.config.rbf_dmax,
            seq_pos=seq_pos,
            chain_id=strc["chain_id"],
            backbone_n=strc["backbone_n"],
            backbone_ca=strc["backbone_ca"],
            backbone_c=strc["backbone_c"],
            backbone_n_mask=strc["backbone_n_mask"],
            backbone_ca_mask=strc["backbone_ca_mask"],
            backbone_c_mask=strc["backbone_c_mask"],
            interface_contact=interface_edges,
            local_cutoff=self.config.local_cutoff,
            return_keys=True,
        )

        stats = compute_graph_stats(edge_index=edge_index, edge_dist=edge_dist, n_nodes=n)
        graph_attr, graph_attr_keys = pack_graph_stats(stats)

        ctx.runtime["node_attr_keys"] = node_attr_keys
        ctx.runtime["edge_attr_keys"] = edge_attr_keys
        ctx.runtime["graph_attr_keys"] = graph_attr_keys

        data: dict[str, Any] = {
            "pos": pos_out.astype(np.float32),
            "centroid": centroid.astype(np.float32),
            "residue_ids": strc["residue_ids"].astype(np.str_),
            "chain_id": strc["chain_id"].astype(np.str_),
            "omitted_residue_ids": np.asarray(strc.get("omitted_residue_ids", []), dtype=np.str_),
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
        if interface_flags is not None:
            data["interface_residue_ids"] = strc["residue_ids"][interface_flags].astype(np.str_)

        return data

    def _output_declarations_v2(self, data: dict[str, np.ndarray]):
        return residue_graph_output_declarations(data)
