from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from eris.core.base_encoder import BaseResidueEncoder, EncodingContext
from eris.core.config import ResidueContactGraphConfig
from eris.core.io import load_and_extract_residue
from eris.core.representations import GraphRepresentation
from eris.core.representations.output_declarations import RESIDUE_GRAPH_OUTPUTS
from eris.features.edge.edge import build_edge_attr
from eris.features.graph_stats import compute_graph_stats, pack_graph_stats
from eris.features.node.residue import build_node_attr
from eris.spatial.contact_graph import neighbor_search_edges

if TYPE_CHECKING:
    from eris.core.cache import DiskCache
    from eris.core.representations.base_rep import BaseRepresentation


@dataclass(init=False)
class ResidueContactGraph(BaseResidueEncoder):
    # Identity
    name: ClassVar[str] = "residue_contact_graph"
    representation: ClassVar[BaseRepresentation] = GraphRepresentation()
    output_declarations_v2: ClassVar = RESIDUE_GRAPH_OUTPUTS

    config: ResidueContactGraphConfig = field(init=False)

    def __init__(
        self,
        *,
        config: ResidueContactGraphConfig | None = None,
        use_cache: bool = True,
        cache: DiskCache | None = None,
        **representation_kwargs: Any,
    ) -> None:
        if config is not None:
            if not isinstance(config, ResidueContactGraphConfig):
                raise TypeError("config must be a ResidueContactGraphConfig")
            if representation_kwargs:
                names = ", ".join(sorted(representation_kwargs))
                raise TypeError(f"config cannot be combined with representation kwargs: {names}")
            resolved_config = config
        else:
            resolved_config = ResidueContactGraphConfig(**representation_kwargs)

        self.config = resolved_config
        self.use_cache = use_cache
        self.cache = cache
        super().__post_init__()

    @property
    def center(self) -> bool:
        return self.config.center

    @property
    def scale(self) -> float | None:
        return self.config.scale

    @property
    def residue_center(self) -> str:
        return self.config.residue_center

    @property
    def amino_acids_only(self) -> bool:
        return self.config.amino_acids_only

    @property
    def altloc(self) -> str:
        return self.config.altloc

    @property
    def missing_residue_center(self) -> str:
        return self.config.missing_residue_center

    @property
    def cutoff(self) -> float:
        return self.config.cutoff

    @property
    def include_self_loops(self) -> bool:
        return self.config.include_self_loops

    @property
    def node_attr_mode(self) -> str:
        return "+".join(token.name for token in self.config.node_attr_mode.tokens)

    @property
    def edge_attr_mode(self) -> str:
        return "+".join(token.name for token in self.config.edge_attr_mode.tokens)

    @property
    def rbf_k(self) -> int:
        return self.config.rbf_k

    @property
    def rbf_dmax(self) -> float:
        return self.config.rbf_dmax

    @property
    def local_cutoff(self) -> int:
        return self.config.local_cutoff

    def get_params(self) -> dict[str, Any]:
        return self.config.to_dict()

    def _load_structure(self, ctx: EncodingContext) -> dict[str, Any]:
        return load_and_extract_residue(
            ctx.inp,
            amino_acids_only=self.config.amino_acids_only,
            mode=self.config.residue_center,
            altloc=self.config.altloc,
            missing_ca=self.config.missing_residue_center,
        )

    def _create_data(self, ctx: EncodingContext) -> dict[str, Any]:
        strc = ctx.raw_structure
        centroid = ctx.centroid
        pos_out = ctx.pos

        n = int(pos_out.shape[0])

        edge_index, edge_dist = neighbor_search_edges(
            pos_out,
            cutoff=self.config.cutoff,
        )

        seq_pos = np.arange(n, dtype=np.int64)

        node_attr, node_attr_keys = build_node_attr(
            strc["aa_index"],
            plddt=strc["plddt"],
            bfactor_ca=strc["bfactor_ca"],
            mode=self.config.node_attr_mode.to_list(),
            return_keys=True,
        )

        if self.config.include_self_loops:
            loops = np.arange(n, dtype=np.int64)
            loop_edges = np.stack([loops, loops], axis=0)
            loop_dist = np.zeros((n,), dtype=np.float32)

            edge_index = np.concatenate([edge_index, loop_edges], axis=1)
            edge_dist = np.concatenate([edge_dist, loop_dist], axis=0)

        edge_attr, edge_attr_keys = build_edge_attr(
            edge_index=edge_index,
            edge_dist=edge_dist,
            mode=self.config.edge_attr_mode.to_list(),
            rbf_k=self.config.rbf_k,
            rbf_dmax=self.config.rbf_dmax,
            seq_pos=seq_pos,
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

        meta["params"] = self.config.to_dict()
        return meta

    def _finalize_metadata(self, meta: dict[str, Any]) -> dict[str, Any]:
        meta = super()._finalize_metadata(meta)
        meta["params"] = self.config.to_dict()
        return meta
