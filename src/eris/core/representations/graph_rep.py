from typing import Any, Dict, ClassVar

import numpy as np

from eris.core.artifact_schema_v2 import build_counts, build_feature_keys
from eris.core.base_encoder import EncodingContext
from eris.core.representations.base_rep import BaseRepresentation
from eris.core.contracts.graph_contract import GraphContract


class GraphRepresentation(BaseRepresentation):
    representation_type: ClassVar[str] = "graph"

    def contracts(self):
        return [GraphContract()]

    def feature_keys(
        self, data: Dict[str, Any], meta: Dict[str, Any], ctx: EncodingContext
    ) -> Dict[str, Any]:
        keys = {}

        # Node feature keys
        if "node_attr" in data:
            keys["node_attr_keys"] = ctx.runtime["node_attr_keys"]

        # Edge feature keys
        if "edge_attr" in data:
            keys["edge_attr_keys"] = ctx.runtime["edge_attr_keys"]

        # Graph feature keys
        if "graph_attr" in data:
            keys["graph_attr_keys"] = ctx.runtime["graph_attr_keys"]

        return keys

    def feature_keys_v2(
        self,
        data: dict[str, Any],
        ctx: EncodingContext,
    ) -> dict[str, list[str]]:
        return build_feature_keys(
            node_attr_keys=ctx.runtime["node_attr_keys"] if "node_attr" in data else None,
            edge_attr_keys=ctx.runtime["edge_attr_keys"] if "edge_attr" in data else None,
            graph_attr_keys=ctx.runtime["graph_attr_keys"] if "graph_attr" in data else None,
        )

    def counts_v2(self, data: dict[str, Any]) -> dict[str, int]:
        edge_index = data["edge_index"]
        counts = {
            "n_nodes": int(data["pos"].shape[0]),
            "n_edges": int(edge_index.shape[1]),
            "n_self_loops": int(np.count_nonzero(edge_index[0] == edge_index[1])),
        }
        if "omitted_residue_ids" in data:
            counts["n_omitted_residues"] = int(data["omitted_residue_ids"].shape[0])
        return build_counts(counts)
