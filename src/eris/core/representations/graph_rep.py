from typing import Any, Dict, ClassVar

from eris.core.base_encoder import EncodingContext
from eris.core.representations.base_rep import BaseRepresentation
from eris.core.contracts.graph_contract import GraphContract


class GraphRepresentation(BaseRepresentation):
    representation_type: ClassVar[str] = "graph"

    def contracts(self):
        return [GraphContract()]

    def feature_keys(self, data: Dict[str, Any], meta: Dict[str, Any], ctx: EncodingContext) -> Dict[str, Any]:
        keys = {}
        params = meta.get("params", {})

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
