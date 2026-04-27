from typing import Any, Dict, ClassVar

from morpheo.core.representations.base_rep import BaseRepresentation
from morpheo.core.contracts.graph_contract import GraphContract
from morpheo.core.encoder_metadata import build_attr_keys, NODE_ATTR_KEY_MAP, EDGE_ATTR_KEY_MAP


class GraphRepresentation(BaseRepresentation):
    representation_type: ClassVar[str] = "graph"

    def contracts(self):
        return [GraphContract()]

    def feature_keys(self, data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
        keys = {}
        params = meta.get("params", {})

        # Node feature keys
        if "node_attr" in data:
            mode = params.get("node_attr_mode")
            if mode is not None:
                keys["node_attr_keys"] = build_attr_keys(
                    mode, NODE_ATTR_KEY_MAP
                )

        # Edge feature keys
        if "edge_attr" in data:
            mode = params.get("edge_attr_mode")
            if mode is not None:
                keys["edge_attr_keys"] = build_attr_keys(
                    mode, EDGE_ATTR_KEY_MAP
            )
                
        # Graph feature keys
        
        if "graph_attr" in data:
            keys["graph_attr_keys"] = list(
                params.get("graph_attr_keys", [])
            )


        return keys
