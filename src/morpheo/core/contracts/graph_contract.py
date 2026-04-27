from typing import Any, Dict, ClassVar

from morpheo.core.contracts.base_contract import EncoderContract
from morpheo.core.contracts._geometric import validate_pos, validate_edge_index, validate_node_feature, validate_edge_features

class GraphContract(EncoderContract):
    """
    """
    contract_name: ClassVar[str] = "graph"

    def validate(self, data: Dict[str, Any]):

        validate_pos(data)
        validate_edge_index(data)
        validate_node_feature(data, "node_attr")
        validate_edge_features(data, "edge_attr")

