from typing import Any, Dict, ClassVar

from eris.core.contracts.base_contract import EncoderContract
from eris.core.contracts.validations import (
    validate_pos,
    validate_edge_index,
    validate_node_feature,
    validate_edge_features,
    validate_centroid,
    validate_residue_data,
)


class GraphContract(EncoderContract):
    """ """

    contract_name: ClassVar[str] = "graph"

    def validate(self, data: Dict[str, Any]):

        validate_pos(data)
        validate_edge_index(data)
        validate_node_feature(data, "node_attr")
        validate_edge_features(data, "edge_attr")
        validate_centroid(data)
        validate_residue_data(data)
