from typing import Any, Dict, ClassVar

from morpheo.core.contracts.base_contract import EncoderContract
from morpheo.core.contracts._geometric import validate_pos, validate_node_feature

class PointCloudContract(EncoderContract):
    """
    """
    contract_name: ClassVar[str] = "point_cloud"

    def validate(self, data: Dict[str, Any]):

        validate_pos(data)
        validate_node_feature(data, "node_attr")