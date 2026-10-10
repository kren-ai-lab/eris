from typing import Any, ClassVar, Dict

from eris.core.contracts.base_contract import EncoderContract
from eris.core.contracts.validations import (
    validate_centroid,
    validate_node_feature,
    validate_pos,
    validate_residue_data,
)


class PointCloudContract(EncoderContract):
    """ """

    contract_name: ClassVar[str] = "point_cloud"

    def validate(self, data: Dict[str, Any]):

        validate_pos(data)
        validate_node_feature(data, "node_attr")
        validate_centroid(data)
        validate_residue_data(data)
