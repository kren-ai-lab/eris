from typing import Any, Dict, ClassVar

from eris.core.contracts.base_contract import EncoderContract
from eris.core.contracts.validations import validate_centroid, validate_residue_data, validate_grid


class VoxelGridContract(EncoderContract):
    """ """

    contract_name: ClassVar[str] = "grid"

    def validate(self, data: Dict[str, Any]):
        validate_grid(data)
        validate_centroid(data)
        validate_residue_data(data)
