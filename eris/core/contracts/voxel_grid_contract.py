from typing import Any, ClassVar, Dict

from eris.core.contracts.base_contract import EncoderContract
from eris.core.contracts.validations import validate_centroid, validate_grid, validate_residue_data


class VoxelGridContract(EncoderContract):
    """ """

    contract_name: ClassVar[str] = "grid"

    def validate(self, data: Dict[str, Any]):
        validate_grid(data)
        validate_centroid(data)
        validate_residue_data(data)
