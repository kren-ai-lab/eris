from typing import Any, ClassVar, Dict

from eris.core.contracts.base_contract import EncoderContract
from eris.core.contracts.validations import validate_centroid, validate_pos, validate_residue_data


class ContactMapContract(EncoderContract):
    """ """

    contract_name: ClassVar[str] = "contact_map"

    def validate(self, data: Dict[str, Any]):

        validate_pos(data)
        validate_centroid(data)
        validate_residue_data(data)
