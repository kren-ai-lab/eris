from typing import Any, Dict, ClassVar

from morpheo.core.contracts.base_contract import EncoderContract
from morpheo.core.contracts._geometric import validate_pos

class ContactMapContract(EncoderContract):
    """
    """
    contract_name: ClassVar[str] = "contact_map"

    def validate(self, data: Dict[str, Any]):

        validate_pos(data)
