from typing import Any, Dict, ClassVar

from eris.core.base_encoder import EncodingContext
from eris.core.representations.base_rep import BaseRepresentation
from eris.core.contracts.conctact_map_contract import ContactMapContract

class ContactMapRepresentation(BaseRepresentation):
    representation_type: ClassVar[str] = "matrix"

    def contracts(self):
        return [ContactMapContract()]
    

    def feature_keys(self, data: Dict[str, Any], meta: Dict[str, Any], ctx: EncodingContext) -> Dict[str, Any]:
        keys = {}

        keys["matrix_key"] = "contact_bins"
        keys["matrix_attr_keys"] = ["contact_bins"]

        return keys


class ContactMapBinsRepresentation(BaseRepresentation):
    representation_type: ClassVar[str] = "matrix"

    def contracts(self):
        return [ContactMapContract()]
    

    def feature_keys(self, data: Dict[str, Any], meta: Dict[str, Any], ctx: EncodingContext) -> Dict[str, Any]:
        keys = {}

        keys["matrix_key"] = "contact_bins"
        keys["matrix_attr_keys"] = ["contact_bins", "bin_edges"]

        return keys