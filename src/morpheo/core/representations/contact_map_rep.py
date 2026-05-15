from typing import Any, Dict, ClassVar

from morpheo.core.representations.base_rep import BaseRepresentation
from morpheo.core.contracts.conctact_map_contract import ContactMapContract

class ContactMapRepresentation(BaseRepresentation):
    representation_type: ClassVar[str] = "matrix"

    def contracts(self):
        return [ContactMapContract()]
    

    def feature_keys(self, data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
        keys = {}

        keys["matrix_key"] = "contact_bins"
        keys["matrix_attr_keys"] = ["contact_bins"]

        return keys


class ContactMapBinsRepresentation(BaseRepresentation):
    representation_type: ClassVar[str] = "matrix"

    def contracts(self):
        return [ContactMapContract()]
    

    def feature_keys(self, data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
        keys = {}

        keys["matrix_key"] = "contact_bins"
        keys["matrix_attr_keys"] = ["contact_bins", "bin_edges"]

        return keys