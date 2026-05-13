from typing import Any, Dict, ClassVar

from morpheo.core.representations.base_rep import BaseRepresentation
from morpheo.core.contracts.distogram_contract import DistogramContract


class DistogramRepresentation(BaseRepresentation):
    representation_type: ClassVar[str] = "distogram"

    def contracts(self):
        return [DistogramContract()]
    

    def feature_keys(self, data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
        keys = {}

        keys["matrix_key"] = "dist"
        keys["matrix_attr_keys"] = ["dist"]

        return keys
    

class DistogramBinsRepresentation(BaseRepresentation):
    representation_type: ClassVar[str] = "distogram"

    def contracts(self):
        return [DistogramContract()]
    

    def feature_keys(self, data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
        keys = {}

        keys["matrix_key"] = "dist_bins"
        keys["matrix_attr_keys"] = ["dist_bins", "bin_edges"]

        return keys
