from typing import Any, Dict, ClassVar

from eris.core.base_encoder import EncodingContext
from eris.core.representations.base_rep import BaseRepresentation
from eris.core.contracts.point_cloud_contract import PointCloudContract


class PointCloudRepresentation(BaseRepresentation):
    representation_type: ClassVar[str] = "cloud"

    def contracts(self):
        return [PointCloudContract()]
    
    def feature_keys(self, data: Dict[str, Any], meta: Dict[str, Any], ctx: EncodingContext) -> Dict[str, Any]:
        keys = {}
        params = meta.get("params", {})

        # Node feature keys
        if "node_attr" in data:
            keys["node_attr_keys"] = ctx.runtime["node_attr_keys"]

        return keys