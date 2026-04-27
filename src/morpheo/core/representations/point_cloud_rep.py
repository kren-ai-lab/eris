from typing import Any, Dict, ClassVar

from morpheo.core.representations.base_rep import BaseRepresentation
from morpheo.core.contracts.point_cloud_contract import PointCloudContract
from morpheo.core.encoder_metadata import build_attr_keys, NODE_ATTR_KEY_MAP


class PointCloudRepresentation(BaseRepresentation):
    representation_type: ClassVar[str] = "point_cloud"

    def contracts(self):
        return [PointCloudContract()]
    
    def feature_keys(self, data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
        keys = {}
        params = meta.get("params", {})

        # Node feature keys
        if "node_attr" in data:
            mode = params.get("node_attr_mode")
            if mode is not None:
                keys["node_attr_keys"] = build_attr_keys(
                    mode, NODE_ATTR_KEY_MAP
                )