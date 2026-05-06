from typing import Any, Dict, ClassVar

from morpheo.core.representations.base_rep import BaseRepresentation
from morpheo.core.contracts.point_cloud_contract import PointCloudContract
from morpheo.core.encoder_metadata import build_attr_keys, NODE_ATTR_KEY_MAP


class VoxelGridRepresentation(BaseRepresentation):
    representation_type: ClassVar[str] = "voxel_grid"

    def contracts(self):
        return []
    

    def feature_keys(self, data: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
        keys = {}

        keys["grid_key"] = "grid"
        keys["grid_layout"] = "CXYZ"

        # grid channels
        if "channels" in data:
            ch_list = data.get("channels")
            expanded_channels = []
            for c in ch_list:
                if c == "aa_onehot":
                    expanded_channels.extend([f"aa_onehot_{i}" for i in range(21)])
                else:
                    expanded_channels.append(c)
            
            keys["grid_channels"] = expanded_channels

        
        return keys