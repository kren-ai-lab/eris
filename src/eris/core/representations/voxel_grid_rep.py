from typing import Any, Dict, ClassVar

from eris.core.base_encoder import EncodingContext
from eris.core.representations.base_rep import BaseRepresentation
from eris.core.contracts.voxel_grid_contract import VoxelGridContract

class VoxelGridRepresentation(BaseRepresentation):
    representation_type: ClassVar[str] = "grid"

    def contracts(self):
        return [VoxelGridContract()]
    

    def feature_keys(self, data: Dict[str, Any], meta: Dict[str, Any], ctx: EncodingContext) -> Dict[str, Any]:
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