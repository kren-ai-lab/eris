from typing import Any, Dict, ClassVar

from eris.core.artifact_schema_v2 import build_counts, build_feature_keys
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
            
        if "channels" in ctx.runtime:
            keys["grid_channels"] = ctx.runtime["channels"]
        
        return keys

    def feature_keys_v2(
        self,
        data: dict[str, Any],
        ctx: EncodingContext,
    ) -> dict[str, list[str]]:
        return build_feature_keys(grid_channels=ctx.runtime["channels"])

    def counts_v2(self, data: dict[str, Any]) -> dict[str, int]:
        return build_counts({})
