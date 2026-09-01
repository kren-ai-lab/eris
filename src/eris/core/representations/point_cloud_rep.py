from typing import Any, Dict, ClassVar

from eris.core.artifact_schema_v2 import build_counts, build_feature_keys
from eris.core.base_encoder import EncodingContext
from eris.core.representations.base_rep import BaseRepresentation
from eris.core.contracts.point_cloud_contract import PointCloudContract


class PointCloudRepresentation(BaseRepresentation):
    representation_type: ClassVar[str] = "cloud"

    def contracts(self):
        return [PointCloudContract()]

    def feature_keys(
        self, data: Dict[str, Any], meta: Dict[str, Any], ctx: EncodingContext
    ) -> Dict[str, Any]:
        keys = {}

        # Node feature keys
        if "node_attr" in data:
            keys["node_attr_keys"] = ctx.runtime["node_attr_keys"]

        return keys

    def feature_keys_v2(
        self,
        data: dict[str, Any],
        ctx: EncodingContext,
    ) -> dict[str, list[str]]:
        if "node_attr" not in data:
            return build_feature_keys()
        return build_feature_keys(node_attr_keys=ctx.runtime["node_attr_keys"])

    def counts_v2(self, data: dict[str, Any]) -> dict[str, int]:
        return build_counts({"n_nodes": int(data["pos"].shape[0])})
