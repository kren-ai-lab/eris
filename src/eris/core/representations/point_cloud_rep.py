from typing import Any, ClassVar

from eris.core.artifact_schema_v2 import build_counts, build_feature_keys
from eris.core.base_encoder import EncodingContext
from eris.core.representations.base_rep import BaseRepresentation
from eris.core.contracts.point_cloud_contract import PointCloudContract


class PointCloudRepresentation(BaseRepresentation):
    representation_type: ClassVar[str] = "cloud"

    def contracts(self):
        return [PointCloudContract()]

    def feature_keys_v2(
        self,
        data: dict[str, Any],
        ctx: EncodingContext,
    ) -> dict[str, list[str]]:
        if "node_attr" not in data:
            return build_feature_keys()
        return build_feature_keys(node_attr_keys=ctx.runtime["node_attr_keys"])

    def counts_v2(self, data: dict[str, Any]) -> dict[str, int]:
        counts = {"n_nodes": int(data["pos"].shape[0])}
        if "omitted_residue_ids" in data:
            counts["n_omitted_residues"] = int(data["omitted_residue_ids"].shape[0])
        return build_counts(counts)
