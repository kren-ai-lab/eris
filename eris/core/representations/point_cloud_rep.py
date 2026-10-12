"""Point cloud representation type."""

from typing import Any, ClassVar

from eris.core.artifact_schema import build_counts, build_feature_keys
from eris.core.base_encoder import EncodingContext


class PointCloudRepresentation:
    """Point cloud with optional node attributes."""

    representation_type: ClassVar[str] = "cloud"

    def feature_keys(
        self,
        data: dict[str, Any],
        ctx: EncodingContext,
    ) -> dict[str, list[str]]:
        """Return node attribute keys when ``node_attr`` is present."""
        if "node_attr" not in data:
            return build_feature_keys()
        return build_feature_keys(node_attr_keys=ctx.runtime["node_attr_keys"])

    def counts(self, data: dict[str, Any]) -> dict[str, int]:
        """Return node and omitted-residue counts."""
        counts = {"n_nodes": int(data["pos"].shape[0])}
        if "omitted_residue_ids" in data:
            counts["n_omitted_residues"] = int(data["omitted_residue_ids"].shape[0])
        return build_counts(counts)
