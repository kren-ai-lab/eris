from typing import Any, ClassVar

from eris.core.artifact_schema import build_counts, build_feature_keys
from eris.core.base_encoder import EncodingContext


class VoxelGridRepresentation:
    representation_type: ClassVar[str] = "grid"

    def feature_keys(
        self,
        data: dict[str, Any],
        ctx: EncodingContext,
    ) -> dict[str, list[str]]:
        return build_feature_keys(grid_channels=ctx.runtime["channels"])

    def counts(self, data: dict[str, Any]) -> dict[str, int]:
        return build_counts({"n_omitted_residues": int(data["omitted_residue_ids"].shape[0])})
