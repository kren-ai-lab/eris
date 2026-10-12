"""Voxel grid representation type."""

from typing import Any, ClassVar

from eris.core.artifact_schema import build_counts, build_feature_keys
from eris.core.base_encoder import EncodingContext


class VoxelGridRepresentation:
    """Voxel grid with named channels."""

    representation_type: ClassVar[str] = "grid"

    def feature_keys(
        self,
        _data: dict[str, Any],
        ctx: EncodingContext,
    ) -> dict[str, list[str]]:
        """Return the grid channel names."""
        return build_feature_keys(grid_channels=ctx.runtime["channels"])

    def counts(self, data: dict[str, Any]) -> dict[str, int]:
        """Return the omitted-residue count."""
        return build_counts({"n_omitted_residues": int(data["omitted_residue_ids"].shape[0])})
