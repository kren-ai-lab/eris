"""Matrix representation type."""

from typing import Any, ClassVar

from eris.core.artifact_schema import build_counts, build_feature_keys
from eris.core.base_encoder import EncodingContext


class MatrixRepresentation:
    """Residue NxN matrix; ``primary_key`` names the matrix output."""

    representation_type: ClassVar[str] = "matrix"

    def __init__(self, primary_key: str) -> None:
        """Set the name of the matrix output."""
        self.primary_key = primary_key

    def feature_keys(
        self,
        _data: dict[str, Any],
        _ctx: EncodingContext,
    ) -> dict[str, list[str]]:
        """Return empty feature keys."""
        return build_feature_keys()

    def counts(self, data: dict[str, Any]) -> dict[str, int]:
        """Return node and omitted-residue counts."""
        return build_counts(
            {
                "n_nodes": int(data[self.primary_key].shape[0]),
                "n_omitted_residues": int(data["omitted_residue_ids"].shape[0]),
            }
        )
