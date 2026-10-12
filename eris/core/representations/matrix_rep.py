from typing import Any, ClassVar

from eris.core.artifact_schema import build_counts, build_feature_keys
from eris.core.base_encoder import EncodingContext


class MatrixRepresentation:
    """Residue NxN matrix; ``primary_key`` names the matrix output."""

    representation_type: ClassVar[str] = "matrix"

    def __init__(self, primary_key: str) -> None:
        self.primary_key = primary_key

    def feature_keys(
        self,
        data: dict[str, Any],
        ctx: EncodingContext,
    ) -> dict[str, list[str]]:
        return build_feature_keys()

    def counts(self, data: dict[str, Any]) -> dict[str, int]:
        return build_counts(
            {
                "n_nodes": int(data[self.primary_key].shape[0]),
                "n_omitted_residues": int(data["omitted_residue_ids"].shape[0]),
            }
        )
