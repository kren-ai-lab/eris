from typing import Any, Dict, ClassVar

from eris.core.artifact_schema_v2 import build_counts, build_feature_keys
from eris.core.base_encoder import EncodingContext
from eris.core.representations.base_rep import BaseRepresentation
from eris.core.contracts.distogram_contract import DistogramContract


class DistogramRepresentation(BaseRepresentation):
    representation_type: ClassVar[str] = "matrix"

    def contracts(self):
        return [DistogramContract()]

    def feature_keys(
        self, data: Dict[str, Any], meta: Dict[str, Any], ctx: EncodingContext
    ) -> Dict[str, Any]:
        keys = {}

        keys["matrix_key"] = "dist"
        keys["matrix_attr_keys"] = ["dist"]

        return keys

    def feature_keys_v2(
        self,
        data: dict[str, Any],
        ctx: EncodingContext,
    ) -> dict[str, list[str]]:
        return build_feature_keys()

    def counts_v2(self, data: dict[str, Any]) -> dict[str, int]:
        return build_counts(
            {
                "n_nodes": int(data["dist"].shape[0]),
                "n_omitted_residues": int(data["omitted_residue_ids"].shape[0]),
            }
        )


class DistogramBinsRepresentation(BaseRepresentation):
    representation_type: ClassVar[str] = "matrix"

    def contracts(self):
        return [DistogramContract()]

    def feature_keys(
        self, data: Dict[str, Any], meta: Dict[str, Any], ctx: EncodingContext
    ) -> Dict[str, Any]:
        keys = {}

        keys["matrix_key"] = "dist_bins"
        keys["matrix_attr_keys"] = ["dist_bins", "bin_edges"]

        return keys

    def feature_keys_v2(
        self,
        data: dict[str, Any],
        ctx: EncodingContext,
    ) -> dict[str, list[str]]:
        return build_feature_keys()

    def counts_v2(self, data: dict[str, Any]) -> dict[str, int]:
        return build_counts(
            {
                "n_nodes": int(data["dist_bins"].shape[0]),
                "n_omitted_residues": int(data["omitted_residue_ids"].shape[0]),
            }
        )
