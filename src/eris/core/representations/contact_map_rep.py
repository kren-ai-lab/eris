from typing import Any, ClassVar

from eris.core.artifact_schema_v2 import build_counts, build_feature_keys
from eris.core.base_encoder import EncodingContext
from eris.core.representations.base_rep import BaseRepresentation
from eris.core.contracts.conctact_map_contract import ContactMapContract


class ContactMapRepresentation(BaseRepresentation):
    representation_type: ClassVar[str] = "matrix"

    def contracts(self):
        return [ContactMapContract()]

    def feature_keys_v2(
        self,
        data: dict[str, Any],
        ctx: EncodingContext,
    ) -> dict[str, list[str]]:
        return build_feature_keys()

    def counts_v2(self, data: dict[str, Any]) -> dict[str, int]:
        return build_counts(
            {
                "n_nodes": int(data["contact"].shape[0]),
                "n_omitted_residues": int(data["omitted_residue_ids"].shape[0]),
            }
        )


class ContactMapBinsRepresentation(BaseRepresentation):
    representation_type: ClassVar[str] = "matrix"

    def contracts(self):
        return [ContactMapContract()]

    def feature_keys_v2(
        self,
        data: dict[str, Any],
        ctx: EncodingContext,
    ) -> dict[str, list[str]]:
        return build_feature_keys()

    def counts_v2(self, data: dict[str, Any]) -> dict[str, int]:
        return build_counts(
            {
                "n_nodes": int(data["contact_bins"].shape[0]),
                "n_omitted_residues": int(data["omitted_residue_ids"].shape[0]),
            }
        )
