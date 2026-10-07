from dataclasses import dataclass
from typing import Any, ClassVar

from eris.core.base_encoder.base_encoder import BaseEncoder, EncodingContext
from eris.core.contracts.granularity import GranularityContract
from eris.core.contracts import ResidueContract
from eris.core.io import load_and_extract_residue


@dataclass
class BaseResidueEncoder(BaseEncoder):
    granularity_contract: ClassVar[GranularityContract] = ResidueContract()
    residue_center: str = "ca"  # ca: alpha carbon - mean: mean of residue atoms - com: center of mass of residue atoms

    @staticmethod
    def _external_parameter_blocks(
        parameters: dict[str, Any],
    ) -> tuple[dict[str, Any], ...]:
        node_attr_mode = parameters.get("node_attr_mode")
        if not isinstance(node_attr_mode, list):
            return ()
        return tuple(
            token["external"]
            for token in node_attr_mode
            if isinstance(token, dict) and set(token) == {"external"}
        )

    def _validate_runtime_config_dependencies(self, parameters: dict[str, Any]) -> None:
        descriptors = self._external_parameter_blocks(parameters)
        if not descriptors:
            return
        from eris.features.node.external import verify_external_params

        for descriptor in descriptors:
            verify_external_params(descriptor)

    def _artifact_extensions(
        self,
        parameters: dict[str, Any],
    ) -> dict[str, dict[str, Any]] | None:
        invocations = [
            {
                "content_sha256": descriptor["content_sha256"],
                "path": descriptor["path"],
            }
            for descriptor in self._external_parameter_blocks(parameters)
        ]
        if not invocations:
            return None
        return {"org.eris.external_features": {"invocations": invocations}}

    def _load_structure(self, ctx: EncodingContext):
        mode = self.residue_center
        strc = load_and_extract_residue(ctx.inp, mode=mode)

        return strc
