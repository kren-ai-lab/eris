"""Base class for residue-level encoders."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast

from eris.core.base_encoder.base_encoder import BaseEncoder
from eris.core.io import load_and_extract_residue

if TYPE_CHECKING:
    from eris.core.base_encoder.base_encoder import EncodingContext
    from eris.core.config import _ResidueSelectionConfig
    from eris.core.io import LoadedStructureData


class BaseResidueEncoder(BaseEncoder):
    """Base class for encoders that operate on residues."""

    granularity: ClassVar[str] = "residue"

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
        from eris.features.node.external import verify_external_params  # noqa: PLC0415

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

    def _load_structure(self, ctx: EncodingContext) -> LoadedStructureData:
        config = cast("_ResidueSelectionConfig", self.config)
        return load_and_extract_residue(
            ctx.inp,
            amino_acids_only=config.amino_acids_only,
            mode=config.residue_center,
            altloc=config.altloc,
            missing_ca=config.missing_residue_center,
            plddt_source=config.plddt_source,
            structure=ctx.structure,
        )
