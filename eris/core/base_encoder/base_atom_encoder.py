"""Base class for atom-level encoders."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

from eris.core.base_encoder.base_encoder import BaseEncoder
from eris.core.io import load_and_extract_atoms

if TYPE_CHECKING:
    from eris.core.base_encoder.base_encoder import EncodingContext
    from eris.core.config import _AtomSelectionConfig
    from eris.core.io import LoadedStructureData


class BaseAtomEncoder(BaseEncoder):
    """Base class for encoders that operate on atoms."""

    granularity: ClassVar[str] = "atom"

    def _load_structure(self, ctx: EncodingContext) -> LoadedStructureData:
        config = cast("_AtomSelectionConfig", self.config)
        return load_and_extract_atoms(
            ctx.inp,
            include_hetero=config.include_hetero,
            include_h=config.include_h,
            drop_water=config.drop_water,
            keep_ions=config.keep_ions,
            altloc=config.altloc,
            amino_acids_only=config.amino_acids_only,
            structure=ctx.structure,
        )
