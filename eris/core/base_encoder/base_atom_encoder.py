from typing import ClassVar

from eris.core.base_encoder.base_encoder import BaseEncoder, EncodingContext
from eris.core.io import load_and_extract_atoms


class BaseAtomEncoder(BaseEncoder):
    granularity: ClassVar[str] = "atom"

    def _load_structure(self, ctx: EncodingContext):
        strc = load_and_extract_atoms(
            ctx.inp,
            include_hetero=self.config.include_hetero,
            include_h=self.config.include_h,
            drop_water=self.config.drop_water,
            keep_ions=self.config.keep_ions,
            altloc=self.config.altloc,
            amino_acids_only=self.config.amino_acids_only,
            structure=ctx.structure,
        )
        return strc
