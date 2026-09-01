import numpy as np

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, ClassVar
from typing import TYPE_CHECKING

from eris.core.base_encoder.base_encoder import BaseEncoder, EncodingContext
from eris.core.contracts.granularity import GranularityContract
from eris.core.contracts import AtomContract
from eris.core.io import load_and_extract_atoms


@dataclass
class BaseAtomEncoder(BaseEncoder):
    granularity_contract: ClassVar[GranularityContract] = AtomContract()

    include_hetero: bool = False
    include_h: bool = False
    drop_water: bool = True
    keep_ions: bool = True
    altloc: str = "first"
    amino_acids_only: bool = False

    def _load_structure(self, ctx: EncodingContext):
        strc = load_and_extract_atoms(
            ctx.inp,
            include_hetero=self.include_hetero,
            include_h=self.include_h,
            drop_water=self.drop_water,
            keep_ions=self.keep_ions,
            altloc=self.altloc,
            amino_acids_only=self.amino_acids_only,
        )
        return strc

    def _update_granularity_metadata(
        self, meta: Dict[str, Any], ctx: EncodingContext
    ) -> Dict[str, Any]:
        meta["params"].update(
            {
                "include_hetero": bool(self.include_hetero),
                "include_h": bool(self.include_h),
                "drop_water": bool(self.drop_water),
                "keep_ions": bool(self.keep_ions),
                "altloc": str(self.altloc),
                "amino_acids_only": bool(self.amino_acids_only),
            }
        )
        return meta
