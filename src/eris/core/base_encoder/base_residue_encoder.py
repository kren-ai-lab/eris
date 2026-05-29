import numpy as np

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, ClassVar
from typing import TYPE_CHECKING

from eris.core.base_encoder.base_encoder import BaseEncoder, EncodingContext
from eris.core.contracts.granularity import GranularityContract
from eris.core.contracts import ResidueContract
from eris.core.io import load_and_extract_residue


@dataclass
class BaseResidueEncoder(BaseEncoder):
    granularity_contract: ClassVar[GranularityContract] = ResidueContract()
    residue_center: str = "ca" # ca: aplha carbon - mean: mean between residue atoms - com: center of mass from residue atoms

    def _load_structure(self, ctx: EncodingContext):
        mode = self.residue_center
        strc = load_and_extract_residue(ctx.inp, mode=mode)

        return strc
    
    
    def _update_granularity_metadata(self, meta: Dict[str, Any], ctx: EncodingContext) -> Dict[str, Any]:
        meta.update({
            "residue_center": str(self.residue_center),}
        )
        return meta
