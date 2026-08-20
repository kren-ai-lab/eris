from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, ClassVar
from typing import TYPE_CHECKING
from typing import Any, Dict, Optional, Sequence

import numpy as np

from eris.core.base_encoder import BaseResidueEncoder, EncodingContext
from eris.core.representations import DistogramRepresentation

if TYPE_CHECKING:
    from eris.core.representations.base_rep import BaseRepresentation


@dataclass
class DistogramCA(BaseResidueEncoder):
    #Identity
    name: ClassVar[str] = "residue_distogram"
    representation: ClassVar[BaseRepresentation] = DistogramRepresentation()

    #Instance  
    include_self: bool = True

    def _create_data(self, ctx: EncodingContext) -> Dict[str, Any]:
        strc = ctx.raw_structure
        centroid = ctx.centroid
        pos_out = ctx.pos

        n = int(pos_out.shape[0])

        diff = pos_out[:, None, :] - pos_out[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)

        if not self.include_self:
            np.fill_diagonal(dist, np.nan)

        data: Dict[str, Any] = {
            "dist": dist.astype(np.float32),
            "pos": pos_out.astype(np.float32),
            "centroid": centroid.astype(np.float32),
            "residue_ids": strc["residue_ids"].astype(np.str_),
            "resname": strc["resname"].astype(np.str_),
            "aa_index": strc["aa_index"].astype(np.int16),
            "plddt": strc["plddt"].astype(np.float32),
            "bfactor_ca": strc["bfactor_ca"].astype(np.float32),
        }

        return data
    

    def _update_params_metadata(self, meta: Dict[str, Any], ctx: EncodingContext) -> Dict[str, Any]:
        #runtime = ctx.runtime
        
        meta["params"].update({
                "include_self": bool(self.include_self),
                "dtype": "float32",
                "diagonal": ("nan" if not self.include_self else "distance_0"),
            })

        return meta
