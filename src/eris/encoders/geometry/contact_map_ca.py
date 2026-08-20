from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, ClassVar
from typing import TYPE_CHECKING
from typing import Any, Dict, Optional, Sequence

import numpy as np

from eris.core.base_encoder import BaseResidueEncoder, EncodingContext
from eris.core.representations import ContactMapRepresentation

if TYPE_CHECKING:
    from eris.core.representations.base_rep import BaseRepresentation


@dataclass
class ContactMapCA(BaseResidueEncoder):
    #Identity
    name: ClassVar[str] = "residue_contact_map"
    representation: ClassVar[BaseRepresentation] = ContactMapRepresentation()

    #Instance
    cutoff: float = 8.0
    include_self: bool = False
    dtype: str = "uint8"  # or "bool"


    def _create_data(self, ctx: EncodingContext) -> Dict[str, Any]:
        strc = ctx.raw_structure
        centroid = ctx.centroid
        pos_out = ctx.pos

        n = int(pos_out.shape[0])

        diff = pos_out[:, None, :] - pos_out[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)

        contact = dist <= float(self.cutoff)
        
        if not self.include_self:
            np.fill_diagonal(contact, False)

        if self.dtype == "bool":
            contact_out = contact
        else:
            contact_out = contact.astype(np.uint8)

        data: Dict[str, Any] = {
            "contact": contact_out,
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
                "cutoff": float(self.cutoff),
                "include_self": bool(self.include_self),
                "dtype": str(self.dtype)
            })

        return meta
