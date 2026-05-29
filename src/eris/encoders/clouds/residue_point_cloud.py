from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, ClassVar
from typing import TYPE_CHECKING

import numpy as np

from eris.core.base_encoder import BaseResidueEncoder, EncodingContext
from eris.features.node.residue import build_node_attr
from eris.core.representations import PointCloudRepresentation

if TYPE_CHECKING:
    from eris.core.representations.base_rep import BaseRepresentation

@dataclass
class ResiduePointCloudCA(BaseResidueEncoder):
    #Identity
    name: ClassVar[str] = "residue_point_cloud"
    representation: ClassVar[BaseRepresentation] = PointCloudRepresentation()

    #Instance
    node_attr_mode: str = "onehot+physchem+plddt_ca+bfactor_ca"

    def _create_data(self, ctx: EncodingContext) -> Dict[str, Any]:
        strc = ctx.raw_structure
        centroid = ctx.centroid
        pos_out = ctx.pos

        node_attr, node_attr_keys = build_node_attr(
            strc["aa_index"],
            plddt=strc["plddt"],
            bfactor_ca=strc["bfactor_ca"],
            mode=self.node_attr_mode,
            return_keys=True
        )

        ctx.runtime["node_attr_keys"] = node_attr_keys

        data: Dict[str, Any] = {
            "pos": pos_out.astype(np.float32),
            "centroid": centroid.astype(np.float32),
            "node_attr": node_attr.astype(np.float32),
            "residue_ids": strc["residue_ids"].astype(np.str_),
            "resname": strc["resname"].astype(np.str_),
            "aa_index": strc["aa_index"].astype(np.int16),
            "plddt": strc["plddt"].astype(np.float32),
            "bfactor_ca": strc["bfactor_ca"].astype(np.float32),
        }

        return data


    def _update_params_metadata(self, meta: Dict[str, Any], ctx: EncodingContext) -> Dict[str, Any]:
        
        meta["params"].update({
            "node_attr_mode": self.node_attr_mode,
        })

        return meta