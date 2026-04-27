from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, ClassVar
from typing import TYPE_CHECKING

import numpy as np

from morpheo.core.base_encoder import BaseResidueEncoder, EncodingContext
from morpheo.features.node import build_node_attr
from morpheo.core.representations import PointCloudRepresentation

if TYPE_CHECKING:
    from morpheo.core.representations.base_rep import BaseRepresentation

@dataclass
class ResiduePointCloudCA(BaseResidueEncoder):
    #Identity
    name: ClassVar[str] = "residue_point_cloud_ca"
    representation: ClassVar[BaseRepresentation] = PointCloudRepresentation()

    #Instance
    node_attr_mode: str = "onehot+physchem+plddt+bfactor"

    def _create_data(self, ctx: EncodingContext) -> Dict[str, Any]:
        strc = ctx.raw_structure
        centroid = ctx.centroid
        pos_out = ctx.pos

        node_attr = build_node_attr(
            strc["aa_index"],
            plddt=strc["plddt"],
            bfactor_ca=strc["bfactor_ca"],
            mode=self.node_attr_mode,
        )

        data: Dict[str, Any] = {
            "pos": pos_out.astype(np.float32),
            "centroid": centroid.astype(np.float32),
            "residue_ids": strc["residue_ids"].astype(np.str_),
            "resname": strc["resname"].astype(np.str_),
            "aa_index": strc["aa_index"].astype(np.int16),
            "node_attr": node_attr.astype(np.float32),
        }

        # keep raw signals if present (useful for downstream / debugging)
        if strc["plddt"] is not None:
            data["plddt"] = strc["plddt"].astype(np.float32)
        if strc["bfactor_ca"] is not None:
            data["bfactor_ca"] = strc["bfactor_ca"].astype(np.float32)

        return data


    def _update_params_metadata(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        
        meta["params"].update({
            "node_attr_mode": self.node_attr_mode,
        })

        return meta