from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from eris.core.base_encoder import BaseResidueEncoder, EncodingContext
from eris.core.config import ResiduePointCloudConfig
from eris.core.representations import PointCloudRepresentation
from eris.core.representations.output_declarations import RESIDUE_POINT_CLOUD_OUTPUTS
from eris.features.node.residue import build_node_attr


class ResiduePointCloudCA(BaseResidueEncoder):
    # Identity
    name: ClassVar[str] = "residue_point_cloud"
    version: ClassVar[str] = "0.1.3"
    representation: ClassVar = PointCloudRepresentation()
    output_declarations_v2: ClassVar = RESIDUE_POINT_CLOUD_OUTPUTS

    config_type: ClassVar = ResiduePointCloudConfig
    config: ResiduePointCloudConfig

    def _create_data(self, ctx: EncodingContext) -> dict[str, Any]:
        strc = ctx.raw_structure
        centroid = ctx.centroid
        pos_out = ctx.pos

        node_attr, node_attr_keys = build_node_attr(
            strc["aa_index"],
            plddt_ca=strc["plddt"],
            bfactor_ca=strc["bfactor_ca"],
            residue_ids=strc["residue_ids"],
            chain_id=strc["chain_id"],
            chain_ids=strc.chain_ids,
            model_index=strc.model_index,
            structural_residue_index=strc["structural_residue_index"],
            backbone_n=strc["backbone_n"],
            backbone_ca=strc["backbone_ca"],
            backbone_c=strc["backbone_c"],
            backbone_n_mask=strc["backbone_n_mask"],
            backbone_ca_mask=strc["backbone_ca_mask"],
            backbone_c_mask=strc["backbone_c_mask"],
            resname=strc["resname"],
            polymer_heavy_atom_coordinates=strc["polymer_heavy_atom_coordinates"],
            polymer_heavy_atom_residue_index=strc["polymer_heavy_atom_residue_index"],
            polymer_heavy_atom_name=strc["polymer_heavy_atom_name"],
            polymer_heavy_atom_element=strc["polymer_heavy_atom_element"],
            mode=self.config.node_attr_mode.to_list(),
            return_keys=True,
        )

        ctx.runtime["node_attr_keys"] = node_attr_keys

        data: dict[str, Any] = {
            "pos": pos_out.astype(np.float32),
            "centroid": centroid.astype(np.float32),
            "node_attr": node_attr.astype(np.float32),
            "residue_ids": strc["residue_ids"].astype(np.str_),
            "chain_id": strc["chain_id"].astype(np.str_),
            "omitted_residue_ids": np.asarray(strc.get("omitted_residue_ids", []), dtype=np.str_),
            "resname": strc["resname"].astype(np.str_),
            "aa_index": strc["aa_index"].astype(np.int16),
            "plddt": strc["plddt"].astype(np.float32),
            "bfactor_ca": strc["bfactor_ca"].astype(np.float32),
        }

        return data
