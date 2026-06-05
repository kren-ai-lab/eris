from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, ClassVar
from typing import TYPE_CHECKING

import numpy as np

from eris.core.base_encoder import BaseAtomEncoder, EncodingContext
from eris.core.representations import PointCloudRepresentation
from eris.features.node.atom import build_node_attr

if TYPE_CHECKING:
    from eris.core.representations.base_rep import BaseRepresentation


@dataclass
class AtomPointCloud(BaseAtomEncoder):
    """
    Atom-level point cloud.

    Stores:
      - pos (N,3)
      - elem_index (N,)
      - atom_role (N,)
      - bfactor (N,)
      - occ (N,)
      - residue_ids (N,)
      - resname (N,)
      - atom_name (N,)
      - seq_pos_residue (N,)
      - aa_index_residue (N,)

    Optional:
      - node_attr (N,F) if node_attr_mode != "none"
    """
    #Identity
    name: ClassVar[str] = "atom_point_cloud"
    representation: ClassVar[BaseRepresentation] = PointCloudRepresentation()
    
    node_attr_mode: str = "elem+role+bfactor+occ"
    

    def _create_data(self, ctx: EncodingContext) -> Dict[str, Any]:
        strc = ctx.raw_structure
        centroid = ctx.centroid
        pos_out = ctx.pos

        node_attr, node_attr_keys = build_node_attr(
            elem_index=strc["elem_index"],
            atom_role=strc["atom_role"],
            bfactor=strc["bfactor"],
            occ=strc["occ"],
            bfactor_norm=True,
            mode=self.node_attr_mode,
            return_keys=True
        )

        ctx.runtime["node_attr_keys"] = node_attr_keys

        # Base data always stored
        data: Dict[str, Any] = {
            "pos": pos_out.astype(np.float32),
            "centroid": centroid.astype(np.float32),
            "node_attr": node_attr.astype(np.float32),
            "elem_index": strc["elem_index"].astype(np.int16),
            "atom_role": strc["atom_role"].astype(np.int8),
            "bfactor": strc["bfactor"].astype(np.float32),
            "occ": strc["occ"].astype(np.float32),
            "residue_ids": strc["residue_ids"].astype(np.str_),
            "resname": strc["resname"].astype(np.str_),
            "atom_name": strc["atom_name"].astype(np.str_),
            "seq_pos_residue": strc["seq_pos_residue"].astype(np.int32),
            "aa_index_residue": strc["aa_index_residue"].astype(np.int16),
        }
        
        return data


    def _update_params_metadata(self, meta: Dict[str, Any], ctx: EncodingContext) -> Dict[str, Any]:

        meta["params"].update({
            "node_attr_mode": str(self.node_attr_mode),}
        )

        return meta