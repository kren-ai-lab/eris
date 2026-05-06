from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, ClassVar
from typing import TYPE_CHECKING

import numpy as np

from morpheo.core.base_encoder import BaseAtomEncoder, EncodingContext
#from morpheo.core.artifacts import ARTIFACT_SCHEMA_VERSION
from morpheo.core.contracts import AtomContract
from morpheo.core.representations import PointCloudRepresentation

if TYPE_CHECKING:
    from morpheo.core.contracts.base_contract import GranularityContract
    from morpheo.core.representations.base_rep import BaseRepresentation

def _onehot(idx: np.ndarray, num: int) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    out = np.zeros((idx.shape[0], num), dtype=np.float32)
    m = (idx >= 0) & (idx < num)
    out[np.arange(idx.shape[0])[m], idx[m]] = 1.0
    return out


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
    name: ClassVar[str] = "atom_point_cloud_ca"
    representation: ClassVar[BaseRepresentation] = PointCloudRepresentation()
    
    include_hetero: bool = False
    include_h: bool = False
    altloc: str = "first"  # "first" or "A"
    amino_acids_only: bool = False
    node_attr_mode: str = "elem+role+bfactor+occ"
    

    def _create_data(self, ctx: EncodingContext) -> Dict[str, Any]:
        strc = ctx.raw_structure
        centroid = ctx.centroid
        pos_out = ctx.pos

        # Base data always stored
        data: Dict[str, Any] = {
            "pos": pos_out.astype(np.float32),
            "centroid": centroid.astype(np.float32),
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

        # Optional node_attr (features per atom)
        mode = (self.node_attr_mode or "none").strip().lower()
        if mode != "none":
            feats = []

            if "elem" in mode:
                e = data["elem_index"].astype(np.int64)
                num_e = int(np.max(e)) + 1 if e.size else 1
                feats.append(_onehot(e, num=num_e))

            if "role" in mode:
                r = data["atom_role"].astype(np.int64)
                num_r = int(np.max(r)) + 1 if r.size else 1
                feats.append(_onehot(r, num=num_r))

            if "bfactor" in mode:
                bf = np.nan_to_num(data["bfactor"].astype(np.float32), nan=0.0).reshape(-1, 1)
                if float(np.nanmax(bf)) > 1.5:
                    bf = bf / 100.0
                feats.append(bf.astype(np.float32))

            if "occ" in mode:
                oc = np.nan_to_num(data["occ"].astype(np.float32), nan=0.0).reshape(-1, 1)
                feats.append(oc.astype(np.float32))

            if feats:
                data["node_attr"] = np.concatenate(feats, axis=1).astype(np.float32)
        
        return data



    def _update_params_metadata(self, meta: Dict[str, Any], ctx: EncodingContext) -> Dict[str, Any]:

        meta["params"].update({
            "include_hetero": bool(self.include_hetero),
            "include_h": bool(self.include_h),
            "altloc": str(self.altloc),
            "amino_acids_only": bool(self.amino_acids_only),
            "mode": str(self.node_attr_mode),}
        )

        return meta