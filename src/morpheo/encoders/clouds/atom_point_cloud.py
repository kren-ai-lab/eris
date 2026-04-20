from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, ClassVar

import numpy as np

from morpheo.core.base_encoder import BaseAtomEncoder
from morpheo.core.encoder_metadata import build_params, build_attr_keys, NODE_ATTR_KEY_MAP
from morpheo.core.io import load_and_extract_atoms
from morpheo.core.types import StructureArtifact, StructureInput
#from morpheo.core.artifacts import ARTIFACT_SCHEMA_VERSION

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
    name: ClassVar[str] = "atom_point_cloud_ca"
    representation_type: ClassVar[str] = "point_cloud"
    family: ClassVar[str] = "cloud"
    level: ClassVar[str] = "ca"
    
    include_hetero: bool = False
    include_h: bool = False
    altloc: str = "first"  # "first" or "A"
    amino_acids_only: bool = False
    node_attr_mode: str = "elem+role+bfactor+occ"
    

    def _create_data(self, inp: StructureInput) -> Dict[str, Any]:
        b = load_and_extract_atoms(
            inp,
            include_hetero=self.include_hetero,
            include_h=self.include_h,
            altloc=self.altloc,
            amino_acids_only=self.amino_acids_only,
        )

        # Base data always stored
        data: Dict[str, Any] = {
            "pos": b["pos"].astype(np.float32),
            "elem_index": b["elem_index"].astype(np.int16),
            "atom_role": b["atom_role"].astype(np.int8),
            "bfactor": b["bfactor"].astype(np.float32),
            "occ": b["occ"].astype(np.float32),
            "residue_ids": b["residue_ids"].astype(np.str_),
            "resname": b["resname"].astype(np.str_),
            "atom_name": b["atom_name"].astype(np.str_),
            "seq_pos_residue": b["seq_pos_residue"].astype(np.int32),
            "aa_index_residue": b["aa_index_residue"].astype(np.int16),
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
        pass



    def _create_metadata(self, inp: StructureInput) -> Dict[str, Any]:
        meta = self._base_metadata(inp)

        # if "node_attr" in data:
        #     keys = []
        #     if "elem" in self.mode:
        #         keys.append("elem_onehot")
        #     if "role" in self.mode:
        #         keys.append("role_onehot")
        #     if "bfactor" in self.mode:
        #         keys.append("bfactor")
        #     if "occ" in self.mode:
        #         keys.append("occ")

        #     meta["feature_keys"] = {
        #         "node_attr_keys": keys
        #     }
        

        meta["params"] = build_params(
            include_hetero = bool(self.include_hetero),
            include_h = bool(self.include_h),
            altloc = str(self.altloc),
            amino_acids_only = bool(self.amino_acids_only),
            mode = str(self.node_attr_mode),
        )

        meta["feature_keys"] = {
            "node_attr_keys": build_attr_keys(
                self.node_attr_mode, NODE_ATTR_KEY_MAP
            ),
        }

        return meta