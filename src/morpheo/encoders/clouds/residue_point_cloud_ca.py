from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, ClassVar, Tuple

import numpy as np

from morpheo.core.base_encoder import BaseResidueEncoder
from morpheo.core.encoder_metadata import build_params, build_attr_keys, NODE_ATTR_KEY_MAP
from morpheo.core.io import load_and_extract
from morpheo.core.encoder_metadata import build_params
from morpheo.core.types import StructureArtifact, StructureInput
from morpheo.features.node import build_node_attr

@dataclass
class ResiduePointCloudCA(BaseResidueEncoder):
    name: ClassVar[str] = "residue_point_cloud_ca"
    representation_type: ClassVar[str] = "point_cloud"
    family: ClassVar[str] = "cloud"
    level: ClassVar[str] = "ca"

    center: bool = True           # center coordinates around centroid
    scale: Optional[float] = None # if set, divide coords by this constant
    node_attr_mode: str = "onehot+physchem+plddt+bfactor"    

    def _normalize_structure(self, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = int(pos.shape[0])
        pos_out = pos.astype(np.float32)

        if self.center and n > 0:
            centroid = pos_out.mean(axis=0, keepdims=True).astype(np.float32)
            pos_out = pos_out - centroid
        else:
            centroid = np.zeros((1, 3), dtype=np.float32)

        if self.scale is not None:
            s = float(self.scale)
            if s <= 0:
                raise ValueError("scale must be > 0")
            pos_out = (pos_out / s).astype(np.float32)

        return centroid, pos_out
    

    def _create_data(self, inp: StructureInput) -> Dict[str, Any]:
        strc = load_and_extract(inp)
        centroid, pos_out = self._normalize_structure(pos=strc["pos"])

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


    def _create_metadata(self, inp: StructureInput) -> Dict[str, Any]:
        meta = self._base_metadata(inp)

        meta["params"] = build_params(
            center=bool(self.center),
            scale=None if self.scale is None else float(self.scale),
            node_attr_mode=self.node_attr_mode,
        )

        meta["feature_keys"] = {
            "node_attr_keys": build_attr_keys(
                self.node_attr_mode, NODE_ATTR_KEY_MAP
            ),
        }

        return meta