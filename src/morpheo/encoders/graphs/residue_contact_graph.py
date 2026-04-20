from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, ClassVar, Tuple

import numpy as np

from morpheo.core.base_encoder import BaseResidueEncoder
from morpheo.core.io import load_and_extract
from morpheo.core.encoder_metadata import build_params
from morpheo.core.types import StructureArtifact, StructureInput


@dataclass
class ResidueContactGraph(BaseResidueEncoder):
    name: ClassVar[str] = "residue_contact_graph"
    representation_type: ClassVar[str] = "graph"
    family: ClassVar[str] = "residue_graph"
    level: ClassVar[str] = "ca"

    include_self_loops: bool = False    
    cutoff: float = 8.0
    

    def _create_data(self, inp: StructureInput) -> Dict[str, Any]:
        strc = load_and_extract(inp)
        n = int(strc["pos"].shape[0])

        diff = strc["pos"][:, None, :] - strc["pos"][None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)

        mask = dist <= float(self.cutoff)
        if not self.include_self_loops:
            np.fill_diagonal(mask, False)

        iu, ju = np.where(np.triu(mask, k=1))
        if iu.size == 0:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_dist = np.zeros((0,), dtype=np.float32)
        else:
            src = np.concatenate([iu, ju]).astype(np.int64)
            dst = np.concatenate([ju, iu]).astype(np.int64)
            edge_index = np.stack([src, dst], axis=0)
            d = dist[iu, ju].astype(np.float32)
            edge_dist = np.concatenate([d, d]).astype(np.float32)

        data: Dict[str, Any] = {
            "pos": strc["pos"].astype(np.float32),
            "residue_ids": strc["residue_ids"].astype(np.str_),
            "resname": strc["resname"].astype(np.str_),
            "aa_index": strc["aa_index"].astype(np.int16),
            "plddt": strc["plddt"].astype(np.float32),
            "bfactor_ca": strc["bfactor_ca"].astype(np.float32),
            "edge_index": edge_index.astype(np.int64),
            "edge_dist": edge_dist.astype(np.float32),
        }

        return data

    
    def _create_metadata(self, inp: StructureInput) -> Dict[str, Any]:
        meta = self._base_metadata(inp)

        meta["params"] = build_params(
            cutoff = float(self.cutoff),
            include_self_loops = bool(self.include_self_loops),
        )

        meta["feature_keys"] = {
            "edge_attr_keys": ["edge_dist"],
        }

        return meta