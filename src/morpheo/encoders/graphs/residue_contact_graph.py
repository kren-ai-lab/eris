from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from morpheo.core.cache import DiskCache, make_cache_key
from morpheo.core.hashing import sha256_file, sha256_text
from morpheo.core.io import load_and_extract
from morpheo.core.registry import register_encoder
from morpheo.core.types import StructureArtifact, StructureInput


@dataclass
class ResidueContactGraph:
    name: str = "residue_contact_graph"
    cutoff: float = 8.0
    include_self_loops: bool = False
    use_cache: bool = True
    cache: Optional[DiskCache] = None

    def fingerprint(self) -> str:
        return sha256_text(
            f"{self.name}|cutoff={self.cutoff}|self_loops={self.include_self_loops}|use_cache={self.use_cache}"
        )

    def encode(self, inp: StructureInput) -> StructureArtifact:
        cache = self.cache or DiskCache()
        file_hash = sha256_file(Path(inp.source))
        key = make_cache_key(self.name, self.fingerprint(), file_hash, inp.to_string())

        if self.use_cache:
            hit = cache.get(key)
            if hit is not None:
                data, meta = hit
                return StructureArtifact(self.name, data=data, meta=meta)

        pos, residue_ids, resname, aa_index, plddt, bfactor_ca = load_and_extract(inp)
        n = pos.shape[0]

        diff = pos[:, None, :] - pos[None, :, :]
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
            d = dist[iu, ju]
            edge_dist = np.concatenate([d, d]).astype(np.float32)

        data = {
            "pos": pos.astype(np.float32),
            "residue_ids": residue_ids.astype(np.str_),
            "resname": resname.astype(np.str_),
            "aa_index": aa_index.astype(np.int16),
            "plddt": plddt.astype(np.float32),
            "edge_index": edge_index.astype(np.int64),
            "edge_dist": edge_dist.astype(np.float32),
            "bfactor_ca": bfactor_ca.astype(np.float32),
        }

        meta: Dict[str, Any] = {
            "encoder_name": self.name,
            "input": inp.to_string(),
            "cutoff": float(self.cutoff),
            "include_self_loops": bool(self.include_self_loops),
            "n_residues": int(n),
            "n_edges": int(edge_index.shape[1]),
            "shapes": {k: list(v.shape) for k, v in data.items()},
        }

        if self.use_cache:
            cache.set(key, data=data, meta=meta)

        return StructureArtifact(self.name, data=data, meta=meta)


register_encoder("residue_contact_graph", lambda: ResidueContactGraph())
