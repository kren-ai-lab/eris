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
class ResiduePointCloudCA:
    name: str = "residue_point_cloud_ca"
    center: bool = True           # center coordinates around centroid
    scale: Optional[float] = None # if set, divide coords by this constant
    use_cache: bool = True
    cache: Optional[DiskCache] = None

    def fingerprint(self) -> str:
        return sha256_text(
            f"{self.name}|center={self.center}|scale={self.scale}|use_cache={self.use_cache}"
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
        n = int(pos.shape[0])

        pos_out = pos.astype(np.float32)

        if self.center and n > 0:
            centroid = pos_out.mean(axis=0, keepdims=True)
            pos_out = pos_out - centroid
        else:
            centroid = np.zeros((1, 3), dtype=np.float32)

        if self.scale is not None:
            s = float(self.scale)
            if s <= 0:
                raise ValueError("scale must be > 0")
            pos_out = pos_out / s

        data = {
            "pos": pos_out.astype(np.float32),
            "centroid": centroid.astype(np.float32),
            "residue_ids": residue_ids.astype(np.str_),
            "resname": resname.astype(np.str_),
            "aa_index": aa_index.astype(np.int16),
            "plddt": plddt.astype(np.float32),
            "bfactor_ca": bfactor_ca.astype(np.float32),
        }

        meta: Dict[str, Any] = {
            "encoder_name": self.name,
            "input": inp.to_string(),
            "center": bool(self.center),
            "scale": None if self.scale is None else float(self.scale),
            "n_residues": int(n),
            "shapes": {k: list(v.shape) for k, v in data.items()},
        }

        if self.use_cache:
            cache.set(key, data=data, meta=meta)

        return StructureArtifact(self.name, data=data, meta=meta)


register_encoder("residue_point_cloud_ca", lambda: ResiduePointCloudCA())
