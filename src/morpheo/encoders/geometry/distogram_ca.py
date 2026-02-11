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
class DistogramCA:
    name: str = "distogram_ca"
    include_self: bool = True
    use_cache: bool = True
    cache: Optional[DiskCache] = None

    def fingerprint(self) -> str:
        return sha256_text(f"{self.name}|include_self={self.include_self}|use_cache={self.use_cache}")

    def encode(self, inp: StructureInput) -> StructureArtifact:
        cache = self.cache or DiskCache()
        file_hash = sha256_file(Path(inp.source))
        key = make_cache_key(self.name, self.fingerprint(), file_hash, inp.to_string())

        if self.use_cache:
            hit = cache.get(key)
            if hit is not None:
                data, meta = hit
                return StructureArtifact(data=data, meta=meta)

        pos, residue_ids, resname, aa_index, plddt, bfactor_ca = load_and_extract(inp)
        n = int(pos.shape[0])

        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)

        if not self.include_self:
            np.fill_diagonal(dist, np.nan)

        data: Dict[str, Any] = {
            "dist": dist.astype(np.float32),
            "pos": pos.astype(np.float32),
            "residue_ids": residue_ids.astype(np.str_),
            "resname": resname.astype(np.str_),
            "aa_index": aa_index.astype(np.int16),
            "plddt": plddt.astype(np.float32),
            "bfactor_ca": bfactor_ca.astype(np.float32),
        }

        encoder_version = getattr(self, "version", None) or "0.1.0"
        src = inp.source
        fmt = Path(src).suffix.lstrip(".").lower() or None

        meta: Dict[str, Any] = {
            "encoder_name": self.name,
            "encoder_version": encoder_version,
            "encoder_fingerprint": self.fingerprint(),
            "type": "matrix",
            "level": "ca",
            "family": "matrix",
            "params": {
                "include_self": bool(self.include_self),
                "dtype": "float32",
                "diagonal": ("nan" if not self.include_self else "distance_0"),
            },
            "input": {
                "source": src,
                "chain_id": inp.chain_id,
                "model_index": inp.model_index,
                "format": fmt,
            },
            "feature_keys": {
                "matrix_key": "dist",
                "matrix_attr_keys": ["dist"],
            },
        }

        if self.use_cache:
            cache.set(key, data=data, meta=meta)

        return StructureArtifact(data=data, meta=meta)