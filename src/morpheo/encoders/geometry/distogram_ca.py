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
    use_cache: bool = True
    cache: Optional[DiskCache] = None

    def fingerprint(self) -> str:
        return sha256_text(f"{self.name}|use_cache={self.use_cache}")

    def encode(self, inp: StructureInput) -> StructureArtifact:
        cache = self.cache or DiskCache()
        file_hash = sha256_file(Path(inp.source))
        key = make_cache_key(self.name, self.fingerprint(), file_hash, inp.to_string())

        if self.use_cache:
            hit = cache.get(key)
            if hit is not None:
                data, meta = hit
                return StructureArtifact(self.name, data=data, meta=meta)

        pos, residue_ids = load_and_extract(inp)

        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)

        data = {"dist": dist, "pos": pos, "residue_ids": residue_ids}
        meta: Dict[str, Any] = {
            "encoder_name": self.name,
            "input": inp.to_string(),
            "n_residues": int(pos.shape[0]),
            "shapes": {k: list(v.shape) for k, v in data.items()},
        }

        if self.use_cache:
            cache.set(key, data=data, meta=meta)

        return StructureArtifact(self.name, data=data, meta=meta)


register_encoder("distogram_ca", lambda: DistogramCA())
