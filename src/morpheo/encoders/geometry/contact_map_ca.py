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
class ContactMapCA:
    name: str = "contact_map_ca"
    cutoff: float = 8.0
    include_self: bool = False
    dtype: str = "uint8"  # or "bool"
    use_cache: bool = True
    cache: Optional[DiskCache] = None

    def fingerprint(self) -> str:
        return sha256_text(
            f"{self.name}|cutoff={self.cutoff}|include_self={self.include_self}|dtype={self.dtype}|use_cache={self.use_cache}"
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

        contact = dist <= float(self.cutoff)
        if not self.include_self:
            np.fill_diagonal(contact, False)

        if self.dtype == "bool":
            contact_out = contact
        else:
            contact_out = contact.astype(np.uint8)

        data = {
            "contact": contact_out,
            "pos": pos.astype(np.float32),
            "residue_ids": residue_ids.astype(np.str_),
            "resname": resname.astype(np.str_),
            "aa_index": aa_index.astype(np.int16),
            "plddt": plddt.astype(np.float32),
            "bfactor_ca": bfactor_ca.astype(np.float32),
        }

        meta: Dict[str, Any] = {
            "encoder_name": self.name,
            "input": inp.to_string(),
            "cutoff": float(self.cutoff),
            "include_self": bool(self.include_self),
            "dtype": self.dtype,
            "n_residues": int(n),
            "shapes": {k: list(v.shape) for k, v in data.items()},
        }

        if self.use_cache:
            cache.set(key, data=data, meta=meta)

        return StructureArtifact(self.name, data=data, meta=meta)


register_encoder("contact_map_ca", lambda: ContactMapCA())
