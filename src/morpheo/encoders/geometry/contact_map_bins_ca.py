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
class ContactMapBinsCA:
    name: str = "contact_map_bins_ca"
    # bins for "contact distances"
    dmin: float = 2.0
    dmax: float = 20.0
    num_bins: int = 36
    include_self: bool = False
    # If True, class 0 = > dmax, and classes 1..K are in-range bins
    add_no_contact_class: bool = True
    dtype: str = "uint8"
    use_cache: bool = True
    cache: Optional[DiskCache] = None
    edges: Optional[list[float]] = None
    
    def fingerprint(self) -> str:
        return sha256_text(
            f"{self.name}|dmin={self.dmin}|dmax={self.dmax}|num_bins={self.num_bins}|include_self={self.include_self}|no_contact={self.add_no_contact_class}|dtype={self.dtype}|use_cache={self.use_cache}"
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

        if not self.include_self:
            np.fill_diagonal(dist, np.nan)

        edges = np.linspace(float(self.dmin), float(self.dmax), int(self.num_bins) + 1, dtype=np.float32)

        # bins in 0..K-1 for in-range; values <dmin -> 0, >dmax -> K-1 after clipping
        raw_bins = np.digitize(dist, edges, right=False) - 1
        raw_bins = np.clip(raw_bins, 0, int(self.num_bins) - 1)

        if self.add_no_contact_class:
            # class 0 = > dmax, class 1..K = in-range bins
            out = raw_bins + 1
            out = out.astype(np.int32)

            # mark > dmax as 0
            out[dist > float(self.dmax)] = 0
            # keep NaNs (e.g., diagonal) at 0
            out[np.isnan(dist)] = 0
        else:
            out = raw_bins.astype(np.int32)
            out[np.isnan(dist)] = 0

        if self.dtype == "uint16":
            contact_bins = out.astype(np.uint16)
        elif self.dtype == "int16":
            contact_bins = out.astype(np.int16)
        else:
            contact_bins = out.astype(np.uint8)

        data = {
            "contact_bins": contact_bins,
            "bin_edges": edges.astype(np.float32),
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
            "dmin": float(self.dmin),
            "dmax": float(self.dmax),
            "num_bins": int(self.num_bins),
            "include_self": bool(self.include_self),
            "add_no_contact_class": bool(self.add_no_contact_class),
            "dtype": self.dtype,
            "n_residues": int(n),
            "shapes": {k: list(v.shape) for k, v in data.items()},
        }

        if self.use_cache:
            cache.set(key, data=data, meta=meta)

        return StructureArtifact(self.name, data=data, meta=meta)


register_encoder("contact_map_bins_ca", lambda: ContactMapBinsCA())
