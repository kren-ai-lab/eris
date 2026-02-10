from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from morpheo.core.cache import DiskCache, make_cache_key
from morpheo.core.hashing import sha256_file, sha256_text
from morpheo.core.io import load_and_extract
from morpheo.core.registry import register_encoder
from morpheo.core.types import StructureArtifact, StructureInput


def _validate_edges(edges: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(edges), dtype=np.float32).reshape(-1)
    if arr.size < 2:
        raise ValueError("edges must have at least 2 values (bin boundaries).")
    if not np.all(np.isfinite(arr)):
        raise ValueError("edges contains non-finite values.")
    if not np.all(arr[1:] > arr[:-1]):
        raise ValueError("edges must be strictly increasing.")
    return arr


@dataclass
class ContactMapBinsCA:
    name: str = "contact_map_bins_ca"

    # Option A: custom edges (if provided, overrides dmin/dmax/num_bins)
    edges: Optional[Sequence[float]] = None

    # Option B: linspace edges
    dmin: float = 2.0
    dmax: float = 20.0
    num_bins: int = 36

    include_self: bool = False

    # If True: class 0 = > dmax (or > last edge), and classes 1..K are in-range bins
    add_no_contact_class: bool = True

    dtype: str = "uint8"
    use_cache: bool = True
    cache: Optional[DiskCache] = None

    def fingerprint(self) -> str:
        if self.edges is not None:
            e = ",".join(f"{float(x):.6g}" for x in list(self.edges))
            edge_sig = f"edges=[{e}]"
        else:
            edge_sig = f"dmin={self.dmin}|dmax={self.dmax}|num_bins={self.num_bins}"
        return sha256_text(
            f"{self.name}|{edge_sig}|include_self={self.include_self}|no_contact={self.add_no_contact_class}|dtype={self.dtype}|use_cache={self.use_cache}"
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

        # Build edges
        if self.edges is not None:
            edges = _validate_edges(self.edges)
        else:
            edges = np.linspace(float(self.dmin), float(self.dmax), int(self.num_bins) + 1, dtype=np.float32)

        K = int(edges.size - 1)  # number of bins

        # bins in [0..K-1] for in-range
        raw_bins = np.digitize(dist, edges, right=False) - 1
        raw_bins = np.clip(raw_bins, 0, K - 1)

        if self.add_no_contact_class:
            # class 0 = > last edge, class 1..K = in-range bins
            out = raw_bins + 1
            out = out.astype(np.int32)

            out[dist > float(edges[-1])] = 0
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
            "include_self": bool(self.include_self),
            "add_no_contact_class": bool(self.add_no_contact_class),
            "dtype": self.dtype,
            "n_residues": int(n),
            "num_bins": int(K),
            "edges": edges.tolist(),
            "shapes": {k: list(v.shape) for k, v in data.items()},
        }

        if self.use_cache:
            cache.set(key, data=data, meta=meta)

        return StructureArtifact(self.name, data=data, meta=meta)

