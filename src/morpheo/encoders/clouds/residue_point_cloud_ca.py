from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from morpheo.core.cache import DiskCache, make_cache_key
from morpheo.core.hashing import sha256_file, sha256_text
from morpheo.core.io import load_and_extract
from morpheo.core.types import StructureArtifact, StructureInput
from morpheo.features.node import build_node_attr

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
                return StructureArtifact(data=data, meta=meta)

        pos, residue_ids, resname, aa_index, plddt, bfactor_ca = load_and_extract(inp)
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

        node_attr_mode = "onehot+physchem+plddt+bfactor"
        node_attr = build_node_attr(
            aa_index,
            plddt=plddt,
            bfactor_ca=bfactor_ca,
            mode=node_attr_mode,
        )

        data: Dict[str, Any] = {
            "pos": pos_out.astype(np.float32),
            "centroid": centroid.astype(np.float32),
            "residue_ids": residue_ids.astype(np.str_),
            "resname": resname.astype(np.str_),
            "aa_index": aa_index.astype(np.int16),
            "node_attr": node_attr.astype(np.float32),
        }

        # keep raw signals if present (useful for downstream / debugging)
        if plddt is not None:
            data["plddt"] = plddt.astype(np.float32)
        if bfactor_ca is not None:
            data["bfactor_ca"] = bfactor_ca.astype(np.float32)

        encoder_version = getattr(self, "version", None) or "0.1.0"
        src = inp.source
        fmt = Path(src).suffix.lstrip(".").lower() or None

        meta: Dict[str, Any] = {
            "encoder_name": self.name,
            "encoder_version": encoder_version,
            "encoder_fingerprint": self.fingerprint(),
            "type": "cloud",
            "level": "ca",
            "family": "cloud",
            "params": {
                "center": bool(self.center),
                "scale": None if self.scale is None else float(self.scale),
                "node_attr_mode": node_attr_mode,
            },
            "input": {
                "source": src,
                "chain_id": inp.chain_id,
                "model_index": inp.model_index,
                "format": fmt,
            },
            "feature_keys": {
                # keep consistent with graphs: features per node/point
                "node_attr_keys": ["aa_onehot", "physchem", "plddt", "bfactor"],
            },
        }

        if self.use_cache:
            cache.set(key, data=data, meta=meta)

        return StructureArtifact(data=data, meta=meta)