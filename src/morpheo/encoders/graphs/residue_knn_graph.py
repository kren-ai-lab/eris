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
class ResidueKNNGraph:
    name: str = "residue_knn_graph"
    k: int = 16
    symmetrize: bool = True
    include_self_loops: bool = False
    use_cache: bool = True
    cache: Optional[DiskCache] = None

    def fingerprint(self) -> str:
        return sha256_text(
            f"{self.name}|k={self.k}|sym={self.symmetrize}|self_loops={self.include_self_loops}|use_cache={self.use_cache}"
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

        strc = load_and_extract(inp)
        n = int(strc["pos"].shape[0])

        if n <= 1:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_dist = np.zeros((0,), dtype=np.float32)
        else:
            diff = strc["pos"][:, None, :] - strc["pos"][None, :, :]
            dist = np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)

            if not self.include_self_loops:
                np.fill_diagonal(dist, np.inf)

            k_eff = int(min(max(1, int(self.k)), n - 1))
            nbrs = np.argsort(dist, axis=1)[:, :k_eff]  # (N,k)

            src = np.repeat(np.arange(n, dtype=np.int64), k_eff)
            dst = nbrs.reshape(-1).astype(np.int64)
            d = dist[np.arange(n)[:, None], nbrs].reshape(-1).astype(np.float32)

            edge_index = np.stack([src, dst], axis=0)
            edge_dist = d

            if self.symmetrize:
                edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)
                edge_dist = np.concatenate([edge_dist, edge_dist], axis=0)

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

        encoder_version = getattr(self, "version", None) or "0.1.0"
        src_path = inp.source
        fmt = Path(src_path).suffix.lstrip(".").lower() or None

        meta: Dict[str, Any] = {
            "encoder_name": self.name,
            "encoder_version": encoder_version,
            "encoder_fingerprint": self.fingerprint(),
            "type": "graph",
            "level": "ca",
            "family": "residue_graph",
            "params": {
                "k": int(self.k),
                "symmetrize": bool(self.symmetrize),
                "include_self_loops": bool(self.include_self_loops),
            },
            "input": {
                "source": src_path,
                "chain_id": inp.chain_id,
                "model_index": inp.model_index,
                "format": fmt,
            },
            "feature_keys": {
                "edge_attr_keys": ["edge_dist"],
            },
        }

        if self.use_cache:
            cache.set(key, data=data, meta=meta)

        return StructureArtifact(data=data, meta=meta)