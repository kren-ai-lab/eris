from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from morpheo.core.cache import DiskCache, make_cache_key
from morpheo.core.hashing import sha256_file, sha256_text
from morpheo.core.io import load_and_extract_atoms
from morpheo.core.types import StructureArtifact, StructureInput


def _knn_edge_index(pos: np.ndarray, k: int) -> np.ndarray:
    """
    Naive O(N^2) kNN (directed).
    Returns edge_index (2,E) int64.
    """
    pos = np.asarray(pos, dtype=np.float32)
    n = int(pos.shape[0])
    if n <= 1:
        return np.zeros((2, 0), dtype=np.int64)

    k_eff = int(max(1, min(k, n - 1)))

    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float32)
    np.fill_diagonal(dist, np.inf)

    nn = np.argpartition(dist, kth=k_eff - 1, axis=1)[:, :k_eff]  # (N,k)
    src = np.repeat(np.arange(n, dtype=np.int64), k_eff)
    dst = nn.reshape(-1).astype(np.int64)
    return np.stack([src, dst], axis=0)


@dataclass
class AtomKNNGraph:
    """
    Atom-level kNN graph built on atomic coordinates.

    Stores:
      - pos (N,3)
      - edge_index (2,E)
      - edge_dist (E,)
      - plus atom descriptors: elem_index, atom_role, bfactor, occ, residue_ids, etc.

    Optional:
      - node_attr (N,F) if mode != "none"
    """
    name: str = "atom_knn_graph"
    k: int = 16
    include_hetero: bool = False
    include_h: bool = False
    altloc: str = "first"
    amino_acids_only: bool = False

    # "none" | "elem+role+bfactor+occ" (same as cloud)
    mode: str = "elem+role+bfactor+occ"

    include_self_loops: bool = False
    use_cache: bool = True
    cache: Optional[DiskCache] = None

    def fingerprint(self) -> str:
        return sha256_text(
            f"{self.name}|k={self.k}|het={self.include_hetero}|h={self.include_h}|altloc={self.altloc}|aa_only={self.amino_acids_only}|mode={self.mode}|self_loops={self.include_self_loops}|use_cache={self.use_cache}"
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

        b = load_and_extract_atoms(
            inp,
            include_hetero=self.include_hetero,
            include_h=self.include_h,
            altloc=self.altloc,
            amino_acids_only=self.amino_acids_only,
        )

        pos = b["pos"].astype(np.float32)
        n = int(pos.shape[0])

        edge_index = _knn_edge_index(pos, k=int(self.k))

        if not self.include_self_loops and edge_index.shape[1] > 0:
            src = edge_index[0]
            dst = edge_index[1]
            keep = src != dst
            edge_index = edge_index[:, keep]

        if edge_index.shape[1] == 0:
            edge_dist = np.zeros((0,), dtype=np.float32)
        else:
            src = edge_index[0]
            dst = edge_index[1]
            d = pos[src] - pos[dst]
            edge_dist = np.sqrt((d * d).sum(axis=1)).astype(np.float32)

        data: Dict[str, Any] = {
            "pos": pos,
            "edge_index": edge_index.astype(np.int64),
            "edge_dist": edge_dist.astype(np.float32),
            "elem_index": b["elem_index"].astype(np.int16),
            "atom_role": b["atom_role"].astype(np.int8),
            "bfactor": b["bfactor"].astype(np.float32),
            "occ": b["occ"].astype(np.float32),
            "residue_ids": b["residue_ids"].astype(np.str_),
            "resname": b["resname"].astype(np.str_),
            "atom_name": b["atom_name"].astype(np.str_),
            "seq_pos_residue": b["seq_pos_residue"].astype(np.int32),
            "aa_index_residue": b["aa_index_residue"].astype(np.int16),
        }

        # Optional: store node_attr for torch/pyg to pick up as x
        mode = (self.mode or "none").strip().lower()
        if mode != "none":
            feats = []

            if "elem" in mode:
                e = data["elem_index"].astype(np.int64)
                num_e = int(np.max(e)) + 1 if e.size else 1
                oh_e = np.zeros((n, num_e), dtype=np.float32)
                m = (e >= 0) & (e < num_e)
                oh_e[np.arange(n)[m], e[m]] = 1.0
                feats.append(oh_e)

            if "role" in mode:
                r = data["atom_role"].astype(np.int64)
                num_r = int(np.max(r)) + 1 if r.size else 1
                oh_r = np.zeros((n, num_r), dtype=np.float32)
                m = (r >= 0) & (r < num_r)
                oh_r[np.arange(n)[m], r[m]] = 1.0
                feats.append(oh_r)

            if "bfactor" in mode:
                bf = np.nan_to_num(data["bfactor"], nan=0.0).reshape(-1, 1).astype(np.float32)
                if float(np.nanmax(bf)) > 1.5:
                    bf = bf / 100.0
                feats.append(bf)

            if "occ" in mode:
                oc = np.nan_to_num(data["occ"], nan=0.0).reshape(-1, 1).astype(np.float32)
                feats.append(oc)

            if feats:
                data["node_attr"] = np.concatenate(feats, axis=1).astype(np.float32)

        meta: Dict[str, Any] = {
            "encoder_name": self.name,
            "type": "graph",
            "level": "atom",
            "family": "atom",
            "input": inp.to_string(),
            "k": int(self.k),
            "include_self_loops": bool(self.include_self_loops),
            "include_hetero": bool(self.include_hetero),
            "include_h": bool(self.include_h),
            "altloc": str(self.altloc),
            "amino_acids_only": bool(self.amino_acids_only),
            "mode": str(self.mode),
            "n_atoms": int(n),
            "n_edges": int(edge_index.shape[1]),
            "shapes": {k: list(v.shape) for k, v in data.items() if hasattr(v, "shape")},
        }

        if self.use_cache:
            cache.set(key, data=data, meta=meta)

        return StructureArtifact(self.name, data=data, meta=meta)
