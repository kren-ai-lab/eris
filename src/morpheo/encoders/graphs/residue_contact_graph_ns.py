from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from morpheo.core.cache import DiskCache, make_cache_key
from morpheo.core.hashing import sha256_file, sha256_text
from morpheo.core.io import load_and_extract
from morpheo.core.registry import register_encoder
from morpheo.core.types import StructureArtifact, StructureInput


def _cell_index(pos: np.ndarray, cell_size: float) -> np.ndarray:
    # floor divide for negatives as well
    return np.floor(pos / float(cell_size)).astype(np.int64)


def _build_cell_dict(cells: np.ndarray) -> dict[Tuple[int, int, int], list[int]]:
    d: dict[Tuple[int, int, int], list[int]] = {}
    for i, c in enumerate(cells):
        key = (int(c[0]), int(c[1]), int(c[2]))
        d.setdefault(key, []).append(i)
    return d


_NEIGHBOR_OFFSETS = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]


def _neighbor_search_edges(pos: np.ndarray, cutoff: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build residue-residue edges using spatial hashing (cell lists).
    Returns (edge_index (2,E), edge_dist (E,))
    Undirected edges are returned as *both* directions (i->j and j->i).
    """
    n = pos.shape[0]
    if n == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    r = float(cutoff)
    r2 = r * r

    cells = _cell_index(pos, r)
    cell_dict = _build_cell_dict(cells)

    # collect unique undirected pairs i<j
    pairs_i: list[int] = []
    pairs_j: list[int] = []
    dists: list[float] = []

    for cell_key, idxs in cell_dict.items():
        cx, cy, cz = cell_key

        # only generate pairs where (cell_key, neighbor_key) is ordered to avoid duplicates
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            nkey = (cx + dx, cy + dy, cz + dz)
            if nkey not in cell_dict:
                continue

            # ordering to avoid double counting across cells
            if nkey < cell_key:
                continue

            jdxs = cell_dict[nkey]

            if nkey == cell_key:
                # within same cell: compare i<j
                for a in range(len(idxs)):
                    i = idxs[a]
                    pi = pos[i]
                    for b in range(a + 1, len(idxs)):
                        j = idxs[b]
                        diff = pi - pos[j]
                        dsq = float(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
                        if dsq <= r2:
                            pairs_i.append(i)
                            pairs_j.append(j)
                            dists.append(np.sqrt(dsq))
            else:
                # across two different cells: full cross
                for i in idxs:
                    pi = pos[i]
                    for j in jdxs:
                        diff = pi - pos[j]
                        dsq = float(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
                        if dsq <= r2:
                            pairs_i.append(i)
                            pairs_j.append(j)
                            dists.append(np.sqrt(dsq))

    if not pairs_i:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    # make directed edges: i->j and j->i
    i_arr = np.asarray(pairs_i, dtype=np.int64)
    j_arr = np.asarray(pairs_j, dtype=np.int64)
    dist_arr = np.asarray(dists, dtype=np.float32)

    edge_i = np.concatenate([i_arr, j_arr], axis=0)
    edge_j = np.concatenate([j_arr, i_arr], axis=0)
    edge_dist = np.concatenate([dist_arr, dist_arr], axis=0)

    edge_index = np.stack([edge_i, edge_j], axis=0).astype(np.int64)
    return edge_index, edge_dist


@dataclass
class ResidueContactGraphNS:
    name: str = "residue_contact_graph_ns"
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

        edge_index, edge_dist = _neighbor_search_edges(pos, cutoff=self.cutoff)

        if self.include_self_loops:
            loops = np.arange(n, dtype=np.int64)
            loop_edges = np.stack([loops, loops], axis=0)
            loop_dist = np.zeros((n,), dtype=np.float32)

            edge_index = np.concatenate([edge_index, loop_edges], axis=1)
            edge_dist = np.concatenate([edge_dist, loop_dist], axis=0)

        data = {
            "pos": pos.astype(np.float32),
            "residue_ids": residue_ids.astype(np.str_),
            "resname": resname.astype(np.str_),
            "aa_index": aa_index.astype(np.int16),
            "plddt": plddt.astype(np.float32),
            "bfactor_ca": bfactor_ca.astype(np.float32),
            "edge_index": edge_index.astype(np.int64),
            "edge_dist": edge_dist.astype(np.float32),
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


register_encoder("residue_contact_graph_ns", lambda: ResidueContactGraphNS())
