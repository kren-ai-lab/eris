from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from morpheo.core.cache import DiskCache, make_cache_key
from morpheo.core.hashing import sha256_file, sha256_text
from morpheo.core.io import load_and_extract
from morpheo.core.types import StructureArtifact, StructureInput
from morpheo.features.node import build_node_attr
from morpheo.features.edge import build_edge_attr


_NEIGHBOR_OFFSETS = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]


def _cell_index(pos: np.ndarray, cell_size: float) -> np.ndarray:
    return np.floor(pos / float(cell_size)).astype(np.int64)


def _build_cell_dict(cells: np.ndarray) -> dict[Tuple[int, int, int], List[int]]:
    d: dict[Tuple[int, int, int], List[int]] = {}
    for i, c in enumerate(cells):
        key = (int(c[0]), int(c[1]), int(c[2]))
        d.setdefault(key, []).append(i)
    return d


def _collect_neighbor_candidates(pos: np.ndarray, cutoff: float) -> List[List[Tuple[int, float]]]:
    """
    Collect neighbor candidates within cutoff for each node using spatial hashing.

    Returns
    -------
    neigh : list of length N
        neigh[i] is a list of (j, dist_ij) with dist_ij <= cutoff and j != i.
    """
    n = pos.shape[0]
    r = float(cutoff)
    r2 = r * r

    cells = _cell_index(pos, r)
    cell_dict = _build_cell_dict(cells)

    neigh: List[List[Tuple[int, float]]] = [[] for _ in range(n)]

    for cell_key, idxs in cell_dict.items():
        cx, cy, cz = cell_key

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
                            dist = float(np.sqrt(dsq))
                            neigh[i].append((j, dist))
                            neigh[j].append((i, dist))
            else:
                # across two different cells: full cross
                for i in idxs:
                    pi = pos[i]
                    for j in jdxs:
                        diff = pi - pos[j]
                        dsq = float(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
                        if dsq <= r2:
                            dist = float(np.sqrt(dsq))
                            neigh[i].append((j, dist))
                            neigh[j].append((i, dist))

    return neigh


def _knn_edges_from_candidates(
    candidates: List[List[Tuple[int, float]]],
    k: int,
    symmetric: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build KNN edges from per-node candidate lists.

    Parameters
    ----------
    candidates
        candidates[i] = list of (j, dist)
    k
        number of nearest neighbors per node
    symmetric
        if True, ensure undirectedness by adding reverse edges for all edges

    Returns
    -------
    edge_index : (2,E) int64
    edge_dist : (E,) float32
    """
    n = len(candidates)
    k = int(k)

    edges_i: List[int] = []
    edges_j: List[int] = []
    dists: List[float] = []

    for i in range(n):
        neigh = candidates[i]
        if not neigh:
            continue

        # sort by distance and take top-k
        neigh_sorted = sorted(neigh, key=lambda t: t[1])[:k]

        for j, dist in neigh_sorted:
            edges_i.append(i)
            edges_j.append(int(j))
            dists.append(float(dist))

    if not edges_i:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    if symmetric:
        # add reverse edges (may create duplicates; acceptable for PyG, or you can unique later)
        rev_i = edges_j
        rev_j = edges_i
        rev_d = dists

        edges_i = edges_i + rev_i
        edges_j = edges_j + rev_j
        dists = dists + rev_d

    edge_index = np.stack([np.asarray(edges_i, dtype=np.int64), np.asarray(edges_j, dtype=np.int64)], axis=0)
    edge_dist = np.asarray(dists, dtype=np.float32)
    return edge_index, edge_dist


@dataclass
class ResidueKNNGraphNS:
    name: str = "residue_knn_graph_ns"
    k: int = 16
    cutoff: float = 12.0  # max search radius for candidates (must be >= typical neighbor distances)
    symmetric: bool = True
    include_self_loops: bool = False
    use_cache: bool = True
    cache: Optional[DiskCache] = None

    def fingerprint(self) -> str:
        return sha256_text(
            f"{self.name}|k={self.k}|cutoff={self.cutoff}|symmetric={self.symmetric}|self_loops={self.include_self_loops}|use_cache={self.use_cache}"
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

        # candidates within cutoff (no NxN)
        candidates = _collect_neighbor_candidates(pos, cutoff=float(self.cutoff))

        edge_index, edge_dist = _knn_edges_from_candidates(
            candidates=candidates,
            k=int(self.k),
            symmetric=bool(self.symmetric),
        )

        seq_pos = np.arange(pos.shape[0], dtype=np.int64)

        node_attr = build_node_attr(
            aa_index,
            plddt=plddt if "plddt" in locals() else None,
            bfactor_ca=bfactor_ca if "bfactor_ca" in locals() else None,
            mode="onehot+physchem+plddt+bfactor",
        )

        edge_attr = build_edge_attr(
            edge_index=edge_index,
            edge_dist=edge_dist,
            mode="dist+rbf+seq+local",
            rbf_k=16,
            rbf_dmax=20.0,
            seq_pos=seq_pos,
            local_cutoff=4,
        )

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
            "edge_index": edge_index.astype(np.int64),
            "edge_dist": edge_dist.astype(np.float32),
            "node_attr": node_attr,
            "edge_attr": edge_attr,
        }

        if plddt is not None:
            data["plddt"] = plddt.astype(np.float32)
        if bfactor_ca is not None:
            data["bfactor_ca"] = bfactor_ca.astype(np.float32)
        
        meta: Dict[str, Any] = {
            "encoder_name": self.name,
            "input": inp.to_string(),
            "k": int(self.k),
            "cutoff": float(self.cutoff),
            "symmetric": bool(self.symmetric),
            "include_self_loops": bool(self.include_self_loops),
            "n_residues": int(n),
            "n_edges": int(edge_index.shape[1]),
            "shapes": {k: list(v.shape) for k, v in data.items()},
        }

        if self.use_cache:
            cache.set(key, data=data, meta=meta)

        return StructureArtifact(self.name, data=data, meta=meta)

