from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, ClassVar
from typing import TYPE_CHECKING

import numpy as np

from morpheo.features.residue import build_node_attr
from morpheo.features.edge import build_edge_attr
from morpheo.features.graph_stats import compute_graph_stats, pack_graph_stats

from morpheo.core.base_encoder import BaseResidueEncoder, EncodingContext
from morpheo.features.residue import build_node_attr
from morpheo.core.representations import GraphRepresentation

if TYPE_CHECKING:
    from morpheo.core.representations.base_rep import BaseRepresentation

_NEIGHBOR_OFFSETS = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]


def _cell_index(pos: np.ndarray, cell_size: float) -> np.ndarray:
    return np.floor(pos / float(cell_size)).astype(np.int64)


def _build_cell_dict(cells: np.ndarray) -> dict[Tuple[int, int, int], List[int]]:
    d: dict[Tuple[int, int, int], List[int]] = {}
    for i, c in enumerate(cells):
        key = (int(c[0]), int(c[1]), int(c[2]))
        d.setdefault(key, []).append(i)
    return d


def _collect_local_neighbor_candidates(pos: np.ndarray, cutoff: float) -> List[List[Tuple[int, float]]]:
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


def _collect_global_neighbor_candidates(pos: np.ndarray) -> List[List[Tuple[int, float]]]:
    """
    Collect all neighbor candidates without considering cutoff.

    Returns
    -------
    neigh : list of length N
        neigh[i] is a list of (j, dist_ij).
    """
    n = pos.shape[0]
    neigh: List[List[Tuple[int, float]]] = [[] for _ in range(n)]

    for i in range(n):
        pi = pos[i]
        for j in range(n):
            if i == j:
                continue
            diff = pi - pos[j]
            dist = float(np.sqrt(diff[0] * diff[0] +
                                 diff[1] * diff[1] +
                                 diff[2] * diff[2]))
            neigh[i].append((j, dist))

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
class ResidueKNNGraph(BaseResidueEncoder):
    #Identity
    name: str = "residue_knn_graph_ns"
    representation: ClassVar[BaseRepresentation] = GraphRepresentation()

    #Instance
    knn_mode: str = "local" #or global
    k: int = 16
    cutoff: float = 12.0  # max search radius for candidates (must be >= typical neighbor distances)
    symmetric: bool = True
    include_self_loops: bool = False
    edge_attr_mode = "dist+rbf+seq+local"
    node_attr_mode = "onehot+physchem+plddt+bfactor"
    rbf_k = 16
    rbf_dmax = 20.0
    local_cutoff = 4


    #Overrides post_init to normalize cutoff before fingerprint calculation
    def __post_init__(self):
        super().__post_init__()
        if self.knn_mode == "global":
            self.cutoff = None


    def _create_data(self, ctx: EncodingContext) -> Dict[str, Any]:
        strc = ctx.raw_structure
        centroid = ctx.centroid
        pos_out = ctx.pos
        n = int(pos_out.shape[0])

        # Candidates
        if self.knn_mode.lower() == "local":
            candidates = _collect_local_neighbor_candidates(pos_out, cutoff=float(self.cutoff))
        elif self.knn_mode.lower() == "global":
            candidates = _collect_global_neighbor_candidates(pos_out)
        else:
            raise ValueError(
                f"Invalid knn_mode='{self.knn_mode}'. "
                "Expected 'local' or 'global'."
            )

        # Graph building    
        edge_index, edge_dist = _knn_edges_from_candidates(
            candidates=candidates,
            k=int(self.k),
            symmetric=bool(self.symmetric),
        )

        seq_pos = np.arange(n, dtype=np.int64)

        stats = compute_graph_stats(edge_index=edge_index, edge_dist=edge_dist, n_nodes=n)
        graph_attr, graph_attr_keys = pack_graph_stats(stats)

        node_attr = build_node_attr(
            strc["aa_index"],
            plddt=strc["plddt"],
            bfactor_ca=strc["bfactor_ca"],
            mode=self.node_attr_mode,
        )

        if self.include_self_loops:
            loops = np.arange(n, dtype=np.int64)
            loop_edges = np.stack([loops, loops], axis=0)
            loop_dist = np.zeros((n,), dtype=np.float32)

            edge_index = np.concatenate([edge_index, loop_edges], axis=1)
            edge_dist = np.concatenate([edge_dist, loop_dist], axis=0)

        edge_attr = build_edge_attr(
            edge_index=edge_index,
            edge_dist=edge_dist,
            mode=self.edge_attr_mode,
            rbf_k=self.rbf_k,
            rbf_dmax=self.rbf_dmax,
            seq_pos=seq_pos,
            local_cutoff=self.local_cutoff,
        )

        data: Dict[str, Any] = {
            "pos": pos_out.astype(np.float32),
            "centroid": centroid.astype(np.float32),
            "residue_ids": strc["residue_ids"].astype(np.str_),
            "resname": strc["resname"].astype(np.str_),
            "aa_index": strc["aa_index"].astype(np.int16),
            "edge_index": edge_index.astype(np.int64),
            "edge_dist": edge_dist.astype(np.float32),
            "node_attr": node_attr.astype(np.float32),
            "edge_attr": edge_attr.astype(np.float32),
            "graph_attr": np.asarray(graph_attr, dtype=np.float32),
            "graph_attr_keys": np.asarray(graph_attr_keys, dtype=np.str_),
        }

        if strc["plddt"] is not None:
            data["plddt"] = strc["plddt"].astype(np.float32)
        if strc["bfactor_ca"] is not None:
            data["bfactor_ca"] = strc["bfactor_ca"].astype(np.float32)

        return data


    def _update_params_metadata(self, meta: Dict[str, Any], ctx: EncodingContext) -> Dict[str, Any]:
        meta["params"].update({
                "k": int(self.k),
                "knn_mode": self.knn_mode,
                **({"cutoff": self.cutoff} if self.knn_mode == "local" else {}),
                "symmetric": bool(self.symmetric),
                "include_self_loops": bool(self.include_self_loops),
                "node_attr_mode": self.node_attr_mode,
                "edge_attr_mode": self.edge_attr_mode,
                "rbf_k": int(self.rbf_k),
                "rbf_dmax": float(self.rbf_dmax),
                "local_cutoff": int(self.local_cutoff),
        })
        
        return meta
