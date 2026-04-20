from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, ClassVar, Tuple

import numpy as np

from morpheo.core.base_encoder import BaseResidueEncoder
from morpheo.core.encoder_metadata import build_params, build_attr_keys, NODE_ATTR_KEY_MAP, EDGE_ATTR_KEY_MAP
from morpheo.core.io import load_and_extract
from morpheo.core.types import StructureArtifact, StructureInput
from morpheo.features.node import build_node_attr
from morpheo.features.edge import build_edge_attr
from morpheo.features.graph_stats import compute_graph_stats, pack_graph_stats

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
class ResidueContactGraphNS(BaseResidueEncoder):
    name: ClassVar[str] = "residue_contact_graph"
    representation_type: ClassVar[str] = "graph"
    family: ClassVar[str] = "residue_graph"
    level: ClassVar[str] = "ca"

    include_self_loops: bool = False    
    cutoff: float = 8.0
    node_attr_mode = "onehot+physchem+plddt+bfactor"
    edge_attr_mode = "dist+rbf+seq+local"
    rbf_k = 16
    rbf_dmax = 20.0
    local_cutoff = 4
    

    def _create_data(self, inp: StructureInput) -> Dict[str, Any]:
        strc = load_and_extract(inp)
        n = int(strc["pos"].shape[0])

        edge_index, edge_dist = _neighbor_search_edges(strc["pos"], cutoff=self.cutoff)

        stats = compute_graph_stats(edge_index=edge_index, edge_dist=edge_dist, n_nodes=n)
        graph_attr, self.graph_attr_keys = pack_graph_stats(stats)

        seq_pos = np.arange(n, dtype=np.int64)

        
        node_attr = build_node_attr(
            strc["aa_index"],
            plddt=strc["plddt"],
            bfactor_ca=strc["bfactor_ca"],
            mode=self.strc["node_attr_mode"],
        )


        if self.include_self_loops:
            loops = np.arange(n, dtype=np.int64)
            loop_edges = np.stack([loops, loops], axis=0)
            loop_dist = np.zeros((n,), dtype=np.float32)

            edge_index = np.concatenate([edge_index, loop_edges], axis=1)
            edge_dist = np.concatenate([edge_dist, loop_dist], axis=0)

            # NOTE: edge_attr currently computed before loops; if loops are enabled,
            # you may want to recompute edge_attr OR append loop edge_attr rows.
            # Here we leave it as-is to match your original logic.

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
            "pos": strc["pos"].astype(np.float32),
            "residue_ids": strc["residue_ids"].astype(np.str_),
            "resname": strc["resname"].astype(np.str_),
            "aa_index": strc["aa_index"].astype(np.int16),
            "edge_index": edge_index.astype(np.int64),
            "edge_dist": edge_dist.astype(np.float32),
            "node_attr": node_attr.astype(np.float32),
            "edge_attr": edge_attr.astype(np.float32),
            "graph_attr": np.asarray(graph_attr, dtype=np.float32),
            "graph_attr_keys": np.asarray(self.graph_attr_keys, dtype=np.str_),
        }

        if strc["plddt"] is not None:
            data["plddt"] = strc["plddt"].astype(np.float32)
        if strc["bfactor_ca"] is not None:
            data["bfactor_ca"] = strc["bfactor_ca"].astype(np.float32)

        return data
    

    def _create_metadata(self, inp: StructureInput) -> Dict[str, Any]:
        meta = self._base_metadata(inp)

        meta["params"] = build_params(
            cutoff = float(self.cutoff),
            include_self_loops = bool(self.include_self_loops),
            node_attr_mode = self.node_attr_mode,
            edge_attr_mode = self.edge_attr_mode,
            rbf_k = int(self.rbf_k),
            rbf_dmax = float(self.rbf_dmax),
            local_cutoff = int(self.local_cutoff),
        )


        meta["feature_keys"] = {
            "node_attr_keys": build_attr_keys(
                self.node_attr_mode, NODE_ATTR_KEY_MAP
            ),
            "edge_attr_keys": build_attr_keys(
                self.edge_attr_mode, EDGE_ATTR_KEY_MAP
            ),
            "graph_attr_keys": list(getattr(self, "graph_attr_keys", [])),
        },


        return meta