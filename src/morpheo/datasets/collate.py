from __future__ import annotations

from typing import Any, Dict, List

import torch

def collate_graph(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate for graphs WITHOUT PyG dependency.

    Expects each sample has:
      - x: (N,F)
      - edge_index: (2,E)
    Optional:
      - edge_dist: (E,)
      - pos: (N,3)
    """
    xs = []
    edge_indices = []
    edge_dists = []
    poss = []
    batch_vec = []

    node_offset = 0
    for i, b in enumerate(batch):
        x = b["x"]
        ei = b["edge_index"]

        n = x.shape[0]
        xs.append(x)
        edge_indices.append(ei + node_offset)

        if "edge_dist" in b:
            edge_dists.append(b["edge_dist"])
        if "pos" in b:
            poss.append(b["pos"])

        batch_vec.append(torch.full((n,), i, dtype=torch.long, device=x.device))
        node_offset += n

    out: Dict[str, Any] = {
        "x": torch.cat(xs, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "batch": torch.cat(batch_vec, dim=0),
    }

    if edge_dists:
        out["edge_dist"] = torch.cat(edge_dists, dim=0)

    if poss:
        out["pos"] = torch.cat(poss, dim=0)

    if "id" in batch[0]:
        out["id"] = [b.get("id") for b in batch]
    if "encoder" in batch[0]:
        out["encoder"] = [b.get("encoder") for b in batch]

    return out

def collate_grid(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate for voxel grids.
    Expects each sample has:
      - x: (C,G,G,G)
    """
    x = torch.stack([b["x"] for b in batch], dim=0)  # (B,C,G,G,G)

    out: Dict[str, Any] = {"x": x}

    # keep ids if present
    if "id" in batch[0]:
        out["id"] = [b.get("id") for b in batch]
    if "encoder" in batch[0]:
        out["encoder"] = [b.get("encoder") for b in batch]

    # optional: carry spatial meta (assumes consistent per-batch)
    for k in ("voxel_size", "origin", "centroid"):
        if k in batch[0]:
            # if stored as tensors
            if torch.is_tensor(batch[0][k]):
                out[k] = batch[0][k]
            else:
                out[k] = batch[0][k]

    return out
