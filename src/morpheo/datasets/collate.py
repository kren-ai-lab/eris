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

def collate_matrix(
    batch: List[Dict[str, Any]],
    *,
    pad_value: float = 0.0,
) -> Dict[str, Any]:
    """
    Collate for matrix representations.

    Expects each sample has:
      - x: (1,N,N) if matrix_as_channel=True, else (N,N)

    Returns:
      - x: (B,1,Nmax,Nmax) (always with channel dim)
      - n: list[int] original sizes
      - matrix_key: list[str] if present
    """
    xs = []
    ns = []
    keys = []

    # Determine max N
    for b in batch:
        x = b["x"]
        if x.ndim == 2:
            n = int(x.shape[0])
        elif x.ndim == 3:
            # (1,N,N)
            n = int(x.shape[-1])
        else:
            raise ValueError(f"Expected x to be 2D or 3D, got shape {tuple(x.shape)}")
        ns.append(n)
        keys.append(b.get("matrix_key"))
        xs.append(x)

    nmax = max(ns)
    device = xs[0].device
    dtype = xs[0].dtype

    out_x = torch.full((len(batch), 1, nmax, nmax), float(pad_value), dtype=dtype, device=device)

    for i, x in enumerate(xs):
        if x.ndim == 2:
            x2 = x
        else:
            x2 = x[0]  # (N,N)

        n = ns[i]
        out_x[i, 0, :n, :n] = x2

    out: Dict[str, Any] = {
        "x": out_x,
        "n": ns,
        "matrix_key": keys,
    }

    # keep ids if present
    if "id" in batch[0]:
        out["id"] = [b.get("id") for b in batch]
    if "encoder" in batch[0]:
        out["encoder"] = [b.get("encoder") for b in batch]

    # optional bin edges: only safe to include if all are identical
    if "bin_edges" in batch[0]:
        same = True
        ref = batch[0].get("bin_edges")
        if torch.is_tensor(ref):
            for b in batch[1:]:
                be = b.get("bin_edges")
                if (be is None) or (not torch.is_tensor(be)) or (be.shape != ref.shape) or (not torch.equal(be, ref)):
                    same = False
                    break
        else:
            same = False

        if same:
            out["bin_edges"] = ref

    return out

def collate_grid_pad(
    batch: List[Dict[str, Any]],
    *,
    pad_value: float = 0.0,
    return_mask: bool = False,
) -> Dict[str, Any]:
    """
    Collate for voxel grids with padding over grid size and channels.

    Expects each sample has:
      - x: (C,G,G,G)

    Returns:
      - x: (B,Cmax,Gmax,Gmax,Gmax)
      - c: list[int] original channels
      - g: list[int] original grid sizes
      - mask (optional): (B,1,Gmax,Gmax,Gmax) valid voxels mask per sample
    """
    xs = [b["x"] for b in batch]
    device = xs[0].device
    dtype = xs[0].dtype

    Cs = [int(x.shape[0]) for x in xs]
    Gs = [int(x.shape[-1]) for x in xs]

    Cmax = max(Cs)
    Gmax = max(Gs)

    out_x = torch.full((len(batch), Cmax, Gmax, Gmax, Gmax), float(pad_value), dtype=dtype, device=device)

    for i, x in enumerate(xs):
        c, g = Cs[i], Gs[i]
        out_x[i, :c, :g, :g, :g] = x

    out: Dict[str, Any] = {"x": out_x, "c": Cs, "g": Gs}

    if "id" in batch[0]:
        out["id"] = [b.get("id") for b in batch]
    if "encoder" in batch[0]:
        out["encoder"] = [b.get("encoder") for b in batch]

    # keep per-sample channels list if present (often stored as list/np array)
    if "channels" in batch[0]:
        out["channels"] = [b.get("channels") for b in batch]

    # keep per-sample spatial metadata if present (cannot stack safely)
    for k in ("voxel_size", "origin", "centroid"):
        if k in batch[0]:
            out[k] = [b.get(k) for b in batch]

    if return_mask:
        mask = torch.zeros((len(batch), 1, Gmax, Gmax, Gmax), dtype=torch.bool, device=device)
        for i, g in enumerate(Gs):
            mask[i, 0, :g, :g, :g] = True
        out["mask"] = mask

    return out

def collate_cloud_pad(
    batch: List[Dict[str, Any]],
    *,
    pad_value_pos: float = 0.0,
    pad_value_x: float = 0.0,
    return_mask: bool = True,
) -> Dict[str, Any]:
    """
    Collate for point clouds with padding over number of points.

    Expects each sample has:
      - pos: (N,3)
    Optional:
      - x: (N,F)

    Returns:
      - pos: (B,Nmax,3)
      - x: (B,Nmax,F) if present
      - n: list[int]
      - mask: (B,Nmax) boolean (True for valid points) if return_mask=True
    """
    poss = [b["pos"] for b in batch]
    device = poss[0].device
    dtype_pos = poss[0].dtype

    ns = [int(p.shape[0]) for p in poss]
    nmax = max(ns)

    out_pos = torch.full((len(batch), nmax, 3), float(pad_value_pos), dtype=dtype_pos, device=device)

    has_x = "x" in batch[0]
    out_x = None
    F = None
    dtype_x = None
    if has_x:
        xs = [b["x"] for b in batch]
        F = int(xs[0].shape[1])
        dtype_x = xs[0].dtype
        out_x = torch.full((len(batch), nmax, F), float(pad_value_x), dtype=dtype_x, device=device)

    for i, b in enumerate(batch):
        n = ns[i]
        out_pos[i, :n, :] = b["pos"]
        if has_x:
            out_x[i, :n, :] = b["x"]

    out: Dict[str, Any] = {"pos": out_pos, "n": ns}

    if has_x and out_x is not None:
        out["x"] = out_x

    if return_mask:
        mask = torch.zeros((len(batch), nmax), dtype=torch.bool, device=device)
        for i, n in enumerate(ns):
            mask[i, :n] = True
        out["mask"] = mask

    if "id" in batch[0]:
        out["id"] = [b.get("id") for b in batch]
    if "encoder" in batch[0]:
        out["encoder"] = [b.get("encoder") for b in batch]

    return out
