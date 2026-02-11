from __future__ import annotations

from typing import Any, Dict, List

import torch

def collate_graph(
    batch: List[Dict[str, Any]],
    *,
    pad_x: bool = True,
    pad_edge_attr: bool = True,
    pad_graph_attr: bool = True,
) -> Dict[str, Any]:
    """
    Collate for graphs WITHOUT PyG dependency.

    Supports mixed feature dims by padding:
      - x: pad to Fmax if pad_x=True
      - edge_attr: pad to Femax if pad_edge_attr=True
      - graph_attr: pad to Fgmax if pad_graph_attr=True

    Expects each sample has:
      - x: (N,F)
      - edge_index: (2,E)
    Optional:
      - edge_attr: (E,Fe)
      - edge_dist: (E,)
      - pos: (N,3)
      - graph_attr: (Fg,)
      - graph_attr_keys: list[str]
    """
    xs = []
    edge_indices = []
    edge_attrs = []
    edge_dists = []
    poss = []
    batch_vec = []

    node_offset = 0

    # Max dims (for padding)
    Fmax = max(int(b["x"].shape[1]) for b in batch)

    has_edge_attr_any = any(("edge_attr" in b and b["edge_attr"] is not None) for b in batch)
    Femax = 0
    if has_edge_attr_any:
        Femax = max(
            int(b["edge_attr"].shape[1])
            for b in batch
            if ("edge_attr" in b and b["edge_attr"] is not None)
        )

    has_graph_attr_any = any(("graph_attr" in b and b["graph_attr"] is not None) for b in batch)
    Fgmax = 0
    if has_graph_attr_any:
        Fgmax = max(
            int(b["graph_attr"].reshape(-1).shape[0])
            for b in batch
            if ("graph_attr" in b and b["graph_attr"] is not None)
        )

    for i, b in enumerate(batch):
        x = b["x"]
        ei = b["edge_index"]
        n = int(x.shape[0])

        # --- pad x
        if pad_x and int(x.shape[1]) != Fmax:
            pad = torch.zeros((n, Fmax), dtype=x.dtype, device=x.device)
            pad[:, : int(x.shape[1])] = x
            x = pad
        xs.append(x)

        edge_indices.append(ei + node_offset)

        # --- edge_attr (optional + pad)
        if has_edge_attr_any:
            if "edge_attr" in b and b["edge_attr"] is not None:
                ea = b["edge_attr"]
                e = int(ea.shape[0])
                if pad_edge_attr and int(ea.shape[1]) != Femax:
                    pad_ea = torch.zeros((e, Femax), dtype=ea.dtype, device=ea.device)
                    pad_ea[:, : int(ea.shape[1])] = ea
                    ea = pad_ea
            else:
                # missing -> zeros
                e = int(ei.shape[1])
                ea = torch.zeros((e, Femax), dtype=x.dtype, device=x.device)
            edge_attrs.append(ea)

        if "edge_dist" in b and b["edge_dist"] is not None:
            edge_dists.append(b["edge_dist"])
        if "pos" in b and b["pos"] is not None:
            poss.append(b["pos"])

        batch_vec.append(torch.full((n,), i, dtype=torch.long, device=x.device))
        node_offset += n

    out: Dict[str, Any] = {
        "x": torch.cat(xs, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "batch": torch.cat(batch_vec, dim=0),
    }

    if has_edge_attr_any:
        out["edge_attr"] = torch.cat(edge_attrs, dim=0)

    if edge_dists:
        out["edge_dist"] = torch.cat(edge_dists, dim=0)

    if poss:
        out["pos"] = torch.cat(poss, dim=0)

    # --- graph_attr (optional + pad)
    if has_graph_attr_any:
        gas = []
        for b in batch:
            if "graph_attr" in b and b["graph_attr"] is not None:
                ga = b["graph_attr"].reshape(-1)
                if pad_graph_attr and int(ga.numel()) != Fgmax:
                    pad_ga = torch.zeros((Fgmax,), dtype=ga.dtype, device=ga.device)
                    pad_ga[: int(ga.numel())] = ga
                    ga = pad_ga
            else:
                ga = torch.zeros((Fgmax,), dtype=out["x"].dtype, device=out["x"].device)
            gas.append(ga.reshape(1, -1))
        out["graph_attr"] = torch.cat(gas, dim=0)

        if "graph_attr_keys" in batch[0] and batch[0]["graph_attr_keys"] is not None:
            out["graph_attr_keys"] = batch[0]["graph_attr_keys"]

    if "id" in batch[0]:
        out["id"] = [b.get("id") for b in batch]
    if "encoder" in batch[0]:
        out["encoder"] = [b.get("encoder") for b in batch]

    return out


def collate_graph_strict(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Strict collate: requires identical feature dimensions across samples.
    Useful for training where you want a fixed schema and no padding.
    """
    # x dim check
    F0 = int(batch[0]["x"].shape[1])
    for j, b in enumerate(batch[1:], start=1):
        Fj = int(b["x"].shape[1])
        if Fj != F0:
            raise RuntimeError(f"collate_graph_strict: x feature dim mismatch: batch[0]={F0}, batch[{j}]={Fj}")

    # edge_attr dim check (only if present in any)
    has_ea_any = any(("edge_attr" in b and b["edge_attr"] is not None) for b in batch)
    if has_ea_any:
        # require all to have it and same Fe
        if not all(("edge_attr" in b and b["edge_attr"] is not None) for b in batch):
            raise RuntimeError("collate_graph_strict: some samples missing edge_attr while others have it.")
        Fe0 = int(batch[0]["edge_attr"].shape[1])
        for j, b in enumerate(batch[1:], start=1):
            Fej = int(b["edge_attr"].shape[1])
            if Fej != Fe0:
                raise RuntimeError(f"collate_graph_strict: edge_attr dim mismatch: batch[0]={Fe0}, batch[{j}]={Fej}")

    # graph_attr dim check (only if present in any)
    has_ga_any = any(("graph_attr" in b and b["graph_attr"] is not None) for b in batch)
    if has_ga_any:
        if not all(("graph_attr" in b and b["graph_attr"] is not None) for b in batch):
            raise RuntimeError("collate_graph_strict: some samples missing graph_attr while others have it.")
        G0 = int(batch[0]["graph_attr"].reshape(-1).numel())
        for j, b in enumerate(batch[1:], start=1):
            Gj = int(b["graph_attr"].reshape(-1).numel())
            if Gj != G0:
                raise RuntimeError(f"collate_graph_strict: graph_attr dim mismatch: batch[0]={G0}, batch[{j}]={Gj}")

    # If all ok, just call padded collate with padding disabled
    return collate_graph(batch, pad_x=False, pad_edge_attr=False, pad_graph_attr=False)

def collate_grid(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate for voxel grids.
    Expects each sample has:
      - x: (C,G,G,G)
    Optional:
      - graph_attr: (Fg,)
      - graph_attr_keys: list[str]
    """

    x0 = batch[0]["x"]
    for j, b in enumerate(batch[1:], start=1):
        if tuple(b["x"].shape) != tuple(x0.shape):
            raise RuntimeError(
                f"collate_grid: grid shape mismatch at batch[{j}]: "
                f"{tuple(x0.shape)} vs {tuple(b['x'].shape)}. "
                "Use collate_grid_pad for mixed grid shapes/channels."
            )
        
    x = torch.stack([b["x"] for b in batch], dim=0)  # (B,C,G,G,G)
    out: Dict[str, Any] = {"x": x}

    # graph-level features (stack)
    if "graph_attr" in batch[0] and batch[0]["graph_attr"] is not None:
        if all(("graph_attr" in b and b["graph_attr"] is not None) for b in batch):
            out["graph_attr"] = torch.cat([b["graph_attr"].reshape(1, -1) for b in batch], dim=0)
            if "graph_attr_keys" in batch[0] and batch[0]["graph_attr_keys"] is not None:
                out["graph_attr_keys"] = batch[0]["graph_attr_keys"]

    # keep ids if present
    if "id" in batch[0]:
        out["id"] = [b.get("id") for b in batch]
    if "encoder" in batch[0]:
        out["encoder"] = [b.get("encoder") for b in batch]

    # optional: carry spatial meta (assumes consistent per-batch)
    for k in ("voxel_size", "origin", "centroid"):
        if k in batch[0]:
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
    Optional:
      - graph_attr: (Fg,) stacked to (B,Fg)
      - graph_attr_keys: list[str]
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
        x2 = x if x.ndim == 2 else x[0]
        n = ns[i]
        out_x[i, 0, :n, :n] = x2

    out: Dict[str, Any] = {
        "x": out_x,
        "n": ns,
        "matrix_key": keys,
    }

    # graph-level features (stack)
    if "graph_attr" in batch[0] and batch[0]["graph_attr"] is not None:
        if all(("graph_attr" in b and b["graph_attr"] is not None) for b in batch):
            out["graph_attr"] = torch.cat([b["graph_attr"].reshape(1, -1) for b in batch], dim=0)
            if "graph_attr_keys" in batch[0] and batch[0]["graph_attr_keys"] is not None:
                out["graph_attr_keys"] = batch[0]["graph_attr_keys"]

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
    Optional:
      - graph_attr: (Fg,) stacked to (B,Fg)
      - graph_attr_keys: list[str]
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

    # graph-level features (stack)
    if "graph_attr" in batch[0] and batch[0]["graph_attr"] is not None:
        if all(("graph_attr" in b and b["graph_attr"] is not None) for b in batch):
            out["graph_attr"] = torch.cat([b["graph_attr"].reshape(1, -1) for b in batch], dim=0)
            if "graph_attr_keys" in batch[0] and batch[0]["graph_attr_keys"] is not None:
                out["graph_attr_keys"] = batch[0]["graph_attr_keys"]

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
      - graph_attr: (Fg,) stacked to (B,Fg)
      - graph_attr_keys: list[str]

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

    has_x = ("x" in batch[0]) and (batch[0]["x"] is not None)
    out_x = None
    if has_x:
        xs = [b["x"] for b in batch]
        F = int(xs[0].shape[1])
        dtype_x = xs[0].dtype
        out_x = torch.full((len(batch), nmax, F), float(pad_value_x), dtype=dtype_x, device=device)

    for i, b in enumerate(batch):
        n = ns[i]
        out_pos[i, :n, :] = b["pos"]
        if has_x and out_x is not None:
            out_x[i, :n, :] = b["x"]

    out: Dict[str, Any] = {"pos": out_pos, "n": ns}

    if has_x and out_x is not None:
        out["x"] = out_x

    if return_mask:
        mask = torch.zeros((len(batch), nmax), dtype=torch.bool, device=device)
        for i, n in enumerate(ns):
            mask[i, :n] = True
        out["mask"] = mask

    # graph-level features (stack)
    if "graph_attr" in batch[0] and batch[0]["graph_attr"] is not None:
        if all(("graph_attr" in b and b["graph_attr"] is not None) for b in batch):
            out["graph_attr"] = torch.cat([b["graph_attr"].reshape(1, -1) for b in batch], dim=0)
            if "graph_attr_keys" in batch[0] and batch[0]["graph_attr_keys"] is not None:
                out["graph_attr_keys"] = batch[0]["graph_attr_keys"]

    if "id" in batch[0]:
        out["id"] = [b.get("id") for b in batch]
    if "encoder" in batch[0]:
        out["encoder"] = [b.get("encoder") for b in batch]

    return out
