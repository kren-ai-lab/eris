from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


def _torch():
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("morpheo.pooling requires torch installed.") from e
    return torch


def _stats_1d(x, stats: Sequence[str]) -> list:
    """
    x: torch.Tensor with shape (N, F) OR (F,)
    returns list of torch.Tensor (F,) for each requested stat
    """
    torch = _torch()

    if x.ndim == 1:
        x2 = x.view(1, -1)
    elif x.ndim == 2:
        x2 = x
    else:
        raise ValueError(f"Expected x.ndim in {{1,2}}, got {x.ndim}")

    if x2.shape[0] == 0:
        # no rows -> zeros
        F = int(x2.shape[1])
        z = torch.zeros((F,), dtype=x2.dtype, device=x2.device)
        return [z for _ in stats]

    outs: list = []
    for s in stats:
        s = s.lower().strip()
        if s == "mean":
            outs.append(x2.mean(dim=0))
        elif s == "std":
            outs.append(x2.std(dim=0, unbiased=False))
        elif s == "min":
            outs.append(x2.min(dim=0).values)
        elif s == "max":
            outs.append(x2.max(dim=0).values)
        else:
            raise ValueError(f"Unknown stat '{s}'. Use: mean|std|min|max")
    return outs


def pool_graph(
    sample: Dict[str, Any],
    *,
    node_key: str = "x",
    edge_key: str = "edge_attr",
    graph_key: str = "graph_attr",
    stats: Sequence[str] = ("mean", "std", "min", "max"),
    include_edges: bool = True,
    include_graph_attr: bool = True,
) -> Any:
    """
    Pool a single-structure sample dict (from morpheo.export.torch.load_torch)
    into a single global vector (F,).

    - Pools node features from sample[node_key] (N, Fn) using stats
    - Optionally pools edge features from sample[edge_key] (E, Fe) using stats
    - Optionally appends sample[graph_key] (Fg,)

    Returns
    -------
    torch.Tensor
        1D tensor with concatenated features.
    """
    torch = _torch()

    if node_key not in sample:
        raise KeyError(f"sample missing '{node_key}' required for pooling.")
    x = sample[node_key]
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)

    if x.ndim != 2:
        raise ValueError(f"sample['{node_key}'] must be (N,F), got shape {tuple(x.shape)}")

    parts = _stats_1d(x, stats)

    if include_edges and edge_key in sample and sample[edge_key] is not None:
        e = sample[edge_key]
        if not isinstance(e, torch.Tensor):
            e = torch.as_tensor(e)
        if e.ndim != 2:
            raise ValueError(f"sample['{edge_key}'] must be (E,F), got shape {tuple(e.shape)}")
        parts.extend(_stats_1d(e, stats))

    if include_graph_attr and graph_key in sample and sample[graph_key] is not None:
        ga = sample[graph_key]
        if not isinstance(ga, torch.Tensor):
            ga = torch.as_tensor(ga)
        ga = ga.reshape(-1).to(dtype=x.dtype, device=x.device)
        parts.append(ga)

    return torch.cat([p.reshape(-1) for p in parts], dim=0)


def pooled_dim(
    *,
    node_dim: int,
    edge_dim: int = 0,
    graph_dim: int = 0,
    stats: Sequence[str] = ("mean", "std", "min", "max"),
    include_edges: bool = True,
    include_graph_attr: bool = True,
) -> int:
    """
    Compute output dimension without running pooling.
    """
    k = len(stats)
    out = k * node_dim
    if include_edges:
        out += k * edge_dim
    if include_graph_attr:
        out += graph_dim
    return int(out)
