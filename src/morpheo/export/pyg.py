from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import numpy as np

from morpheo.core.types import StructureArtifact
from morpheo.features.rbf import rbf

XMode = Literal[
    "aa_index",
    "onehot",
    "aa_index+plddt",
    "onehot+plddt",
    "onehot+plddt+emb",
    "aa_index+plddt+emb",
]


def _onehot_from_aa_index(aa_index: np.ndarray, num_classes: int = 21) -> np.ndarray:
    aa_index = aa_index.astype(np.int64).reshape(-1)
    out = np.zeros((aa_index.shape[0], num_classes), dtype=np.float32)
    valid = (aa_index >= 0) & (aa_index < num_classes)
    out[np.arange(aa_index.shape[0])[valid], aa_index[valid]] = 1.0
    return out


def to_pyg(
    artifact: StructureArtifact,
    x_mode: XMode = "onehot+plddt",
    rbf_k: int = 16,
    rbf_dmax: float = 20.0,
) :
    """
    Convert a Morpheo artifact into a torch_geometric.data.Data.
    Requires torch + torch_geometric installed.
    """
    try:
        import torch
        from torch_geometric.data import Data
    except Exception as e:
        raise ImportError("PyG export requires morpheo[gdl] (torch + torch_geometric).") from e

    d = artifact.data

    pos = torch.from_numpy(d["pos"]).float()
    edge_index = torch.from_numpy(d["edge_index"]).long()

    # Node features
    aa_index = d.get("aa_index", None)
    plddt = d.get("plddt", None)
    node_emb = d.get("node_emb", None)

    if aa_index is None:
        raise KeyError("Artifact missing 'aa_index' required for x construction.")

    if x_mode.startswith("onehot"):
        x_parts = [_onehot_from_aa_index(aa_index)]
    else:
        x_parts = [aa_index.astype(np.float32).reshape(-1, 1)]

    if "plddt" in x_mode:
        if plddt is None:
            p = np.full((aa_index.shape[0], 1), np.nan, dtype=np.float32)
        else:
            p = plddt.astype(np.float32).reshape(-1, 1)
        x_parts.append(p)

    if "+emb" in x_mode:
        if node_emb is None:
            raise KeyError("x_mode requests embeddings but artifact has no 'node_emb'.")
        x_parts.append(node_emb.astype(np.float32))

    x = torch.from_numpy(np.concatenate(x_parts, axis=1)).float()

    # Edge attributes (RBF from edge_dist if available)
    edge_attr = None
    if "edge_dist" in d:
        ed = d["edge_dist"].astype(np.float32)
        ea = rbf(ed, num_kernels=rbf_k, dmax=rbf_dmax)
        edge_attr = torch.from_numpy(ea).float()

    data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)
    return data


def load_pyg(
    artifact_dir: str | Path,
    x_mode: XMode = "onehot+plddt",
    rbf_k: int = 16,
    rbf_dmax: float = 20.0,
):
    artifact = StructureArtifact.load(Path(artifact_dir))
    return to_pyg(artifact, x_mode=x_mode, rbf_k=rbf_k, rbf_dmax=rbf_dmax)
