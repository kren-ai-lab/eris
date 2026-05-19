import numpy as np

from typing import Any, Dict


def validate_pos(data: Dict[str, Any]):
    if "pos" not in data:
        raise ValueError("UniversalContract: Missing 'pos'")
    
    if not isinstance(data["pos"], np.ndarray):
        raise TypeError("UniversalContract: 'pos' must be np.ndarray")
    
    if data["pos"].ndim != 2 or data["pos"].shape[1] != 3:
        raise ValueError("UniversalContract: 'pos' must have shape (N, 3)")
    
    if data["pos"].dtype != np.float32:
        raise TypeError("UniversalContract: 'pos' items must be np.float32")
    
    if not np.isfinite(data["pos"]).all():
        raise TypeError("UniversalContract: 'pos' items must be finite")  


def validate_node_feature(data: Dict[str, Any], key: str):
    if key not in data:
        return

    arr = data[key]
    pos = data["pos"]

    if arr.shape[0] != pos.shape[0]:
        raise ValueError(
            f"'{key}' must have same length as number of nodes"
        )
    

def validate_edge_index(data: Dict[str, Any]):
    if "edge_index" not in data:
        raise ValueError("GraphContract: falta 'edge_index'")

    edge_index = data["edge_index"]

    # Shape (2, E)
    if not isinstance(edge_index, np.ndarray):
        raise TypeError("GraphContract: 'edge_index' debe ser np.ndarray")

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            "GraphContract: 'edge_index' debe tener shape (2, E)"
        )

    # Dtype
    if edge_index.dtype != np.int64:
        raise TypeError(
            "GraphContract: 'edge_index' debe ser int64"
        )
    
    num_nodes = data["pos"].shape[0]

    if edge_index.size > 0:
        if edge_index.min() < 0 or edge_index.max() >= num_nodes:
            raise ValueError(
                "GraphContract: edge_index contiene índices fuera de rango"
            )
    

def validate_edge_features(data: Dict[str, Any], key: str = "edge_attr"):
    if key not in data:
        return

    edge_attr = data[key]
    edge_index = data["edge_index"]

    if not isinstance(edge_attr, np.ndarray):
        raise TypeError(f"{key} must be np.ndarray")

    if edge_attr.ndim != 2:
        raise ValueError(f"{key} must be 2D (E, D)")

    if edge_attr.shape[0] != edge_index.shape[1]:
        raise ValueError(f"{key} must have length E")

    if not np.isfinite(edge_attr).all():
        raise ValueError(f"{key} must contain finite values")

