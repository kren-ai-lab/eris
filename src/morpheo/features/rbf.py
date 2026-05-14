from __future__ import annotations

import numpy as np


def old_rbf(dist: np.ndarray, num_kernels: int = 16, dmin: float = 0.0, dmax: float = 20.0) -> np.ndarray:
    """
    Radial basis encoding for distances.

    Parameters
    ----------
    dist : (E,) float
    num_kernels : int
    dmin, dmax : float

    Returns
    -------
    (E, K) float32
    """
    dist = dist.astype(np.float32).reshape(-1, 1)
    centers = np.linspace(dmin, dmax, num_kernels, dtype=np.float32).reshape(1, -1)
    gamma = 1.0 / ((centers[0, 1] - centers[0, 0]) ** 2 + 1e-8)
    out = np.exp(-gamma * (dist - centers) ** 2).astype(np.float32)
    return out


def compute_rbf(
    dist: np.ndarray,
    *,
    num_kernels: int = 16,
    dmin: float = 0.0,
    dmax: float = 20.0,
    sigma_scale: float = 1.0,
) -> np.ndarray:
    """
    Radial basis encoding for distances.

    Parameters
    ----------
    dist : (E,) float
    num_kernels : int
    dmin, dmax : float

    Returns
    -------
    (E, K) float32
    """
    dist = np.asarray(dist, dtype=np.float32).reshape(-1, 1)
    centers = np.linspace(dmin, dmax, num_kernels, dtype=np.float32).reshape(1, -1)

    if num_kernels <= 1:
        delta = 1.0
    else:
        delta = centers[0, 1] - centers[0, 0]

    sigma = max(delta * sigma_scale, 1e-8)
    gamma = 1.0 / (2.0 * sigma * sigma)

    return np.exp(-gamma * (dist - centers) ** 2).astype(np.float32)

