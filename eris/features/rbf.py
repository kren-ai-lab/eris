"""Radial basis distance encoding."""

from __future__ import annotations

import numpy as np


def compute_rbf(
    dist: np.ndarray,
    *,
    num_kernels: int = 16,
    dmin: float = 0.0,
    dmax: float = 20.0,
    sigma_scale: float = 1.0,
) -> np.ndarray:
    """Encode distances with Gaussian radial basis kernels.

    Args:
        dist: Distances, shape ``(E,)``.
        num_kernels: Number of kernels ``K``.
        dmin: Center of the first kernel.
        dmax: Center of the last kernel.
        sigma_scale: Multiplier on the kernel spacing used as the width.

    Returns:
        Float32 array of shape ``(E, K)``.

    """
    dist = np.asarray(dist, dtype=np.float32).reshape(-1, 1)
    centers = np.linspace(dmin, dmax, num_kernels, dtype=np.float32).reshape(1, -1)

    delta = 1.0 if num_kernels <= 1 else centers[0, 1] - centers[0, 0]

    sigma = max(delta * sigma_scale, 1e-8)
    gamma = 1.0 / (2.0 * sigma * sigma)

    return np.exp(-gamma * (dist - centers) ** 2).astype(np.float32)
