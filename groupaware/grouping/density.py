"""Group density features."""

from __future__ import annotations

import numpy as np


def group_density(
    positions_t: np.ndarray,
    members: list[int],
    eps: float = 1e-8,
) -> float:
    """
    Compute density as members per mean radial distance from centroid.

    Engineering assumption:
    - Paper includes group density but does not prescribe exact formula.
      We use a stable geometric proxy suitable for relative comparisons.
    """
    k = len(members)
    if k <= 1:
        return 1.0

    pts = positions_t[members]
    centroid = np.mean(pts, axis=0, keepdims=True)
    radial = np.linalg.norm(pts - centroid, axis=1)
    mean_radial = float(np.mean(radial))
    return float(k / (mean_radial + eps))
