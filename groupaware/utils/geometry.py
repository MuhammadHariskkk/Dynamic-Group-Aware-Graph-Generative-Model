"""Geometry helpers for trajectory visualization."""

from __future__ import annotations

import numpy as np


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix for [N,2] points."""
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape [N, 2]")
    diff = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def heading_from_velocity(vx: float, vy: float) -> float:
    """Compute heading angle in radians."""
    return float(np.arctan2(vy, vx))


def trajectory_length(traj: np.ndarray) -> float:
    """Return polyline length for trajectory [T,2]."""
    if traj.ndim != 2 or traj.shape[1] != 2:
        raise ValueError("traj must have shape [T, 2]")
    if traj.shape[0] < 2:
        return 0.0
    diff = traj[1:] - traj[:-1]
    return float(np.linalg.norm(diff, axis=-1).sum())
