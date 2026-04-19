"""Inter-group conflict modelling and normalization."""

from __future__ import annotations

import numpy as np


def _softmax_rows_excluding_diagonal(matrix: np.ndarray) -> np.ndarray:
    """
    Row-wise softmax over off-diagonal entries only.

    Paper alignment:
    - Eq. (13)-style normalization excludes self-term (n != k).
    """
    if matrix.size == 0:
        return matrix

    k = matrix.shape[0]
    out = np.zeros_like(matrix, dtype=np.float32)
    for i in range(k):
        mask = np.ones((k,), dtype=bool)
        mask[i] = False
        row_vals = matrix[i, mask]
        if row_vals.size == 0:
            continue
        max_v = np.max(row_vals)
        exp_v = np.exp(row_vals - max_v)
        denom = np.sum(exp_v) + 1e-8
        out[i, mask] = exp_v / denom
        out[i, i] = 0.0
    return out


def group_centroid(positions_t: np.ndarray, members: list[int]) -> np.ndarray:
    """Compute centroid for a group."""
    return np.mean(positions_t[members], axis=0)


def group_mean_velocity(velocities_t: np.ndarray, members: list[int]) -> np.ndarray:
    """Compute mean velocity for a group."""
    return np.mean(velocities_t[members], axis=0)


def pair_group_conflict(
    centroid_a: np.ndarray,
    vel_a: np.ndarray,
    centroid_b: np.ndarray,
    vel_b: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Compute directed conflict score inspired by paper Eq. (12).
    """
    disp_ab = centroid_b - centroid_a
    dist = float(np.linalg.norm(disp_ab))
    if dist <= eps:
        dist = eps

    disp_ba = -disp_ab
    norm_ab = disp_ab / dist
    norm_ba = disp_ba / dist

    speed_a = float(np.linalg.norm(vel_a))
    speed_b = float(np.linalg.norm(vel_b))
    if speed_a <= eps and speed_b <= eps:
        return 0.0

    dir_a = vel_a / (speed_a + eps)
    dir_b = vel_b / (speed_b + eps)

    cos_mu = float(np.clip(np.dot(dir_a, norm_ab), -1.0, 1.0))
    cos_theta = float(np.clip(np.dot(dir_b, norm_ba), -1.0, 1.0))

    if cos_mu >= 0.0 and cos_theta >= 0.0:
        return float((speed_a * cos_mu + speed_b * cos_theta) / dist)
    if cos_theta >= 0.0 and cos_mu < 0.0:
        return float((speed_b * cos_theta) / dist)
    if cos_mu >= 0.0 and cos_theta < 0.0:
        return float((speed_a * cos_mu) / dist)
    return 0.0


def build_group_conflict_matrices(
    positions_t: np.ndarray,
    velocities_t: np.ndarray,
    group_to_members: dict[int, list[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build raw and row-softmax conflict matrices, with group centroids/velocities.
    """
    group_ids = sorted(group_to_members.keys())
    k = len(group_ids)
    centroids = np.zeros((k, 2), dtype=np.float32)
    mean_vels = np.zeros((k, 2), dtype=np.float32)

    for idx, gid in enumerate(group_ids):
        members = group_to_members[gid]
        centroids[idx] = group_centroid(positions_t, members)
        mean_vels[idx] = group_mean_velocity(velocities_t, members)

    raw = np.zeros((k, k), dtype=np.float32)
    for i in range(k):
        for j in range(k):
            if i == j:
                raw[i, j] = 0.0
            else:
                raw[i, j] = pair_group_conflict(
                    centroids[i],
                    mean_vels[i],
                    centroids[j],
                    mean_vels[j],
                )

    norm = _softmax_rows_excluding_diagonal(raw)
    return raw, norm.astype(np.float32), centroids, mean_vels
