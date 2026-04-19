"""Intra-group consistency computations."""

from __future__ import annotations

import numpy as np


def pairwise_consistency(
    pos_i: np.ndarray,
    vel_i: np.ndarray,
    pos_j: np.ndarray,
    vel_j: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Pairwise consistency proxy aligned with paper Eq. (10) intent.

    Paper-specified structure:
    - higher for close distance, aligned directions, and similar speed.
    Engineering note:
    - Equation typesetting in PDF is ambiguous around parentheses.
      We implement a stable interpretation:
      C_ij = 1 - exp(-(dir_dot / (dist + eps)) * (1 / (|speed_i - speed_j| + eps))).
    """
    dist = float(np.linalg.norm(pos_i - pos_j))
    speed_i = float(np.linalg.norm(vel_i))
    speed_j = float(np.linalg.norm(vel_j))

    dir_i = vel_i / (speed_i + eps)
    dir_j = vel_j / (speed_j + eps)
    dir_dot = float(np.clip(np.dot(dir_i, dir_j), -1.0, 1.0))

    score = 1.0 - np.exp(-((dir_dot / (dist + eps)) * (1.0 / (abs(speed_i - speed_j) + eps))))
    return float(np.clip(score, 0.0, 1.0))


def group_average_consistency(
    positions_t: np.ndarray,
    velocities_t: np.ndarray,
    members: list[int],
    eps: float = 1e-8,
) -> float:
    """Average pairwise consistency for one group."""
    k = len(members)
    if k <= 1:
        return 1.0

    scores: list[float] = []
    for i in range(k):
        for j in range(i + 1, k):
            mi = members[i]
            mj = members[j]
            scores.append(
                pairwise_consistency(
                    positions_t[mi],
                    velocities_t[mi],
                    positions_t[mj],
                    velocities_t[mj],
                    eps=eps,
                )
            )
    return float(np.mean(scores)) if scores else 1.0
