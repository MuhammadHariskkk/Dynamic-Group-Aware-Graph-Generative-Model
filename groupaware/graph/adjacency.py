"""Dynamic adjacency matrix construction for hybrid graph."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AdjacencyConfig:
    """Adjacency weighting coefficients and scaling parameters."""

    gamma_spatial: float = 0.25
    gamma_vel: float = 0.25
    gamma_dir: float = 0.25
    gamma_conflict: float = 0.25
    sigma_distance: float = 1.0
    sigma_velocity: float = 1.0
    eps: float = 1e-8


def _row_softmax_excluding_diagonal(matrix: np.ndarray) -> np.ndarray:
    """Row-wise softmax over off-diagonal neighbor set."""
    n = matrix.shape[0]
    out = np.zeros_like(matrix, dtype=np.float32)
    for i in range(n):
        mask = np.ones((n,), dtype=bool)
        mask[i] = False
        row = matrix[i, mask]
        if row.size == 0:
            continue
        max_v = np.max(row)
        exp_v = np.exp(row - max_v)
        denom = np.sum(exp_v) + 1e-8
        out[i, mask] = exp_v / denom
        out[i, i] = 0.0
    return out


def _unit(v: np.ndarray, eps: float) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm <= eps:
        return np.zeros_like(v)
    return v / norm


def build_dynamic_adjacency(
    node_positions: np.ndarray,
    node_velocities: np.ndarray,
    node_types: np.ndarray,
    num_ped_nodes: int,
    conflict_softmax_group: np.ndarray,
    cfg: AdjacencyConfig,
) -> dict[str, np.ndarray]:
    """
    Build raw and normalized adjacency for one timestep.

    Args:
        node_positions: [V, 2]
        node_velocities: [V, 2]
        node_types: [V], 0=pedestrian, 1=group
        num_ped_nodes: number of pedestrian nodes in first block.
        conflict_softmax_group: [G, G], normalized inter-group conflict.
    """
    v = int(node_positions.shape[0])
    raw = np.zeros((v, v), dtype=np.float32)

    for i in range(v):
        for j in range(v):
            if i == j:
                continue

            dist_ij = float(np.linalg.norm(node_positions[i] - node_positions[j]))
            speed_i = float(np.linalg.norm(node_velocities[i]))
            speed_j = float(np.linalg.norm(node_velocities[j]))
            vel_diff = abs(speed_i - speed_j)

            dir_i = _unit(node_velocities[i], cfg.eps)
            dir_j = _unit(node_velocities[j], cfg.eps)
            dir_align = float(np.clip(np.dot(dir_i, dir_j), -1.0, 1.0))

            spatial_term = np.exp(-dist_ij / (cfg.sigma_distance + cfg.eps))
            vel_term = np.exp(-vel_diff / (cfg.sigma_velocity + cfg.eps))
            dir_term = (1.0 + dir_align) / 2.0

            conflict_term = 0.0
            if node_types[i] == 1 and node_types[j] == 1:
                gi = i - num_ped_nodes
                gj = j - num_ped_nodes
                if 0 <= gi < conflict_softmax_group.shape[0] and 0 <= gj < conflict_softmax_group.shape[1]:
                    conflict_term = float(conflict_softmax_group[gi, gj])

            raw[i, j] = (
                cfg.gamma_spatial * spatial_term
                + cfg.gamma_vel * vel_term
                + cfg.gamma_dir * dir_term
                + cfg.gamma_conflict * conflict_term
            )

    normalized = _row_softmax_excluding_diagonal(raw)

    idx = np.arange(v)
    is_group = node_types.astype(bool)
    is_ped = ~is_group
    pp_mask = np.outer(is_ped, is_ped) & (idx[:, None] != idx[None, :])
    pg_mask = np.outer(is_ped, is_group) | np.outer(is_group, is_ped)
    pg_mask &= idx[:, None] != idx[None, :]
    gg_mask = np.outer(is_group, is_group) & (idx[:, None] != idx[None, :])

    return {
        "adjacency_raw": raw,
        "adjacency": normalized,
        "pp_edge_mask": pp_mask,
        "pg_edge_mask": pg_mask,
        "gg_edge_mask": gg_mask,
    }
