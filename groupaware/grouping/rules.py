"""Pairwise grouping rules from paper-defined thresholds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroupRuleConfig:
    """Thresholds for dynamic group assignment."""

    distance_threshold_m: float = 1.0
    velocity_diff_threshold_mps: float = 0.2
    directional_coherence_threshold: float = 0.95
    eps: float = 1e-8


def _safe_unit_vector(v: np.ndarray, eps: float) -> np.ndarray:
    """Return unit vector and keep zeros for near-zero speed vectors."""
    norm = float(np.linalg.norm(v))
    if norm <= eps:
        return np.zeros_like(v)
    return v / norm


def pairwise_rule_satisfied(
    pos_i: np.ndarray,
    vel_i: np.ndarray,
    pos_j: np.ndarray,
    vel_j: np.ndarray,
    cfg: GroupRuleConfig,
) -> bool:
    """
    Check paper-specified pairwise group criteria (Eq. 7 style).
    """
    dist = float(np.linalg.norm(pos_i - pos_j))
    vel_diff = float(np.linalg.norm(vel_i - vel_j))
    dir_i = _safe_unit_vector(vel_i, cfg.eps)
    dir_j = _safe_unit_vector(vel_j, cfg.eps)
    dir_dot = float(np.clip(np.dot(dir_i, dir_j), -1.0, 1.0))

    return (
        dist <= cfg.distance_threshold_m
        and vel_diff <= cfg.velocity_diff_threshold_mps
        and dir_dot >= cfg.directional_coherence_threshold
    )


def build_grouping_adjacency(
    positions_t: np.ndarray,
    velocities_t: np.ndarray,
    valid_mask_t: np.ndarray,
    cfg: GroupRuleConfig,
) -> np.ndarray:
    """
    Build undirected binary adjacency for grouping graph at timestep t.

    Engineering assumption:
    - The paper defines pairwise conditions and disjoint groups, but not exact
      partition algorithm. We build a binary graph then use connected components.
    """
    num_agents = int(positions_t.shape[0])
    adjacency = np.zeros((num_agents, num_agents), dtype=bool)

    for i in range(num_agents):
        if not bool(valid_mask_t[i]):
            continue
        adjacency[i, i] = True
        for j in range(i + 1, num_agents):
            if not bool(valid_mask_t[j]):
                continue
            if pairwise_rule_satisfied(
                positions_t[i],
                velocities_t[i],
                positions_t[j],
                velocities_t[j],
                cfg,
            ):
                adjacency[i, j] = True
                adjacency[j, i] = True
    return adjacency
