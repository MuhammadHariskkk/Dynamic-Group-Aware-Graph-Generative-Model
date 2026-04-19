"""Pedestrian-level group-aware context vector assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GroupContextConfig:
    """Configuration for group context construction."""

    eps: float = 1e-8


def build_group_context_per_timestep(
    positions_t: np.ndarray,
    velocities_t: np.ndarray,
    valid_mask_t: np.ndarray,
    group_ids_t: np.ndarray,
    group_sizes_t: np.ndarray,
    group_consistency_t: np.ndarray,
    group_density_t: np.ndarray,
    group_mean_velocity_t: np.ndarray,
    group_centroids_t: np.ndarray,
    agent_inter_group_conflict_t: np.ndarray,
    config: GroupContextConfig | None = None,
) -> np.ndarray:
    """
    Build per-pedestrian context vectors for one timestep.

    Context composition (dimension 8):
    - intra_group_consistency (1)
    - inter_group_conflict (1)
    - group_density (1)
    - group_size (1)
    - relative_position_to_group_centroid (2)
    - relative_velocity_to_group_mean (2)

    Paper-specified core:
    - uses intra consistency, conflict, density, and group velocity information.
    Engineering assumption:
    - explicit vector layout for encoder/decoder/export integration.
    """
    n = positions_t.shape[0]
    out = np.zeros((n, 8), dtype=np.float32)

    for i in range(n):
        if not bool(valid_mask_t[i]):
            continue
        gid = int(group_ids_t[i])
        if gid < 0 or gid >= group_sizes_t.shape[0]:
            continue

        group_size = float(group_sizes_t[gid])
        group_cons = float(group_consistency_t[gid])
        group_den = float(group_density_t[gid])
        group_vel = group_mean_velocity_t[gid]
        group_ctr = group_centroids_t[gid]

        rel_pos = positions_t[i] - group_ctr
        rel_vel = velocities_t[i] - group_vel
        inter_conf = float(agent_inter_group_conflict_t[i])

        out[i, 0] = group_cons
        out[i, 1] = inter_conf
        out[i, 2] = group_den
        out[i, 3] = group_size
        out[i, 4:6] = rel_pos
        out[i, 6:8] = rel_vel

    return out


def build_group_context_sequence(
    positions: np.ndarray,
    velocities: np.ndarray,
    valid_mask: np.ndarray,
    grouping_output: dict[str, Any],
    config: GroupContextConfig | None = None,
) -> np.ndarray:
    """
    Build context vectors over observed timesteps.

    Args:
        positions: [T_obs, N, 2]
        velocities: [T_obs, N, 2]
        valid_mask: [T_obs, N]
        grouping_output: output from compute_dynamic_group_features(...)
    Returns:
        contexts: [T_obs, N, 8]
    """
    t_obs = positions.shape[0]
    contexts = np.zeros((t_obs, positions.shape[1], 8), dtype=np.float32)

    for t in range(t_obs):
        per_t = grouping_output["per_timestep"][t]
        contexts[t] = build_group_context_per_timestep(
            positions_t=positions[t],
            velocities_t=velocities[t],
            valid_mask_t=valid_mask[t],
            group_ids_t=per_t["group_ids"],
            group_sizes_t=per_t["group_sizes"],
            group_consistency_t=per_t["group_consistency"],
            group_density_t=per_t["group_density"],
            group_mean_velocity_t=per_t["group_mean_velocity"],
            group_centroids_t=per_t["group_centroids"],
            agent_inter_group_conflict_t=per_t["agent_inter_group_conflict"],
            config=config,
        )
    return contexts
