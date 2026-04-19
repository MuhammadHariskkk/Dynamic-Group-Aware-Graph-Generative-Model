"""Dynamic grouping and group feature assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from groupaware.grouping.conflict import build_group_conflict_matrices
from groupaware.grouping.consistency import group_average_consistency
from groupaware.grouping.density import group_density
from groupaware.grouping.graph_partition import connected_components_from_adjacency, membership_from_components
from groupaware.grouping.rules import GroupRuleConfig, build_grouping_adjacency


@dataclass(frozen=True)
class GroupFeatureConfig:
    """Configuration for group feature extraction."""

    distance_threshold_m: float = 1.0
    velocity_diff_threshold_mps: float = 0.2
    directional_coherence_threshold: float = 0.95
    eps: float = 1e-8

    def to_rule_config(self) -> GroupRuleConfig:
        """Convert to rule config."""
        return GroupRuleConfig(
            distance_threshold_m=self.distance_threshold_m,
            velocity_diff_threshold_mps=self.velocity_diff_threshold_mps,
            directional_coherence_threshold=self.directional_coherence_threshold,
            eps=self.eps,
        )


def detect_groups_per_timestep(
    positions_t: np.ndarray,
    velocities_t: np.ndarray,
    valid_mask_t: np.ndarray,
    cfg: GroupFeatureConfig,
) -> dict[str, Any]:
    """Detect groups for one timestep and return deterministic memberships."""
    adjacency = build_grouping_adjacency(
        positions_t=positions_t,
        velocities_t=velocities_t,
        valid_mask_t=valid_mask_t,
        cfg=cfg.to_rule_config(),
    )
    components = connected_components_from_adjacency(adjacency)
    group_ids, group_to_members = membership_from_components(components, valid_mask_t=valid_mask_t)
    return {
        "adjacency": adjacency,
        "components": components,
        "group_ids": group_ids,
        "group_to_members": group_to_members,
    }


def compute_group_features_per_timestep(
    positions_t: np.ndarray,
    velocities_t: np.ndarray,
    valid_mask_t: np.ndarray,
    cfg: GroupFeatureConfig,
) -> dict[str, Any]:
    """
    Compute timestep-level group features required by the paper.
    """
    detection = detect_groups_per_timestep(positions_t, velocities_t, valid_mask_t, cfg)
    group_to_members: dict[int, list[int]] = detection["group_to_members"]
    num_groups = len(group_to_members)

    if num_groups == 0:
        return {
            **detection,
            "group_sizes": np.zeros((0,), dtype=np.int64),
            "group_consistency": np.zeros((0,), dtype=np.float32),
            "group_density": np.zeros((0,), dtype=np.float32),
            "group_mean_velocity": np.zeros((0, 2), dtype=np.float32),
            "group_centroids": np.zeros((0, 2), dtype=np.float32),
            "conflict_raw": np.zeros((0, 0), dtype=np.float32),
            "conflict_softmax": np.zeros((0, 0), dtype=np.float32),
            "agent_inter_group_conflict": np.zeros((positions_t.shape[0],), dtype=np.float32),
            "agent_intra_group_consistency": np.zeros((positions_t.shape[0],), dtype=np.float32),
        }

    group_ids_sorted = sorted(group_to_members.keys())
    size = np.zeros((num_groups,), dtype=np.int64)
    consistency = np.zeros((num_groups,), dtype=np.float32)
    density = np.zeros((num_groups,), dtype=np.float32)

    for idx, gid in enumerate(group_ids_sorted):
        members = group_to_members[gid]
        size[idx] = len(members)
        consistency[idx] = group_average_consistency(positions_t, velocities_t, members, eps=cfg.eps)
        density[idx] = group_density(positions_t, members, eps=cfg.eps)

    conflict_raw, conflict_softmax, centroids, mean_vel = build_group_conflict_matrices(
        positions_t=positions_t,
        velocities_t=velocities_t,
        group_to_members=group_to_members,
    )

    # Per-agent features for downstream modules and exporter.
    n = positions_t.shape[0]
    agent_inter_conf = np.zeros((n,), dtype=np.float32)
    agent_intra_cons = np.zeros((n,), dtype=np.float32)
    for idx, gid in enumerate(group_ids_sorted):
        members = group_to_members[gid]
        inter_val = float(np.max(conflict_softmax[idx])) if conflict_softmax.size > 0 else 0.0
        intra_val = float(consistency[idx])
        for m in members:
            agent_inter_conf[m] = inter_val
            agent_intra_cons[m] = intra_val

    return {
        **detection,
        "group_sizes": size,
        "group_consistency": consistency,
        "group_density": density,
        "group_mean_velocity": mean_vel.astype(np.float32),
        "group_centroids": centroids.astype(np.float32),
        "conflict_raw": conflict_raw.astype(np.float32),
        "conflict_softmax": conflict_softmax.astype(np.float32),
        "agent_inter_group_conflict": agent_inter_conf,
        "agent_intra_group_consistency": agent_intra_cons,
    }


def compute_dynamic_group_features(
    positions: np.ndarray,
    velocities: np.ndarray,
    valid_mask: np.ndarray,
    cfg: GroupFeatureConfig,
) -> dict[str, Any]:
    """
    Compute dynamic grouping/features over all observed timesteps.

    Args:
        positions: [T_obs, N, 2]
        velocities: [T_obs, N, 2]
        valid_mask: [T_obs, N]
    """
    t_obs, n, _ = positions.shape
    per_timestep: list[dict[str, Any]] = []
    max_groups = 0

    for t in range(t_obs):
        features_t = compute_group_features_per_timestep(
            positions_t=positions[t],
            velocities_t=velocities[t],
            valid_mask_t=valid_mask[t],
            cfg=cfg,
        )
        per_timestep.append(features_t)
        max_groups = max(max_groups, int(features_t["group_sizes"].shape[0]))

    # Stack membership + per-agent feature arrays for easy downstream consumption.
    group_ids = np.stack([entry["group_ids"] for entry in per_timestep], axis=0).astype(np.int64)
    agent_inter_conf = np.stack(
        [entry["agent_inter_group_conflict"] for entry in per_timestep],
        axis=0,
    ).astype(np.float32)
    agent_intra_cons = np.stack(
        [entry["agent_intra_group_consistency"] for entry in per_timestep],
        axis=0,
    ).astype(np.float32)

    # Padded group-level tensors [T_obs, G_max, *]
    group_sizes = np.zeros((t_obs, max_groups), dtype=np.int64)
    group_consistency = np.zeros((t_obs, max_groups), dtype=np.float32)
    group_density_arr = np.zeros((t_obs, max_groups), dtype=np.float32)
    group_mean_velocity = np.zeros((t_obs, max_groups, 2), dtype=np.float32)
    group_centroids = np.zeros((t_obs, max_groups, 2), dtype=np.float32)

    for t, entry in enumerate(per_timestep):
        g = int(entry["group_sizes"].shape[0])
        if g == 0:
            continue
        group_sizes[t, :g] = entry["group_sizes"]
        group_consistency[t, :g] = entry["group_consistency"]
        group_density_arr[t, :g] = entry["group_density"]
        group_mean_velocity[t, :g] = entry["group_mean_velocity"]
        group_centroids[t, :g] = entry["group_centroids"]

    return {
        "group_ids": group_ids,  # [T_obs, N], -1 for invalid
        "agent_inter_group_conflict": agent_inter_conf,  # [T_obs, N]
        "agent_intra_group_consistency": agent_intra_cons,  # [T_obs, N]
        "group_sizes": group_sizes,  # [T_obs, G_max]
        "group_consistency": group_consistency,  # [T_obs, G_max]
        "group_density": group_density_arr,  # [T_obs, G_max]
        "group_mean_velocity": group_mean_velocity,  # [T_obs, G_max, 2]
        "group_centroids": group_centroids,  # [T_obs, G_max, 2]
        "per_timestep": per_timestep,
    }
