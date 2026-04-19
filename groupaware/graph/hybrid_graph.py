"""Hybrid graph assembly per timestep and over observed sequence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from groupaware.graph.adjacency import AdjacencyConfig, build_dynamic_adjacency
from groupaware.graph.node_encoders import build_group_node_features, build_pedestrian_node_features


@dataclass(frozen=True)
class HybridGraphConfig:
    """Configuration for hybrid graph construction."""

    gamma_spatial: float = 0.25
    gamma_vel: float = 0.25
    gamma_dir: float = 0.25
    gamma_conflict: float = 0.25
    sigma_distance: float = 1.0
    sigma_velocity: float = 1.0

    def to_adjacency_config(self) -> AdjacencyConfig:
        """Convert to adjacency config."""
        return AdjacencyConfig(
            gamma_spatial=self.gamma_spatial,
            gamma_vel=self.gamma_vel,
            gamma_dir=self.gamma_dir,
            gamma_conflict=self.gamma_conflict,
            sigma_distance=self.sigma_distance,
            sigma_velocity=self.sigma_velocity,
        )


def build_hybrid_graph_timestep(
    positions_t: np.ndarray,
    velocities_t: np.ndarray,
    valid_mask_t: np.ndarray,
    grouping_t: dict[str, Any],
    config: HybridGraphConfig,
) -> dict[str, Any]:
    """
    Build one timestep hybrid graph with deterministic node ordering.

    Node ordering:
    1) pedestrian nodes (valid agents only), ascending original agent index
    2) group nodes, ascending group id
    """
    ped_indices = np.where(valid_mask_t)[0].astype(np.int64)
    ped_positions = positions_t[ped_indices]
    ped_velocities = velocities_t[ped_indices]
    ped_features = build_pedestrian_node_features(ped_positions, ped_velocities)

    group_sizes_all = grouping_t["group_sizes"]
    group_consistency_all = grouping_t["group_consistency"]
    group_density_all = grouping_t["group_density"]
    group_mean_velocity_all = grouping_t["group_mean_velocity"]
    group_centroids_all = grouping_t["group_centroids"]
    group_to_members = grouping_t["group_to_members"]
    group_ids_sorted = sorted(group_to_members.keys())
    g = len(group_ids_sorted)

    if g > 0:
        group_consistency = group_consistency_all[:g]
        group_density = group_density_all[:g]
        group_mean_velocity = group_mean_velocity_all[:g]
        group_centroids = group_centroids_all[:g]
        conflict_softmax = grouping_t["conflict_softmax"][:g, :g]
    else:
        group_consistency = np.zeros((0,), dtype=np.float32)
        group_density = np.zeros((0,), dtype=np.float32)
        group_mean_velocity = np.zeros((0, 2), dtype=np.float32)
        group_centroids = np.zeros((0, 2), dtype=np.float32)
        conflict_softmax = np.zeros((0, 0), dtype=np.float32)

    group_features = build_group_node_features(
        group_consistency_t=group_consistency,
        conflict_softmax_t=conflict_softmax,
        group_mean_velocity_t=group_mean_velocity,
        group_density_t=group_density,
    )

    # Unify pedestrian/group feature dimensionality for a single node matrix.
    # Engineering assumption: pad smaller feature block with zeros.
    feature_dim = max(
        ped_features.shape[1] if ped_features.size > 0 else 0,
        group_features.shape[1] if group_features.size > 0 else 0,
    )
    ped_feat_pad = np.zeros((ped_features.shape[0], feature_dim), dtype=np.float32)
    if ped_features.shape[0] > 0:
        ped_feat_pad[:, : ped_features.shape[1]] = ped_features

    group_feat_pad = np.zeros((group_features.shape[0], feature_dim), dtype=np.float32)
    if group_features.shape[0] > 0:
        group_feat_pad[:, : group_features.shape[1]] = group_features

    node_features = np.concatenate([ped_feat_pad, group_feat_pad], axis=0).astype(np.float32)
    node_positions = np.concatenate([ped_positions, group_centroids], axis=0).astype(np.float32)
    node_velocities = np.concatenate([ped_velocities, group_mean_velocity], axis=0).astype(np.float32)

    node_types = np.zeros((node_features.shape[0],), dtype=np.int64)  # 0=ped, 1=group
    if g > 0:
        node_types[len(ped_indices) :] = 1

    adjacency_out = build_dynamic_adjacency(
        node_positions=node_positions,
        node_velocities=node_velocities,
        node_types=node_types,
        num_ped_nodes=len(ped_indices),
        conflict_softmax_group=conflict_softmax,
        cfg=config.to_adjacency_config(),
    )

    ped_to_group = np.full((len(ped_indices),), -1, dtype=np.int64)
    local_ped_map = {orig_idx: local_idx for local_idx, orig_idx in enumerate(ped_indices.tolist())}
    for gid in group_ids_sorted:
        members = group_to_members[gid]
        for orig_idx in members:
            if orig_idx in local_ped_map:
                ped_to_group[local_ped_map[orig_idx]] = gid

    return {
        "node_features": node_features,  # [V, F], F=4 for ped, 5 for group (padded by concatenation consumer)
        "node_positions": node_positions,  # [V, 2]
        "node_velocities": node_velocities,  # [V, 2]
        "node_types": node_types,  # [V], 0 ped / 1 group
        "num_ped_nodes": len(ped_indices),
        "num_group_nodes": g,
        "ped_original_indices": ped_indices,  # [P]
        "group_ids": np.asarray(group_ids_sorted, dtype=np.int64),  # [G]
        "ped_to_group": ped_to_group,  # [P]
        "group_sizes": group_sizes_all[:g].astype(np.int64),
        "group_consistency": group_consistency.astype(np.float32),
        "group_density": group_density.astype(np.float32),
        "group_mean_velocity": group_mean_velocity.astype(np.float32),
        "group_centroids": group_centroids.astype(np.float32),
        "conflict_softmax": conflict_softmax.astype(np.float32),
        **adjacency_out,
    }


def build_hybrid_graph_sequence(
    positions: np.ndarray,
    velocities: np.ndarray,
    valid_mask: np.ndarray,
    grouping_output: dict[str, Any],
    config: HybridGraphConfig,
) -> dict[str, Any]:
    """
    Build hybrid graph for all observed timesteps.

    Args:
        positions: [T_obs, N, 2]
        velocities: [T_obs, N, 2]
        valid_mask: [T_obs, N]
        grouping_output: output from compute_dynamic_group_features(...)
    """
    t_obs = positions.shape[0]
    per_timestep = []
    for t in range(t_obs):
        graph_t = build_hybrid_graph_timestep(
            positions_t=positions[t],
            velocities_t=velocities[t],
            valid_mask_t=valid_mask[t],
            grouping_t=grouping_output["per_timestep"][t],
            config=config,
        )
        per_timestep.append(graph_t)

    return {
        "per_timestep": per_timestep,
        "obs_len": t_obs,
    }
