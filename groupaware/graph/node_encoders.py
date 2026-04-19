"""Node feature builders for hybrid pedestrian-group graph."""

from __future__ import annotations

import numpy as np


def build_pedestrian_node_features(
    positions_t: np.ndarray,
    velocities_t: np.ndarray,
) -> np.ndarray:
    """
    Build pedestrian node features f_i^t = [x_i^t, y_i^t, vx_i^t, vy_i^t].

    Paper alignment:
    - Motion state uses position + velocity.
    """
    return np.concatenate([positions_t, velocities_t], axis=-1).astype(np.float32)


def build_group_node_features(
    group_consistency_t: np.ndarray,
    conflict_softmax_t: np.ndarray,
    group_mean_velocity_t: np.ndarray,
    group_density_t: np.ndarray,
) -> np.ndarray:
    """
    Build group node features f_g^t.

    Feature composition:
    - intra-group consistency C_g^t (1)
    - inter-group conflict summary \hat{Conf}_g^t (1)
    - group mean velocity vel_g^t (2)
    - group density Den_g^t (1)
    Total dimension = 5.

    Engineering assumption:
    - The paper defines pairwise normalized inter-group conflict matrix, but does
      not prescribe a single scalar summary per group node feature. We use rowwise
      max conflict toward any other group.
    """
    g = int(group_consistency_t.shape[0])
    if g == 0:
        return np.zeros((0, 5), dtype=np.float32)

    if conflict_softmax_t.size == 0:
        conflict_summary = np.zeros((g,), dtype=np.float32)
    else:
        conflict_summary = np.max(conflict_softmax_t, axis=1).astype(np.float32)

    return np.concatenate(
        [
            group_consistency_t.reshape(g, 1).astype(np.float32),
            conflict_summary.reshape(g, 1),
            group_mean_velocity_t.astype(np.float32),
            group_density_t.reshape(g, 1).astype(np.float32),
        ],
        axis=1,
    )
