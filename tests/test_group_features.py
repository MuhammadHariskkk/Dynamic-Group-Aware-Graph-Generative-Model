"""Tests for dynamic group feature extraction."""

from __future__ import annotations

import numpy as np

from groupaware.grouping.group_features import GroupFeatureConfig, compute_dynamic_group_features


def test_dynamic_group_features_shapes() -> None:
    t, n = 8, 4
    pos = np.zeros((t, n, 2), dtype=np.float32)
    vel = np.zeros((t, n, 2), dtype=np.float32)
    valid = np.ones((t, n), dtype=bool)
    pos[:, 0] = [0.0, 0.0]
    pos[:, 1] = [0.4, 0.0]
    pos[:, 2] = [4.0, 0.0]
    pos[:, 3] = [4.4, 0.0]
    vel[:, 0] = [1.0, 0.0]
    vel[:, 1] = [1.0, 0.0]
    vel[:, 2] = [-1.0, 0.0]
    vel[:, 3] = [-1.0, 0.0]

    out = compute_dynamic_group_features(pos, vel, valid, GroupFeatureConfig())
    assert out["group_ids"].shape == (t, n)
    assert out["agent_intra_group_consistency"].shape == (t, n)
    assert out["agent_inter_group_conflict"].shape == (t, n)
    assert out["group_sizes"].shape[0] == t
