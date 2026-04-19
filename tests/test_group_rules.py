"""Tests for grouping rule logic."""

from __future__ import annotations

import numpy as np

from groupaware.grouping.rules import GroupRuleConfig, build_grouping_adjacency, pairwise_rule_satisfied


def test_pairwise_rule_true_for_close_aligned_agents() -> None:
    cfg = GroupRuleConfig(distance_threshold_m=1.0, velocity_diff_threshold_mps=0.2, directional_coherence_threshold=0.9)
    ok = pairwise_rule_satisfied(
        np.array([0.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.5, 0.0], dtype=np.float32),
        np.array([1.1, 0.0], dtype=np.float32),
        cfg,
    )
    assert ok


def test_grouping_adjacency_respects_thresholds() -> None:
    cfg = GroupRuleConfig()
    pos = np.array([[0.0, 0.0], [0.6, 0.0], [3.0, 0.0]], dtype=np.float32)
    vel = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    valid = np.array([True, True, True])
    adj = build_grouping_adjacency(pos, vel, valid, cfg)
    assert adj[0, 1] and adj[1, 0]
    assert not adj[0, 2]
