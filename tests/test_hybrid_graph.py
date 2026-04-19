"""Tests for hybrid graph construction."""

from __future__ import annotations

import numpy as np

from groupaware.graph.hybrid_graph import HybridGraphConfig, build_hybrid_graph_sequence
from groupaware.grouping.group_features import GroupFeatureConfig, compute_dynamic_group_features


def test_hybrid_graph_basic_shapes() -> None:
    t, n = 8, 3
    pos = np.zeros((t, n, 2), dtype=np.float32)
    vel = np.zeros((t, n, 2), dtype=np.float32)
    valid = np.ones((t, n), dtype=bool)
    pos[:, 0] = [0.0, 0.0]
    pos[:, 1] = [0.5, 0.0]
    pos[:, 2] = [4.0, 0.0]
    vel[:, 0] = [1.0, 0.0]
    vel[:, 1] = [1.0, 0.0]
    vel[:, 2] = [0.0, 1.0]

    g = compute_dynamic_group_features(pos, vel, valid, GroupFeatureConfig())
    hg = build_hybrid_graph_sequence(pos, vel, valid, g, HybridGraphConfig())
    t0 = hg["per_timestep"][0]
    v = t0["node_features"].shape[0]
    assert t0["adjacency"].shape == (v, v)
    assert t0["node_positions"].shape[0] == v
