"""Tests for full integrated model forward."""

from __future__ import annotations

import torch

from groupaware.models.group_aware_model import GroupAwareModelConfig, GroupAwareTrajectoryModel


def test_group_aware_model_forward_shapes() -> None:
    b, t, n = 2, 8, 4
    batch = {
        "observed": {
            "positions": torch.randn(b, t, n, 2),
            "velocities": torch.randn(b, t, n, 2),
            "valid_mask": torch.ones(b, t, n, dtype=torch.bool),
        }
    }
    model = GroupAwareTrajectoryModel(GroupAwareModelConfig())
    out = model(batch)
    assert out["means"].shape == (b, n, 3, 12, 2)
    assert out["stds"].shape == (b, n, 3, 12, 2)
    assert out["mode_probs"].shape == (b, n, 3)
    assert out["deterministic_traj"].shape == (b, n, 12, 2)
