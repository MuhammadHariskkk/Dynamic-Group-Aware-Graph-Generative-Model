"""Tests for visualization export schema and writers."""

from __future__ import annotations

from pathlib import Path

import torch

from groupaware.exporters import build_export_package, save_visualizer_files
from groupaware.exporters.schema import EXPORT_COLUMNS


def _fake_batch() -> dict:
    b, t_obs, t_pred, n = 1, 8, 12, 2
    return {
        "scene_ids": ["eth"],
        "sequence_ids": torch.tensor([1]),
        "agent_ids": torch.tensor([[10, 11]]),
        "num_agents": torch.tensor([2]),
        "observed": {
            "positions": torch.zeros(b, t_obs, n, 2),
            "velocities": torch.zeros(b, t_obs, n, 2),
            "frame_index": torch.arange(t_obs).view(1, t_obs),
        },
        "future": {
            "positions": torch.zeros(b, t_pred, n, 2),
            "velocities": torch.zeros(b, t_pred, n, 2),
            "frame_index": torch.arange(t_obs, t_obs + t_pred).view(1, t_pred),
        },
    }


def _fake_outputs() -> dict:
    b, n, m, t = 1, 2, 3, 12
    return {
        "deterministic_traj": torch.zeros(b, n, t, 2),
        "means": torch.zeros(b, n, m, t, 2),
        "mode_probs": torch.softmax(torch.randn(b, n, m), dim=-1),
        "group_metadata": [{}],
    }


def test_export_schema_and_save(tmp_path: Path) -> None:
    df = build_export_package(_fake_batch(), _fake_outputs(), include_observed=True, include_ground_truth=True, include_predictions=True)
    for c in EXPORT_COLUMNS:
        assert c in df.columns
    outs = save_visualizer_files(df, tmp_path, run_name="t", export_format="csv", one_file_per_run=True, one_file_per_scene=True)
    assert len(outs) >= 1
    for p in outs:
        assert p.exists()
