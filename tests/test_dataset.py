"""Tests for ETH/UCY dataset loading and collate schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from groupaware.datasets import ETHUCYDataset, collate_eth_ucy


def _make_sequence(seq_id: int, n: int = 3, obs_len: int = 8, pred_len: int = 12) -> dict:
    t = obs_len + pred_len
    positions = np.random.randn(t, n, 2).astype(np.float32)
    displacements = np.zeros_like(positions)
    displacements[1:] = positions[1:] - positions[:-1]
    velocities = displacements / 0.4
    headings = np.arctan2(velocities[..., 1], velocities[..., 0]).astype(np.float32)
    valid = np.ones((t, n), dtype=bool)
    frames = np.arange(100 + seq_id * t, 100 + (seq_id + 1) * t, dtype=np.int64)
    return {
        "scene_id": "eth",
        "sequence_id": seq_id,
        "agent_ids": np.arange(1, n + 1, dtype=np.int64),
        "num_agents": n,
        "obs_len": obs_len,
        "pred_len": pred_len,
        "frame_dt": 0.4,
        "observed": {
            "positions": positions[:obs_len],
            "displacements": displacements[:obs_len],
            "velocities": velocities[:obs_len],
            "headings": headings[:obs_len],
            "valid_mask": valid[:obs_len],
            "frame_index": frames[:obs_len],
        },
        "future": {
            "positions": positions[obs_len:],
            "displacements": displacements[obs_len:],
            "velocities": velocities[obs_len:],
            "headings": headings[obs_len:],
            "valid_mask": valid[obs_len:],
            "frame_index": frames[obs_len:],
        },
        "timestep_states": {
            "positions": positions[:obs_len],
            "velocities": velocities[:obs_len],
            "headings": headings[:obs_len],
            "valid_mask": valid[:obs_len],
            "frame_index": frames[:obs_len],
        },
    }


def test_dataset_and_collate_shapes(tmp_path: Path) -> None:
    payload = {
        "scene_id": "eth",
        "obs_len": 8,
        "pred_len": 12,
        "frame_dt": 0.4,
        "num_sequences": 2,
        "sequences": [_make_sequence(0, n=3), _make_sequence(1, n=2)],
    }
    p = tmp_path / "eth_obs8_pred12.pt"
    torch.save(payload, p)

    ds = ETHUCYDataset(p)
    assert len(ds) == 2
    dl = DataLoader(ds, batch_size=2, collate_fn=collate_eth_ucy, shuffle=False)
    batch = next(iter(dl))

    assert batch["observed"]["positions"].shape == (2, 8, 3, 2)
    assert batch["future"]["positions"].shape == (2, 12, 3, 2)
    assert batch["timestep_states"]["velocities"].shape == (2, 8, 3, 2)
    assert batch["agent_padding_mask"].shape == (2, 3)
