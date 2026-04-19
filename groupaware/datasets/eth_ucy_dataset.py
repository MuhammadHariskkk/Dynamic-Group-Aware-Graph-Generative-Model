"""PyTorch Dataset for preprocessed ETH/UCY trajectory windows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class ETHUCYDataset(Dataset):
    """
    Dataset backed by preprocessed `.pt` scene files.

    Each item is a dictionary with variable number of agents and fixed timesteps:
    - observed: [obs_len, num_agents, *]
    - future: [pred_len, num_agents, *]
    """

    def __init__(self, processed_file: str | Path) -> None:
        self.processed_file = Path(processed_file)
        if not self.processed_file.exists():
            raise FileNotFoundError(f"Processed data file not found: {self.processed_file}")

        payload = torch.load(self.processed_file, map_location="cpu")
        self.scene_id: str = str(payload["scene_id"])
        self.obs_len: int = int(payload["obs_len"])
        self.pred_len: int = int(payload["pred_len"])
        self.frame_dt: float = float(payload["frame_dt"])
        self.sequences: list[dict[str, Any]] = list(payload["sequences"])

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.sequences[index]
        # Return shallow copy to protect in-memory source.
        return {
            "scene_id": sample["scene_id"],
            "sequence_id": int(sample["sequence_id"]),
            "agent_ids": sample["agent_ids"].copy(),
            "num_agents": int(sample["num_agents"]),
            "obs_len": int(sample["obs_len"]),
            "pred_len": int(sample["pred_len"]),
            "frame_dt": float(sample["frame_dt"]),
            "observed": {k: v.copy() for k, v in sample["observed"].items()},
            "future": {k: v.copy() for k, v in sample["future"].items()},
            "timestep_states": {k: v.copy() for k, v in sample["timestep_states"].items()},
        }
