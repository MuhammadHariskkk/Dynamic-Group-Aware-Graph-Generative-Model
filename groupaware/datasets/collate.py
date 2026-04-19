"""Batch collation for variable-agent ETH/UCY sequences."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def collate_eth_ucy(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate variable-agent samples to fixed batch tensors.

    Output tensor shapes:
    - observed.positions: [B, obs_len, Amax, 2]
    - observed.displacements: [B, obs_len, Amax, 2]
    - observed.velocities: [B, obs_len, Amax, 2]
    - observed.headings: [B, obs_len, Amax]
    - observed.valid_mask: [B, obs_len, Amax]
    - future.positions: [B, pred_len, Amax, 2]
    - future.valid_mask: [B, pred_len, Amax]
    - agent_padding_mask: [B, Amax] (True for real agents)
    """
    if not batch:
        raise ValueError("Empty batch passed to collate_eth_ucy.")

    bsz = len(batch)
    obs_len = int(batch[0]["obs_len"])
    pred_len = int(batch[0]["pred_len"])
    max_agents = max(int(item["num_agents"]) for item in batch)

    observed = {
        "positions": torch.zeros((bsz, obs_len, max_agents, 2), dtype=torch.float32),
        "displacements": torch.zeros((bsz, obs_len, max_agents, 2), dtype=torch.float32),
        "velocities": torch.zeros((bsz, obs_len, max_agents, 2), dtype=torch.float32),
        "headings": torch.zeros((bsz, obs_len, max_agents), dtype=torch.float32),
        "valid_mask": torch.zeros((bsz, obs_len, max_agents), dtype=torch.bool),
        "frame_index": torch.zeros((bsz, obs_len), dtype=torch.int64),
    }
    future = {
        "positions": torch.zeros((bsz, pred_len, max_agents, 2), dtype=torch.float32),
        "displacements": torch.zeros((bsz, pred_len, max_agents, 2), dtype=torch.float32),
        "velocities": torch.zeros((bsz, pred_len, max_agents, 2), dtype=torch.float32),
        "headings": torch.zeros((bsz, pred_len, max_agents), dtype=torch.float32),
        "valid_mask": torch.zeros((bsz, pred_len, max_agents), dtype=torch.bool),
        "frame_index": torch.zeros((bsz, pred_len), dtype=torch.int64),
    }
    timestep_states = {
        "positions": torch.zeros((bsz, obs_len, max_agents, 2), dtype=torch.float32),
        "velocities": torch.zeros((bsz, obs_len, max_agents, 2), dtype=torch.float32),
        "headings": torch.zeros((bsz, obs_len, max_agents), dtype=torch.float32),
        "valid_mask": torch.zeros((bsz, obs_len, max_agents), dtype=torch.bool),
        "frame_index": torch.zeros((bsz, obs_len), dtype=torch.int64),
    }

    agent_padding_mask = torch.zeros((bsz, max_agents), dtype=torch.bool)

    scene_ids: list[str] = []
    sequence_ids: list[int] = []
    agent_ids: list[torch.Tensor] = []
    num_agents: list[int] = []

    for b_idx, item in enumerate(batch):
        a = int(item["num_agents"])
        scene_ids.append(str(item["scene_id"]))
        sequence_ids.append(int(item["sequence_id"]))
        num_agents.append(a)
        agent_padding_mask[b_idx, :a] = True

        agent_id_tensor = torch.zeros((max_agents,), dtype=torch.int64)
        agent_id_tensor[:a] = torch.from_numpy(item["agent_ids"].astype(np.int64))
        agent_ids.append(agent_id_tensor)

        for key in observed:
            src = item["observed"][key]
            if src.ndim == 1:
                observed[key][b_idx] = torch.from_numpy(src)
            elif src.ndim == 2:
                observed[key][b_idx, :, :a] = torch.from_numpy(src)
            else:
                observed[key][b_idx, :, :a, :] = torch.from_numpy(src)

        for key in future:
            src = item["future"][key]
            if src.ndim == 1:
                future[key][b_idx] = torch.from_numpy(src)
            elif src.ndim == 2:
                future[key][b_idx, :, :a] = torch.from_numpy(src)
            else:
                future[key][b_idx, :, :a, :] = torch.from_numpy(src)

        for key in timestep_states:
            src = item["timestep_states"][key]
            if src.ndim == 1:
                timestep_states[key][b_idx] = torch.from_numpy(src)
            elif src.ndim == 2:
                timestep_states[key][b_idx, :, :a] = torch.from_numpy(src)
            else:
                timestep_states[key][b_idx, :, :a, :] = torch.from_numpy(src)

    return {
        "scene_ids": scene_ids,
        "sequence_ids": torch.tensor(sequence_ids, dtype=torch.int64),
        "agent_ids": torch.stack(agent_ids, dim=0),
        "num_agents": torch.tensor(num_agents, dtype=torch.int64),
        "agent_padding_mask": agent_padding_mask,
        "obs_len": obs_len,
        "pred_len": pred_len,
        "observed": observed,
        "future": future,
        "timestep_states": timestep_states,
    }
