"""ETH/UCY preprocessing utilities for trajectory windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from groupaware.datasets.scene_split import ETH_UCY_SCENES, normalize_scene_name


@dataclass
class PreprocessConfig:
    """Configuration for ETH/UCY preprocessing."""

    raw_root: Path
    processed_root: Path
    obs_len: int = 8
    pred_len: int = 12
    frame_dt: float = 0.4
    min_agents: int = 1

    @property
    def seq_len(self) -> int:
        return self.obs_len + self.pred_len


def _read_eth_ucy_file(file_path: Path) -> pd.DataFrame:
    """
    Read ETH/UCY style trajectory text file.

    Engineering assumptions:
    - Raw files are whitespace-delimited with columns:
      frame_id, pedestrian_id, x, y
    - If extra columns exist, the first four are used.
    """
    df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] < 4:
        raise ValueError(f"Expected at least 4 columns in {file_path}, found {df.shape[1]}")
    df = df.iloc[:, :4].copy()
    df.columns = ["frame_id", "agent_id", "x", "y"]
    df["frame_id"] = df["frame_id"].astype(int)
    df["agent_id"] = df["agent_id"].astype(int)
    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)
    return df.sort_values(["frame_id", "agent_id"]).reset_index(drop=True)


def _candidate_files(scene_dir: Path) -> list[Path]:
    """Return candidate raw trajectory files from a scene directory."""
    patterns = ("*.txt", "*.csv")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(scene_dir.glob(pattern)))
    return files


def _build_sequence_record(
    seq_df: pd.DataFrame,
    scene_id: str,
    sequence_id: int,
    obs_len: int,
    pred_len: int,
    frame_dt: float,
) -> dict[str, Any]:
    """Build one variable-agent sequence record with motion features."""
    frames = sorted(seq_df["frame_id"].unique().tolist())
    seq_len = obs_len + pred_len
    if len(frames) != seq_len:
        raise ValueError("Sequence does not match required length.")

    agents = sorted(seq_df["agent_id"].unique().tolist())
    num_agents = len(agents)
    agent_to_idx = {agent_id: idx for idx, agent_id in enumerate(agents)}
    frame_to_t = {frame_id: t for t, frame_id in enumerate(frames)}

    positions = np.zeros((seq_len, num_agents, 2), dtype=np.float32)
    valid_mask = np.zeros((seq_len, num_agents), dtype=bool)
    frame_index = np.asarray(frames, dtype=np.int64)

    for row in seq_df.itertuples(index=False):
        t = frame_to_t[int(row.frame_id)]
        a = agent_to_idx[int(row.agent_id)]
        positions[t, a, 0] = float(row.x)
        positions[t, a, 1] = float(row.y)
        valid_mask[t, a] = True

    # Relative displacement from previous timestep.
    displacements = np.zeros_like(positions, dtype=np.float32)
    displacements[1:] = positions[1:] - positions[:-1]
    displacements[~valid_mask] = 0.0

    # Paper states velocity as x_t - x_{t-1}, y_t - y_{t-1}.
    # We convert to m/s using known dataset dt=0.4 s from paper.
    velocities = displacements / float(frame_dt)
    velocities[~valid_mask] = 0.0

    speed = np.linalg.norm(velocities, axis=-1)
    headings = np.arctan2(velocities[..., 1], velocities[..., 0]).astype(np.float32)
    headings[(speed <= 1e-8) | (~valid_mask)] = 0.0

    observed = {
        "positions": positions[:obs_len],
        "displacements": displacements[:obs_len],
        "velocities": velocities[:obs_len],
        "headings": headings[:obs_len],
        "valid_mask": valid_mask[:obs_len],
        "frame_index": frame_index[:obs_len],
    }
    future = {
        "positions": positions[obs_len : obs_len + pred_len],
        "displacements": displacements[obs_len : obs_len + pred_len],
        "velocities": velocities[obs_len : obs_len + pred_len],
        "headings": headings[obs_len : obs_len + pred_len],
        "valid_mask": valid_mask[obs_len : obs_len + pred_len],
        "frame_index": frame_index[obs_len : obs_len + pred_len],
    }

    return {
        "scene_id": scene_id,
        "sequence_id": sequence_id,
        "agent_ids": np.asarray(agents, dtype=np.int64),
        "num_agents": num_agents,
        "obs_len": obs_len,
        "pred_len": pred_len,
        "frame_dt": float(frame_dt),
        "observed": observed,
        "future": future,
        # Useful for dynamic grouping (Phase 3), keeping per-timestep states explicit.
        "timestep_states": {
            "positions": positions[:obs_len],
            "velocities": velocities[:obs_len],
            "headings": headings[:obs_len],
            "valid_mask": valid_mask[:obs_len],
            "frame_index": frame_index[:obs_len],
        },
    }


def preprocess_scene(config: PreprocessConfig, scene_name: str) -> Path:
    """Preprocess one scene and persist sequence records as .pt artifact."""
    scene_norm = normalize_scene_name(scene_name)
    if scene_norm not in ETH_UCY_SCENES:
        raise ValueError(f"Unsupported scene: {scene_name}")

    scene_dir = config.raw_root / scene_norm
    if not scene_dir.exists():
        raise FileNotFoundError(f"Raw scene directory not found: {scene_dir}")

    files = _candidate_files(scene_dir)
    if not files:
        raise FileNotFoundError(f"No raw trajectory files found in {scene_dir}")

    frames_all: list[pd.DataFrame] = []
    for file_path in files:
        file_df = _read_eth_ucy_file(file_path)
        file_df["source_file"] = file_path.name
        frames_all.append(file_df)

    df = pd.concat(frames_all, axis=0, ignore_index=True)
    unique_frames = sorted(df["frame_id"].unique().tolist())
    seq_len = config.seq_len

    sequences: list[dict[str, Any]] = []
    sequence_id = 0

    for start in range(0, max(0, len(unique_frames) - seq_len + 1)):
        frame_window = unique_frames[start : start + seq_len]
        seq_df = df[df["frame_id"].isin(frame_window)].copy()

        # Keep agents present at all timesteps for fixed-size sequence tensors.
        counts = seq_df.groupby("agent_id")["frame_id"].nunique()
        full_agents = counts[counts == seq_len].index.tolist()
        if len(full_agents) < config.min_agents:
            continue

        seq_df = seq_df[seq_df["agent_id"].isin(full_agents)]
        seq_df = seq_df.sort_values(["frame_id", "agent_id"]).reset_index(drop=True)
        if seq_df.empty:
            continue

        record = _build_sequence_record(
            seq_df=seq_df,
            scene_id=scene_norm,
            sequence_id=sequence_id,
            obs_len=config.obs_len,
            pred_len=config.pred_len,
            frame_dt=config.frame_dt,
        )
        sequences.append(record)
        sequence_id += 1

    if not sequences:
        raise RuntimeError(f"No valid sequences generated for scene {scene_norm}")

    output_dir = config.processed_root
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{scene_norm}_obs{config.obs_len}_pred{config.pred_len}.pt"
    payload = {
        "scene_id": scene_norm,
        "obs_len": config.obs_len,
        "pred_len": config.pred_len,
        "frame_dt": config.frame_dt,
        "num_sequences": len(sequences),
        "sequences": sequences,
    }
    torch.save(payload, out_path)
    return out_path


def preprocess_all_scenes(config: PreprocessConfig, scenes: list[str] | None = None) -> list[Path]:
    """Preprocess requested scenes and return artifact paths."""
    target_scenes = scenes or list(ETH_UCY_SCENES)
    outputs: list[Path] = []
    for scene in target_scenes:
        outputs.append(preprocess_scene(config=config, scene_name=scene))
    return outputs
