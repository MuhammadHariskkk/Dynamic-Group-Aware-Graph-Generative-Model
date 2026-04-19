"""Inspect dynamic grouping statistics for ETH/UCY processed samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from groupaware.datasets.eth_ucy_dataset import ETHUCYDataset
from groupaware.grouping.group_features import GroupFeatureConfig, compute_dynamic_group_features
from groupaware.utils.config import load_config
from groupaware.utils.logger import create_logger


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Inspect dynamic group stats from processed ETH/UCY data.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Base config file.")
    parser.add_argument("--scene-config", type=str, default=None, help="Optional scene config overlay.")
    parser.add_argument(
        "--processed-file",
        type=str,
        required=True,
        help="Processed .pt file produced by scripts/preprocess_data.py",
    )
    parser.add_argument("--sequence-index", type=int, default=0, help="Sequence index to inspect.")
    return parser.parse_args()


def _summarize_sequence(group_out: dict[str, np.ndarray], logger_name: str) -> None:
    """Print concise timestep-wise and global group stats."""
    logger = create_logger(logger_name)
    group_ids = group_out["group_ids"]
    t_obs = group_ids.shape[0]

    logger.info("Observed timesteps: %d", t_obs)
    for t in range(t_obs):
        gids = group_ids[t]
        valid = gids[gids >= 0]
        unique_g = np.unique(valid) if valid.size > 0 else np.array([], dtype=np.int64)
        sizes = group_out["group_sizes"][t]
        g_count = int(np.sum(sizes > 0))
        logger.info(
            "t=%02d | groups=%d | members_per_group=%s",
            t,
            g_count,
            sizes[:g_count].tolist(),
        )
        if unique_g.size > 0:
            logger.info(
                "t=%02d | mean_consistency=%.4f | mean_density=%.4f",
                t,
                float(np.mean(group_out["group_consistency"][t, :g_count])),
                float(np.mean(group_out["group_density"][t, :g_count])),
            )

    overall_cons = group_out["group_consistency"][group_out["group_sizes"] > 0]
    overall_den = group_out["group_density"][group_out["group_sizes"] > 0]
    logger.info(
        "Global | groups_total=%d | avg_consistency=%.4f | avg_density=%.4f",
        int(np.sum(group_out["group_sizes"] > 0)),
        float(np.mean(overall_cons)) if overall_cons.size > 0 else 0.0,
        float(np.mean(overall_den)) if overall_den.size > 0 else 0.0,
    )


def main() -> None:
    """Run grouping inspection for one processed sample."""
    args = parse_args()
    config_path = args.scene_config or args.config
    base_config = args.config if args.scene_config else None
    cfg = load_config(config_path, base_config_path=base_config)

    dataset = ETHUCYDataset(Path(args.processed_file))
    if args.sequence_index < 0 or args.sequence_index >= len(dataset):
        raise IndexError(f"sequence-index {args.sequence_index} is out of range [0, {len(dataset)-1}]")

    sample = dataset[args.sequence_index]
    positions = sample["timestep_states"]["positions"]  # [T_obs, N, 2]
    velocities = sample["timestep_states"]["velocities"]  # [T_obs, N, 2]
    valid_mask = sample["timestep_states"]["valid_mask"]  # [T_obs, N]

    group_cfg = GroupFeatureConfig(
        distance_threshold_m=float(cfg["grouping"]["distance_threshold_m"]),
        velocity_diff_threshold_mps=float(cfg["grouping"]["velocity_diff_threshold_mps"]),
        directional_coherence_threshold=float(cfg["grouping"]["directional_coherence_threshold"]),
    )
    group_out = compute_dynamic_group_features(
        positions=positions,
        velocities=velocities,
        valid_mask=valid_mask,
        cfg=group_cfg,
    )

    logger = create_logger("inspect_groups", log_dir=cfg["paths"]["logs_dir"])
    logger.info("scene_id=%s sequence_id=%d", sample["scene_id"], sample["sequence_id"])
    logger.info(
        "obs_len=%d pred_len=%d agents=%d",
        sample["obs_len"],
        sample["pred_len"],
        sample["num_agents"],
    )
    _summarize_sequence(group_out, logger_name="inspect_groups_summary")


if __name__ == "__main__":
    main()
