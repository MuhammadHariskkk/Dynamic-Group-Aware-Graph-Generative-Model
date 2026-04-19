"""CLI for preprocessing ETH/UCY trajectory data."""

from __future__ import annotations

import argparse
from pathlib import Path

from groupaware.datasets.preprocessing import PreprocessConfig, preprocess_all_scenes
from groupaware.datasets.scene_split import ETH_UCY_SCENES
from groupaware.utils.config import load_config
from groupaware.utils.logger import create_logger
from groupaware.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess ETH/UCY trajectories.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Base config path.")
    parser.add_argument("--scene-config", type=str, default=None, help="Optional scene config overlay.")
    parser.add_argument(
        "--scene",
        type=str,
        default="all",
        help="Scene name (eth/hotel/univ/zara1/zara2) or 'all'.",
    )
    parser.add_argument("--raw-root", type=str, default=None, help="Override raw root directory.")
    parser.add_argument("--processed-root", type=str, default=None, help="Override processed root directory.")
    parser.add_argument("--obs-len", type=int, default=None, help="Override observation length.")
    parser.add_argument("--pred-len", type=int, default=None, help="Override prediction length.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Config override with dotpath, e.g. data.obs_len=8",
    )
    return parser.parse_args()


def main() -> None:
    """Run preprocessing for selected ETH/UCY scenes."""
    args = parse_args()
    config_path = args.scene_config or args.config
    base_config = args.config if args.scene_config else None
    cfg = load_config(config_path, base_config_path=base_config, cli_overrides=args.override)

    logger = create_logger("preprocess_data", log_dir=cfg["paths"]["logs_dir"])
    set_seed(int(cfg["project"]["seed"]))

    raw_root = Path(args.raw_root or cfg["data"]["raw_root"])
    processed_root = Path(args.processed_root or cfg["data"]["processed_root"])
    obs_len = int(args.obs_len or cfg["data"]["obs_len"])
    pred_len = int(args.pred_len or cfg["data"]["pred_len"])

    # Paper-specified ETH/UCY timestep is 0.4 seconds.
    # We derive frame_dt from config frame_rate_hz if provided.
    frame_rate_hz = float(cfg["data"].get("frame_rate_hz", 2.5))
    frame_dt = 1.0 / frame_rate_hz

    target_scene = args.scene.lower()
    scenes = list(ETH_UCY_SCENES) if target_scene == "all" else [target_scene]

    pp_cfg = PreprocessConfig(
        raw_root=raw_root,
        processed_root=processed_root,
        obs_len=obs_len,
        pred_len=pred_len,
        frame_dt=frame_dt,
        min_agents=1,
    )
    logger.info("Starting preprocessing for scenes: %s", scenes)
    logger.info("obs_len=%d pred_len=%d frame_dt=%.3f", obs_len, pred_len, frame_dt)

    outputs = preprocess_all_scenes(config=pp_cfg, scenes=scenes)
    for out_path in outputs:
        logger.info("Saved processed artifact: %s", out_path)

    logger.info("Preprocessing finished. Total outputs: %d", len(outputs))


if __name__ == "__main__":
    main()
