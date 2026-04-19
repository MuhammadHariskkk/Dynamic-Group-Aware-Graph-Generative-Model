"""Train CLI."""

from __future__ import annotations

import argparse

from groupaware.experiments.runner import train_run
from groupaware.utils.config import load_config
from groupaware.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train group-aware model.")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--scene-config", type=str, default=None)
    p.add_argument("--train-scene", type=str, default="all")
    p.add_argument("--val-scene", type=str, default="all")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--override", action="append", default=[])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(
        args.scene_config or args.config,
        base_config_path=(args.config if args.scene_config else None),
        cli_overrides=args.override,
    )
    set_seed(int(cfg["project"]["seed"]))
    train_run(cfg, train_scene=args.train_scene, val_scene=args.val_scene, device=args.device)


if __name__ == "__main__":
    main()
