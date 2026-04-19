"""Inference CLI."""

from __future__ import annotations

import argparse

from groupaware.experiments.runner import infer_run
from groupaware.utils.config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run model inference.")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--scene-config", type=str, default=None)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--scene", type=str, default="all")
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
    batches, outputs = infer_run(cfg, checkpoint_path=args.checkpoint, scene=args.scene, device=args.device)
    print({"num_batches": len(batches), "num_outputs": len(outputs)})


if __name__ == "__main__":
    main()
