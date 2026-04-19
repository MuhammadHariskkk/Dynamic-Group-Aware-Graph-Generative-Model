"""Export evaluation summary CSV helper."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from groupaware.experiments.runner import evaluate_run
from groupaware.utils.config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate and export summary results.")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--scene-config", type=str, default=None)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--scene", type=str, default="all")
    p.add_argument("--output", type=str, default="outputs/metrics/results_summary.csv")
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
    summary = evaluate_run(cfg, checkpoint_path=args.checkpoint, scene=args.scene, device=args.device)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(out, index=False)
    print({"saved": str(out), **summary})


if __name__ == "__main__":
    main()
