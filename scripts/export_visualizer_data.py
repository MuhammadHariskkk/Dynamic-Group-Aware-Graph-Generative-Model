"""Export visualization-ready trajectory files."""

from __future__ import annotations

import argparse

from groupaware.experiments.runner import export_run
from groupaware.utils.config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export ETH/UCY visualization data.")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--scene-config", type=str, default=None)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--scene", type=str, default="all")
    p.add_argument("--multimodal", action="store_true")
    p.add_argument("--include-observed", action="store_true")
    p.add_argument("--include-gt", action="store_true")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--run-name", type=str, default="export")
    p.add_argument("--format", type=str, default="csv", choices=["csv", "npy", "npz"])
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
    paths = export_run(
        cfg,
        checkpoint_path=args.checkpoint,
        scene=args.scene,
        multimodal=args.multimodal,
        include_observed=args.include_observed,
        include_gt=args.include_gt,
        export_format=args.format,
        output_dir=args.output_dir,
        run_name=args.run_name,
        device=args.device,
    )
    print({"saved_files": [str(p) for p in paths]})


if __name__ == "__main__":
    main()
