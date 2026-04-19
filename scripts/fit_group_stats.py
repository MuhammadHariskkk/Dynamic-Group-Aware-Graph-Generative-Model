"""Compute simple aggregate group statistics from processed files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from groupaware.grouping.group_features import GroupFeatureConfig, compute_dynamic_group_features
from groupaware.utils.config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit dataset-level group stats.")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--scene-config", type=str, default=None)
    p.add_argument("--scene", type=str, default="all")
    p.add_argument("--output", type=str, default="outputs/metrics/group_stats.csv")
    p.add_argument("--override", action="append", default=[])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(
        args.scene_config or args.config,
        base_config_path=(args.config if args.scene_config else None),
        cli_overrides=args.override,
    )
    root = Path(cfg["data"]["processed_root"])
    files = sorted(root.glob("*_obs*_pred*.pt")) if args.scene == "all" else sorted(root.glob(f"{args.scene}_obs*_pred*.pt"))
    if not files:
        raise FileNotFoundError("No processed files found.")

    gcfg = GroupFeatureConfig(
        distance_threshold_m=float(cfg["grouping"]["distance_threshold_m"]),
        velocity_diff_threshold_mps=float(cfg["grouping"]["velocity_diff_threshold_mps"]),
        directional_coherence_threshold=float(cfg["grouping"]["directional_coherence_threshold"]),
    )

    rows: list[dict[str, float]] = []
    for f in files:
        payload = torch.load(f, map_location="cpu")
        for seq in payload["sequences"]:
            out = compute_dynamic_group_features(
                positions=seq["timestep_states"]["positions"],
                velocities=seq["timestep_states"]["velocities"],
                valid_mask=seq["timestep_states"]["valid_mask"],
                cfg=gcfg,
            )
            sizes = out["group_sizes"][out["group_sizes"] > 0]
            cons = out["group_consistency"][out["group_sizes"] > 0]
            dens = out["group_density"][out["group_sizes"] > 0]
            rows.append(
                {
                    "scene_id": str(seq["scene_id"]),
                    "mean_group_size": float(np.mean(sizes) if sizes.size else 0.0),
                    "mean_consistency": float(np.mean(cons) if cons.size else 0.0),
                    "mean_density": float(np.mean(dens) if dens.size else 0.0),
                    "num_groups": float(sizes.size),
                }
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print({"saved": str(out), "num_rows": len(rows)})


if __name__ == "__main__":
    main()
