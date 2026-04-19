"""Optional debug visualization script."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from groupaware.experiments.runner import infer_run
from groupaware.utils.config import load_config
from groupaware.utils.visualization import plot_prediction_sample, plot_training_curves


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize predicted trajectories (debug).")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--scene-config", type=str, default=None)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--scene", type=str, default="all")
    p.add_argument("--batch-index", type=int, default=0)
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="outputs/figures")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--show-groups", action="store_true")
    p.add_argument("--show-centroids", action="store_true")
    p.add_argument("--show-graph", action="store_true")
    p.add_argument("--show-conflict", action="store_true")
    p.add_argument("--plot-curves", action="store_true")
    p.add_argument("--history-csv", type=str, default="outputs/metrics/history.csv")
    p.add_argument("--override", action="append", default=[])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(
        args.scene_config or args.config,
        base_config_path=(args.config if args.scene_config else None),
        cli_overrides=args.override,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    batches, outputs = infer_run(cfg, checkpoint_path=args.checkpoint, scene=args.scene, device=args.device)
    if not batches:
        raise RuntimeError("No inference batches available for visualization.")
    if args.batch_index < 0 or args.batch_index >= len(batches):
        raise IndexError(f"batch-index out of range: {args.batch_index}")

    batch = batches[args.batch_index]
    out = outputs[args.batch_index]
    sample_idx = args.sample_index
    if sample_idx < 0 or sample_idx >= int(batch["num_agents"].shape[0]):
        raise IndexError(f"sample-index out of range: {sample_idx}")

    # Batch tensors use [B,T,N,2] for observed/future and model deterministic [B,N,T,2].
    obs = batch["observed"]["positions"][sample_idx].detach().cpu().numpy()  # [T_obs,N,2]
    fut = batch["future"]["positions"][sample_idx].detach().cpu().numpy()  # [T_pred,N,2]
    pred = out["deterministic_traj"][sample_idx].detach().cpu().numpy().transpose(1, 0, 2)  # [T_pred,N,2]

    group_ids = None
    centroids = None
    graph_t = None
    conflict_t = None
    if args.show_groups and "group_metadata" in out:
        gmeta = out["group_metadata"][sample_idx]
        group_ids = gmeta["group_ids"][-1]
        if args.show_centroids:
            valid_groups = gmeta["group_sizes"][-1] > 0
            centroids = gmeta["group_centroids"][-1][valid_groups]
        if args.show_conflict and "per_timestep" in gmeta:
            conflict_t = gmeta["per_timestep"][-1]["conflict_softmax"]
    if args.show_graph and "graph_metadata" in out:
        graph_t = out["graph_metadata"][sample_idx]["per_timestep"][-1]

    fig_path = out_dir / f"pred_scene_{args.scene}_b{args.batch_index}_s{sample_idx}.png"
    saved = plot_prediction_sample(
        observed_pos=obs,
        future_gt_pos=fut,
        pred_pos=pred,
        output_path=fig_path,
        title=f"Scene {args.scene} | batch {args.batch_index} sample {sample_idx}",
        group_ids_t=group_ids,
        group_centroids_t=centroids,
        graph_t=graph_t,
        conflict_t=conflict_t,
    )
    print({"saved_prediction_plot": str(saved)})

    if args.plot_curves:
        curve_path = out_dir / "training_curves.png"
        saved_curve = plot_training_curves(args.history_csv, curve_path)
        print({"saved_training_curves": str(saved_curve)})


if __name__ == "__main__":
    main()
