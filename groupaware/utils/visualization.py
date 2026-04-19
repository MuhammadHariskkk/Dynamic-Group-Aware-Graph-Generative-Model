"""Optional debug plotting helpers for trajectories and training curves."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_prediction_sample(
    observed_pos: np.ndarray,
    future_gt_pos: np.ndarray,
    pred_pos: np.ndarray,
    output_path: str | Path,
    title: str = "Trajectory Prediction",
    group_ids_t: np.ndarray | None = None,
    group_centroids_t: np.ndarray | None = None,
    graph_t: dict[str, Any] | None = None,
    conflict_t: np.ndarray | None = None,
) -> Path:
    """
    Plot observed, GT future, predicted future with optional group/graph overlays.

    Args:
        observed_pos: [T_obs, N, 2]
        future_gt_pos: [T_pred, N, 2]
        pred_pos: [T_pred, N, 2]
    """
    out_path = Path(output_path)
    _ensure_dir(out_path.parent)

    t_obs, n, _ = observed_pos.shape
    t_pred = future_gt_pos.shape[0]
    cmap = plt.cm.get_cmap("tab20", n)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(n):
        c = cmap(i)
        ax.plot(observed_pos[:, i, 0], observed_pos[:, i, 1], color=c, linestyle="-", linewidth=2, label=f"agent{i}-obs" if i == 0 else None)
        ax.plot(future_gt_pos[:, i, 0], future_gt_pos[:, i, 1], color=c, linestyle="--", linewidth=1.8, label=f"agent{i}-gt" if i == 0 else None)
        ax.plot(pred_pos[:, i, 0], pred_pos[:, i, 1], color=c, linestyle="-.", linewidth=1.8, label=f"agent{i}-pred" if i == 0 else None)
        ax.scatter(observed_pos[-1, i, 0], observed_pos[-1, i, 1], color=c, s=20)

    # Optional grouping overlay at last observed timestep.
    if group_ids_t is not None:
        for i in range(n):
            gid = int(group_ids_t[i])
            txt = f"g{gid}" if gid >= 0 else "g-"
            ax.text(observed_pos[-1, i, 0], observed_pos[-1, i, 1], txt, fontsize=8)

    if group_centroids_t is not None and group_centroids_t.size > 0:
        ax.scatter(group_centroids_t[:, 0], group_centroids_t[:, 1], marker="X", s=80, color="black", label="group centroids")

    # Optional hybrid graph edge overlay (last timestep).
    if graph_t is not None:
        node_pos = graph_t.get("node_positions")
        adj = graph_t.get("adjacency")
        if node_pos is not None and adj is not None:
            v = node_pos.shape[0]
            for i in range(v):
                for j in range(v):
                    if i == j:
                        continue
                    w = float(adj[i, j])
                    if w > 0.05:
                        ax.plot(
                            [node_pos[i, 0], node_pos[j, 0]],
                            [node_pos[i, 1], node_pos[j, 1]],
                            color="gray",
                            alpha=min(0.6, w),
                            linewidth=0.5,
                        )

    # Optional conflict summary text.
    if conflict_t is not None and conflict_t.size > 0:
        ax.text(0.01, 0.99, f"mean conflict={float(np.mean(conflict_t)):.3f}", transform=ax.transAxes, va="top")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_training_curves(
    history_csv: str | Path,
    output_path: str | Path,
    keys: tuple[str, ...] = ("train_loss", "val_loss", "train_ade", "val_ade"),
) -> Path:
    """Plot training curves from trainer history CSV."""
    hist_path = Path(history_csv)
    if not hist_path.exists():
        raise FileNotFoundError(f"History CSV not found: {hist_path}")

    df = pd.read_csv(hist_path)
    out_path = Path(output_path)
    _ensure_dir(out_path.parent)

    fig, ax = plt.subplots(figsize=(8, 5))
    for key in keys:
        if key in df.columns:
            ax.plot(df["epoch"], df[key], label=key)
    ax.set_title("Training Curves")
    ax.set_xlabel("epoch")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
