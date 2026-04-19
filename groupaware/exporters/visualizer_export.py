"""Structured export utilities for ETH/UCY-compatible visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from groupaware.exporters.schema import ExportRow, rows_to_dataframe, validate_export_dataframe


def _heading_from_velocity(vx: float, vy: float) -> float:
    return float(np.arctan2(vy, vx))


def _safe_group_features(group_meta: dict[str, Any], t: int, agent_idx: int) -> tuple[int, int, float, float, float]:
    """
    Return (group_id, group_size, intra_consistency, inter_conflict, density).
    """
    gid = int(group_meta["group_ids"][t, agent_idx]) if "group_ids" in group_meta else -1
    if gid < 0:
        return -1, 1, 0.0, 0.0, 0.0

    sizes = group_meta.get("group_sizes")
    cons = group_meta.get("group_consistency")
    dens = group_meta.get("group_density")
    inter = group_meta.get("agent_inter_group_conflict")
    gsize = int(sizes[t, gid]) if sizes is not None and gid < sizes.shape[1] else 1
    gcons = float(cons[t, gid]) if cons is not None and gid < cons.shape[1] else 0.0
    gdens = float(dens[t, gid]) if dens is not None and gid < dens.shape[1] else 0.0
    ginter = float(inter[t, agent_idx]) if inter is not None else 0.0
    return gid, gsize, gcons, ginter, gdens


def build_export_package(
    batch: dict[str, Any],
    model_outputs: dict[str, Any],
    include_observed: bool = True,
    include_ground_truth: bool = True,
    include_predictions: bool = True,
    multimodal: bool = False,
) -> pd.DataFrame:
    """
    Build unified export dataframe for scene/sequence/agent trajectories.
    """
    rows: list[ExportRow] = []
    bsz = int(batch["num_agents"].shape[0])

    obs_pos = batch["observed"]["positions"].detach().cpu().numpy()  # [B,T,N,2]
    obs_vel = batch["observed"]["velocities"].detach().cpu().numpy()
    obs_frame = batch["observed"]["frame_index"].detach().cpu().numpy()  # [B,T]
    fut_pos = batch["future"]["positions"].detach().cpu().numpy()
    fut_vel = batch["future"]["velocities"].detach().cpu().numpy()
    fut_frame = batch["future"]["frame_index"].detach().cpu().numpy()
    seq_ids = batch["sequence_ids"].detach().cpu().numpy()
    agent_ids = batch["agent_ids"].detach().cpu().numpy()
    num_agents = batch["num_agents"].detach().cpu().numpy()
    scene_ids = batch["scene_ids"]

    det = None
    means = None
    probs = None
    if include_predictions:
        det = model_outputs["deterministic_traj"].detach().cpu().numpy()  # [B,N,T,2]
        means = model_outputs["means"].detach().cpu().numpy()  # [B,N,M,T,2]
        probs = model_outputs["mode_probs"].detach().cpu().numpy()  # [B,N,M]
    group_meta = model_outputs.get("group_metadata", [{} for _ in range(bsz)])
    if not isinstance(group_meta, list):
        group_meta = [{} for _ in range(bsz)]
    if len(group_meta) < bsz:
        group_meta = list(group_meta) + [{} for _ in range(bsz - len(group_meta))]

    for b in range(bsz):
        n_valid = int(num_agents[b])
        scene = str(scene_ids[b])
        seq = int(seq_ids[b])
        for ai in range(n_valid):
            aid = int(agent_ids[b, ai])

            if include_observed:
                t_obs = obs_pos.shape[1]
                for t in range(t_obs):
                    x, y = obs_pos[b, t, ai]
                    vx, vy = obs_vel[b, t, ai]
                    gid, gsize, gcons, ginter, gdens = _safe_group_features(group_meta[b], t, ai)
                    rows.append(
                        ExportRow(
                            scene_id=scene,
                            sequence_id=seq,
                            agent_id=aid,
                            time_index=t,
                            frame_index=int(obs_frame[b, t]),
                            phase="observed",
                            mode_id=-1,
                            mode_probability=1.0,
                            x=float(x),
                            y=float(y),
                            vx=float(vx),
                            vy=float(vy),
                            heading_rad=_heading_from_velocity(float(vx), float(vy)),
                            group_id=gid,
                            group_size=gsize,
                            intra_group_consistency=gcons,
                            inter_group_conflict=ginter,
                            group_density=gdens,
                            is_prediction=0,
                            is_deterministic=1,
                        )
                    )

            if include_ground_truth:
                t_pred = fut_pos.shape[1]
                for t in range(t_pred):
                    x, y = fut_pos[b, t, ai]
                    vx, vy = fut_vel[b, t, ai]
                    rows.append(
                        ExportRow(
                            scene_id=scene,
                            sequence_id=seq,
                            agent_id=aid,
                            time_index=t,
                            frame_index=int(fut_frame[b, t]),
                            phase="ground_truth",
                            mode_id=-1,
                            mode_probability=1.0,
                            x=float(x),
                            y=float(y),
                            vx=float(vx),
                            vy=float(vy),
                            heading_rad=_heading_from_velocity(float(vx), float(vy)),
                            group_id=-1,
                            group_size=1,
                            intra_group_consistency=0.0,
                            inter_group_conflict=0.0,
                            group_density=0.0,
                            is_prediction=0,
                            is_deterministic=1,
                        )
                    )

            if include_predictions:
                if det is None or means is None or probs is None:
                    raise ValueError("Predictions requested but model outputs are missing prediction tensors.")
                t_pred = det.shape[2]
                if multimodal:
                    modes = means.shape[2]
                    for m in range(modes):
                        for t in range(t_pred):
                            x, y = means[b, ai, m, t]
                            if t == 0:
                                vx, vy = x - obs_pos[b, -1, ai, 0], y - obs_pos[b, -1, ai, 1]
                            else:
                                vx, vy = means[b, ai, m, t, 0] - means[b, ai, m, t - 1, 0], means[b, ai, m, t, 1] - means[b, ai, m, t - 1, 1]
                            rows.append(
                                ExportRow(
                                    scene_id=scene,
                                    sequence_id=seq,
                                    agent_id=aid,
                                    time_index=t,
                                    frame_index=int(fut_frame[b, t]),
                                    phase="predicted",
                                    mode_id=int(m),
                                    mode_probability=float(probs[b, ai, m]),
                                    x=float(x),
                                    y=float(y),
                                    vx=float(vx),
                                    vy=float(vy),
                                    heading_rad=_heading_from_velocity(float(vx), float(vy)),
                                    group_id=-1,
                                    group_size=1,
                                    intra_group_consistency=0.0,
                                    inter_group_conflict=0.0,
                                    group_density=0.0,
                                    is_prediction=1,
                                    is_deterministic=0,
                                )
                            )
                else:
                    mode_id = int(np.argmax(probs[b, ai]))
                    mode_prob = float(np.max(probs[b, ai]))
                    for t in range(t_pred):
                        x, y = det[b, ai, t]
                        if t == 0:
                            vx, vy = x - obs_pos[b, -1, ai, 0], y - obs_pos[b, -1, ai, 1]
                        else:
                            vx, vy = det[b, ai, t, 0] - det[b, ai, t - 1, 0], det[b, ai, t, 1] - det[b, ai, t - 1, 1]
                        rows.append(
                            ExportRow(
                                scene_id=scene,
                                sequence_id=seq,
                                agent_id=aid,
                                time_index=t,
                                frame_index=int(fut_frame[b, t]),
                                phase="predicted",
                                mode_id=mode_id,
                                mode_probability=mode_prob,
                                x=float(x),
                                y=float(y),
                                vx=float(vx),
                                vy=float(vy),
                                heading_rad=_heading_from_velocity(float(vx), float(vy)),
                                group_id=-1,
                                group_size=1,
                                intra_group_consistency=0.0,
                                inter_group_conflict=0.0,
                                group_density=0.0,
                                is_prediction=1,
                                is_deterministic=1,
                            )
                        )

    df = rows_to_dataframe(rows)
    validate_export_dataframe(df)
    return df


def export_predictions(batch: dict[str, Any], model_outputs: dict[str, Any], multimodal: bool = False) -> pd.DataFrame:
    """Export only predictions (deterministic or multimodal)."""
    return build_export_package(
        batch=batch,
        model_outputs=model_outputs,
        include_observed=False,
        include_ground_truth=False,
        include_predictions=True,
        multimodal=multimodal,
    )


def export_ground_truth(batch: dict[str, Any], model_outputs: dict[str, Any] | None = None) -> pd.DataFrame:
    """Export observed + GT only."""
    return build_export_package(
        batch=batch,
        model_outputs=model_outputs or {"group_metadata": []},
        include_observed=True,
        include_ground_truth=True,
        include_predictions=False,
        multimodal=False,
    )


def export_combined(batch: dict[str, Any], model_outputs: dict[str, Any], multimodal: bool = False) -> pd.DataFrame:
    """Export observed + GT + predictions together."""
    return build_export_package(
        batch=batch,
        model_outputs=model_outputs,
        include_observed=True,
        include_ground_truth=True,
        include_predictions=True,
        multimodal=multimodal,
    )


def save_visualizer_files(
    df: pd.DataFrame,
    output_dir: str | Path,
    run_name: str,
    export_format: str = "csv",
    one_file_per_scene: bool = True,
    one_file_per_run: bool = True,
) -> list[Path]:
    """Save structured exports as CSV/NPY/NPZ."""
    export_format = export_format.lower()
    if export_format not in {"csv", "npy", "npz"}:
        raise ValueError("export_format must be one of: csv, npy, npz")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    if one_file_per_run:
        base = out_dir / f"{run_name}_all"
        if export_format == "csv":
            path = base.with_suffix(".csv")
            df.to_csv(path, index=False)
        elif export_format == "npy":
            path = base.with_suffix(".npy")
            np.save(path, df.to_records(index=False), allow_pickle=False)
        else:
            path = base.with_suffix(".npz")
            np.savez_compressed(path, data=df.to_records(index=False))
        outputs.append(path)

    if one_file_per_scene:
        for scene_id, scene_df in df.groupby("scene_id", sort=True):
            base = out_dir / f"{run_name}_{scene_id}"
            if export_format == "csv":
                path = base.with_suffix(".csv")
                scene_df.to_csv(path, index=False)
            elif export_format == "npy":
                path = base.with_suffix(".npy")
                np.save(path, scene_df.to_records(index=False), allow_pickle=False)
            else:
                path = base.with_suffix(".npz")
                np.savez_compressed(path, data=scene_df.to_records(index=False))
            outputs.append(path)
    return outputs
