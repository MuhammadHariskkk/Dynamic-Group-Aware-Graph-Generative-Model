"""Experiment runner for train/eval/infer/export workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

from groupaware.datasets import ETHUCYDataset, collate_eth_ucy
from groupaware.exporters.visualizer_export import build_export_package, save_visualizer_files
from groupaware.models.group_aware_model import GroupAwareModelConfig, GroupAwareTrajectoryModel
from groupaware.trainers import Trainer, TrainerConfig
from groupaware.utils.checkpoint import load_checkpoint
from groupaware.utils.logger import create_logger


def _collect_processed_files(processed_root: str | Path, scene: str = "all") -> list[Path]:
    root = Path(processed_root)
    if scene == "all":
        return sorted(root.glob("*_obs*_pred*.pt"))
    return sorted(root.glob(f"{scene.lower()}_obs*_pred*.pt"))


def _build_loader(files: list[Path], batch_size: int, shuffle: bool = False) -> DataLoader:
    if not files:
        raise FileNotFoundError("No processed dataset files found for requested scene/filter.")
    datasets = [ETHUCYDataset(f) for f in files]
    merged = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    return DataLoader(
        merged,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_eth_ucy,
    )


def build_model_from_config(cfg: dict[str, Any], device: str | torch.device) -> GroupAwareTrajectoryModel:
    model_cfg = GroupAwareModelConfig.from_dict(cfg)
    model = GroupAwareTrajectoryModel(model_cfg)
    model.to(device)
    return model


def train_run(cfg: dict[str, Any], train_scene: str = "all", val_scene: str = "all", device: str = "cpu") -> dict[str, Any]:
    logger = create_logger("runner_train", log_dir=cfg["paths"]["logs_dir"])
    model = build_model_from_config(cfg, device=device)

    train_files = _collect_processed_files(cfg["data"]["processed_root"], scene=train_scene)
    val_files = _collect_processed_files(cfg["data"]["processed_root"], scene=val_scene)
    train_loader = _build_loader(train_files, batch_size=int(cfg["training"]["batch_size"]), shuffle=True)
    val_loader = _build_loader(val_files, batch_size=int(cfg["training"]["batch_size"]), shuffle=False)

    trainer = Trainer(
        model=model,
        config=TrainerConfig(
            learning_rate=float(cfg["training"]["learning_rate"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
            epochs=int(cfg["training"]["epochs"]),
            lambda_nll=float(cfg["loss"]["lambda_nll"]),
            lambda_kl=float(cfg["loss"]["lambda_kl"]),
            grad_clip_norm=float(cfg["training"]["grad_clip_norm"]),
            early_stopping_patience=int(cfg["training"]["early_stopping"]["patience"]),
            monitor=str(cfg["training"]["early_stopping"]["monitor"]),
            mode=str(cfg["training"]["early_stopping"]["mode"]),
            checkpoint_dir=str(cfg["paths"]["checkpoints_dir"]),
            logs_dir=str(cfg["paths"]["logs_dir"]),
            metrics_dir=str(cfg["paths"]["metrics_dir"]),
            use_multimodal_metrics=True,
        ),
        device=device,
    )
    result = trainer.fit(train_loader, val_loader)
    logger.info("Training finished. Best checkpoint: %s", result["best_checkpoint"])
    return result


def evaluate_run(
    cfg: dict[str, Any],
    checkpoint_path: str,
    scene: str = "all",
    device: str = "cpu",
) -> dict[str, Any]:
    logger = create_logger("runner_eval", log_dir=cfg["paths"]["logs_dir"])
    model = build_model_from_config(cfg, device=device)
    load_checkpoint(checkpoint_path, model=model, map_location=device)
    model.eval()

    files = _collect_processed_files(cfg["data"]["processed_root"], scene=scene)
    loader = _build_loader(files, batch_size=int(cfg["training"]["batch_size"]), shuffle=False)

    rows: list[dict[str, float]] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: ({kk: vv.to(device) if torch.is_tensor(vv) else vv for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in batch.items()}
            out = model(batch)
            pred = out["deterministic_traj"]
            tgt = batch["future"]["positions"].permute(0, 2, 1, 3).contiguous()
            ade = torch.norm(pred - tgt, dim=-1).mean().item()
            fde = torch.norm(pred[:, :, -1] - tgt[:, :, -1], dim=-1).mean().item()
            rows.append({"ade": ade, "fde": fde})

    df = pd.DataFrame(rows)
    summary = {"ade": float(df["ade"].mean()), "fde": float(df["fde"].mean()), "num_batches": int(len(df))}
    out_path = Path(cfg["paths"]["metrics_dir"]) / f"eval_{scene}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Saved evaluation metrics to %s", out_path)
    return summary


def infer_run(
    cfg: dict[str, Any],
    checkpoint_path: str,
    scene: str = "all",
    device: str = "cpu",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model = build_model_from_config(cfg, device=device)
    load_checkpoint(checkpoint_path, model=model, map_location=device)
    model.eval()
    files = _collect_processed_files(cfg["data"]["processed_root"], scene=scene)
    loader = _build_loader(files, batch_size=int(cfg["training"]["batch_size"]), shuffle=False)

    batches: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            device_batch = {
                k: ({kk: vv.to(device) if torch.is_tensor(vv) else vv for kk, vv in v.items()} if isinstance(v, dict) else v)
                for k, v in batch.items()
            }
            out = model(device_batch)
            batches.append(batch)
            outputs.append({k: v.cpu() if torch.is_tensor(v) else v for k, v in out.items()})
    return batches, outputs


def export_run(
    cfg: dict[str, Any],
    checkpoint_path: str,
    scene: str = "all",
    multimodal: bool = False,
    include_observed: bool = True,
    include_gt: bool = True,
    export_format: str = "csv",
    output_dir: str | None = None,
    run_name: str = "run",
    device: str = "cpu",
) -> list[Path]:
    batches, outputs = infer_run(cfg, checkpoint_path=checkpoint_path, scene=scene, device=device)
    frames: list[pd.DataFrame] = []
    for batch, out in zip(batches, outputs):
        frames.append(
            build_export_package(
                batch=batch,
                model_outputs=out,
                include_observed=include_observed,
                include_ground_truth=include_gt,
                include_predictions=True,
                multimodal=multimodal,
            )
        )
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_root = output_dir or cfg["paths"]["exports_dir"]
    return save_visualizer_files(
        merged,
        output_dir=out_root,
        run_name=run_name,
        export_format=export_format,
        one_file_per_scene=bool(cfg["export"]["one_file_per_scene"]),
        one_file_per_run=bool(cfg["export"]["one_file_per_run"]),
    )
