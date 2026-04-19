"""Training and validation loops for group-aware model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from groupaware.losses.gmm_nll import gmm_nll_loss
from groupaware.losses.kl_divergence import kl_divergence_standard_normal
from groupaware.metrics.ade import ade, best_of_m_ade
from groupaware.metrics.fde import best_of_m_fde, fde
from groupaware.utils.checkpoint import save_checkpoint
from groupaware.utils.logger import create_logger


@dataclass
class TrainerConfig:
    """Trainer runtime configuration."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 500
    lambda_nll: float = 1.0
    lambda_kl: float = 0.1
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 30
    monitor: str = "val_loss"
    mode: str = "min"
    checkpoint_dir: str = "outputs/checkpoints"
    logs_dir: str = "outputs/logs"
    metrics_dir: str = "outputs/metrics"
    use_multimodal_metrics: bool = True


@dataclass
class TrainerState:
    """Training state for early stopping/checkpointing."""

    best_metric: float = float("inf")
    best_epoch: int = -1
    patience_counter: int = 0
    history: list[dict[str, float]] = field(default_factory=list)


class Trainer:
    """Model trainer with train/val loops, logging, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.cfg = config
        self.device = torch.device(device)
        self.model.to(self.device)

        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.state = TrainerState()
        self.logger = create_logger("trainer", log_dir=config.logs_dir)

        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.metrics_dir).mkdir(parents=True, exist_ok=True)

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move tensor values to target device recursively."""
        out: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, dict):
                out[key] = self._move_batch_to_device(value)
            elif torch.is_tensor(value):
                out[key] = value.to(self.device)
            else:
                out[key] = value
        return out

    def _compute_losses_and_metrics(self, outputs: dict[str, Any], batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Compute total loss and core metrics from model outputs."""
        # Collate schema uses [B,T,N,2], model uses [B,N,T,2].
        target = batch["future"]["positions"].permute(0, 2, 1, 3).contiguous()
        valid_mask = batch["future"]["valid_mask"].permute(0, 2, 1).contiguous()

        nll = gmm_nll_loss(
            means=outputs["means"],
            stds=outputs["stds"],
            mode_probs=outputs["mode_probs"],
            target=target,
            valid_mask=valid_mask,
        )
        kl = kl_divergence_standard_normal(outputs["mu"], outputs["logvar"])
        total = (self.cfg.lambda_nll * nll) + (self.cfg.lambda_kl * kl)

        det_pred = outputs["deterministic_traj"]
        det_ade = ade(det_pred, target, valid_mask=valid_mask)
        det_fde = fde(det_pred, target, valid_mask=valid_mask)

        metrics: dict[str, torch.Tensor] = {
            "loss": total,
            "nll": nll,
            "kl": kl,
            "ade": det_ade,
            "fde": det_fde,
        }
        if self.cfg.use_multimodal_metrics:
            metrics["best_of_m_ade"] = best_of_m_ade(outputs["means"], target, valid_mask=valid_mask)
            metrics["best_of_m_fde"] = best_of_m_fde(outputs["means"], target, valid_mask=valid_mask)
        return metrics

    @staticmethod
    def _mean_metric(values: list[float]) -> float:
        return float(sum(values) / max(1, len(values)))

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict[str, float]:
        """Run one training or validation epoch."""
        self.model.train(mode=train)
        agg: dict[str, list[float]] = {}

        desc = "train" if train else "val"
        for raw_batch in tqdm(loader, desc=desc, leave=False):
            batch = self._move_batch_to_device(raw_batch)
            if train:
                self.optimizer.zero_grad(set_to_none=True)

            outputs = self.model(batch)
            metrics = self._compute_losses_and_metrics(outputs, batch)

            if train:
                metrics["loss"].backward()
                if self.cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
                self.optimizer.step()

            for key, value in metrics.items():
                agg.setdefault(key, []).append(float(value.detach().cpu().item()))

        return {k: self._mean_metric(v) for k, v in agg.items()}

    def _is_improved(self, current: float, best: float) -> bool:
        if self.cfg.mode == "min":
            return current < best
        return current > best

    def _save_history(self) -> None:
        """Persist training history to CSV and JSON."""
        metrics_dir = Path(self.cfg.metrics_dir)
        csv_path = metrics_dir / "history.csv"
        json_path = metrics_dir / "history.json"

        if self.state.history:
            pd.DataFrame(self.state.history).to_csv(csv_path, index=False)
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(self.state.history, f, indent=2)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict[str, Any]:
        """Train model with early stopping and best-checkpoint selection."""
        self.logger.info("Starting training for %d epochs", self.cfg.epochs)
        for epoch in range(1, self.cfg.epochs + 1):
            train_metrics = self._run_epoch(train_loader, train=True)
            val_metrics = self._run_epoch(val_loader, train=False)

            row: dict[str, float] = {"epoch": float(epoch)}
            row.update({f"train_{k}": v for k, v in train_metrics.items()})
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            self.state.history.append(row)

            monitor_key = f"val_{self.cfg.monitor.replace('val_', '')}"
            current_metric = row.get(monitor_key, row.get("val_loss", float("inf")))
            self.logger.info(
                "Epoch %d | train_loss=%.6f | val_loss=%.6f | val_ade=%.6f | val_fde=%.6f",
                epoch,
                row.get("train_loss", float("nan")),
                row.get("val_loss", float("nan")),
                row.get("val_ade", float("nan")),
                row.get("val_fde", float("nan")),
            )

            if self._is_improved(current_metric, self.state.best_metric):
                self.state.best_metric = current_metric
                self.state.best_epoch = epoch
                self.state.patience_counter = 0
                save_checkpoint(
                    Path(self.cfg.checkpoint_dir) / "best.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    best_val_loss=row.get("val_loss"),
                    extra={"monitor_metric": current_metric},
                )
            else:
                self.state.patience_counter += 1

            save_checkpoint(
                Path(self.cfg.checkpoint_dir) / "last.pt",
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                best_val_loss=self.state.best_metric,
                extra={"monitor_metric": current_metric},
            )
            self._save_history()

            if self.state.patience_counter >= self.cfg.early_stopping_patience:
                self.logger.info(
                    "Early stopping at epoch %d (best epoch=%d).",
                    epoch,
                    self.state.best_epoch,
                )
                break

        return {
            "best_epoch": self.state.best_epoch,
            "best_metric": self.state.best_metric,
            "history": self.state.history,
            "best_checkpoint": str(Path(self.cfg.checkpoint_dir) / "best.pt"),
            "last_checkpoint": str(Path(self.cfg.checkpoint_dir) / "last.pt"),
        }
