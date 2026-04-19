"""Checkpoint save/load helpers for model training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    epoch: int | None = None,
    best_val_loss: float | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save model/optimizer state and metadata into a single checkpoint file."""
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {"model_state_dict": model.state_dict()}
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch
    if best_val_loss is not None:
        payload["best_val_loss"] = best_val_loss
    if extra:
        payload["extra"] = extra

    torch.save(payload, path)


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load checkpoint into model/optimizer and return metadata dictionary."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    return {
        "epoch": payload.get("epoch"),
        "best_val_loss": payload.get("best_val_loss"),
        "extra": payload.get("extra", {}),
    }
