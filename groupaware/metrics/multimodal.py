"""Convenience wrappers for multimodal trajectory metrics."""

from __future__ import annotations

import torch

from groupaware.metrics.ade import best_of_m_ade
from groupaware.metrics.fde import best_of_m_fde


def multimodal_metrics(
    multimodal_pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return best-of-M ADE/FDE."""
    return {
        "best_of_m_ade": best_of_m_ade(multimodal_pred, target, valid_mask=valid_mask),
        "best_of_m_fde": best_of_m_fde(multimodal_pred, target, valid_mask=valid_mask),
    }
