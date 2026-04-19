"""Average Displacement Error metrics."""

from __future__ import annotations

import torch


def ade(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Deterministic ADE.

    Args:
        pred: [B, N, T, 2]
        target: [B, N, T, 2]
        valid_mask: optional [B, N, T]
    """
    if pred.shape != target.shape or pred.dim() != 4:
        raise ValueError("pred/target must share shape [B, N, T, 2].")
    dist = torch.norm(pred - target, dim=-1)  # [B,N,T]

    if valid_mask is not None:
        if valid_mask.shape != dist.shape:
            raise ValueError("valid_mask must have shape [B, N, T].")
        return (dist * valid_mask.float()).sum() / (valid_mask.float().sum() + eps)
    return dist.mean()


def best_of_m_ade(
    multimodal_pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Best-of-M ADE.

    Args:
        multimodal_pred: [B, N, M, T, 2]
        target: [B, N, T, 2]
    """
    if multimodal_pred.dim() != 5:
        raise ValueError("multimodal_pred must have shape [B, N, M, T, 2].")
    b, n, m, t, d = multimodal_pred.shape
    if target.shape != (b, n, t, d):
        raise ValueError("target must have shape [B, N, T, 2].")

    target_m = target.unsqueeze(2).expand_as(multimodal_pred)
    dist = torch.norm(multimodal_pred - target_m, dim=-1)  # [B,N,M,T]

    if valid_mask is not None:
        if valid_mask.shape != (b, n, t):
            raise ValueError("valid_mask must have shape [B, N, T].")
        vm = valid_mask.unsqueeze(2).float()  # [B,N,1,T]
        ade_m = (dist * vm).sum(dim=-1) / (vm.sum(dim=-1) + eps)  # [B,N,M]
    else:
        ade_m = dist.mean(dim=-1)  # [B,N,M]

    best = ade_m.min(dim=2).values  # [B,N]
    return best.mean()
