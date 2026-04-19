"""Final Displacement Error metrics."""

from __future__ import annotations

import torch


def fde(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Deterministic FDE.

    Args:
        pred: [B, N, T, 2]
        target: [B, N, T, 2]
        valid_mask: optional [B, N, T]
    """
    if pred.shape != target.shape or pred.dim() != 4:
        raise ValueError("pred/target must share shape [B, N, T, 2].")

    pred_f = pred[:, :, -1, :]
    tgt_f = target[:, :, -1, :]
    dist = torch.norm(pred_f - tgt_f, dim=-1)  # [B,N]

    if valid_mask is not None:
        if valid_mask.shape != (pred.shape[0], pred.shape[1], pred.shape[2]):
            raise ValueError("valid_mask must have shape [B, N, T].")
        valid_last = valid_mask[:, :, -1].float()
        return (dist * valid_last).sum() / (valid_last.sum() + eps)
    return dist.mean()


def best_of_m_fde(
    multimodal_pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Best-of-M FDE.

    Args:
        multimodal_pred: [B, N, M, T, 2]
        target: [B, N, T, 2]
    """
    if multimodal_pred.dim() != 5:
        raise ValueError("multimodal_pred must have shape [B, N, M, T, 2].")
    b, n, m, t, d = multimodal_pred.shape
    if target.shape != (b, n, t, d):
        raise ValueError("target must have shape [B, N, T, 2].")

    pred_f = multimodal_pred[:, :, :, -1, :]  # [B,N,M,2]
    tgt_f = target[:, :, -1, :].unsqueeze(2).expand(b, n, m, d)  # [B,N,M,2]
    dist = torch.norm(pred_f - tgt_f, dim=-1)  # [B,N,M]

    if valid_mask is not None:
        if valid_mask.shape != (b, n, t):
            raise ValueError("valid_mask must have shape [B, N, T].")
        valid_last = valid_mask[:, :, -1].unsqueeze(-1).float()  # [B,N,1]
        dist = dist + (1.0 - valid_last) * 1e6

    best = dist.min(dim=2).values  # [B,N]
    if valid_mask is not None:
        valid_last = valid_mask[:, :, -1].float()
        return (best * valid_last).sum() / (valid_last.sum() + eps)
    return best.mean()
