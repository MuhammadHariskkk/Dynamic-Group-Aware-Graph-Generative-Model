"""Gaussian mixture negative log-likelihood loss."""

from __future__ import annotations

import torch


def gmm_nll_loss(
    means: torch.Tensor,
    stds: torch.Tensor,
    mode_probs: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute NLL for diagonal-covariance trajectory GMM.

    Args:
        means: [B, N, M, T, 2]
        stds: [B, N, M, T, 2]
        mode_probs: [B, N, M]
        target: [B, N, T, 2]
        valid_mask: optional [B, N, T]
    """
    if means.dim() != 5 or stds.dim() != 5 or mode_probs.dim() != 3 or target.dim() != 4:
        raise ValueError("Invalid input rank for gmm_nll_loss.")
    if means.shape != stds.shape:
        raise ValueError("means and stds must have identical shapes.")
    if means.shape[:2] != mode_probs.shape[:2]:
        raise ValueError("means and mode_probs must share [B, N].")

    b, n, m, t, d = means.shape
    if d != 2:
        raise ValueError("Last dimension must be 2 for x/y coordinates.")
    if target.shape != (b, n, t, d):
        raise ValueError("target shape must be [B, N, T, 2].")

    # Expand target to per-mode shape.
    target_m = target.unsqueeze(2).expand(b, n, m, t, d)
    var = stds * stds + eps

    # log N(y | mu, diag(var))
    log_det = torch.log(2.0 * torch.pi * var).sum(dim=(-1, -2)) * 0.5  # [B,N,M]
    sq = ((target_m - means) ** 2 / var).sum(dim=(-1, -2)) * 0.5  # [B,N,M]
    log_gauss = -(log_det + sq)  # [B,N,M]

    log_pi = torch.log(mode_probs + eps)  # [B,N,M]
    log_mix = torch.logsumexp(log_pi + log_gauss, dim=2)  # [B,N]
    nll = -log_mix  # [B,N]

    if valid_mask is not None:
        if valid_mask.shape != (b, n, t):
            raise ValueError("valid_mask must have shape [B, N, T].")
        valid_agent = (valid_mask.sum(dim=-1) > 0).float()  # [B,N]
        return (nll * valid_agent).sum() / (valid_agent.sum() + eps)

    return nll.mean()
