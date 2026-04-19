"""KL divergence regularization for VAE posterior."""

from __future__ import annotations

import torch


def kl_divergence_standard_normal(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    KL(q(z|x)||N(0,I)) for diagonal Gaussian posterior.

    Args:
        mu: [B, N, Z] or [N, Z]
        logvar: same shape as mu
    """
    if mu.shape != logvar.shape:
        raise ValueError("mu and logvar must have identical shapes.")

    if mu.dim() == 2:
        mu = mu.unsqueeze(0)
        logvar = logvar.unsqueeze(0)
    if mu.dim() != 3:
        raise ValueError("Expected mu/logvar rank 2 or 3.")

    kl = -0.5 * (1.0 + logvar - mu.pow(2) - torch.exp(logvar))
    # Sum latent dimension, then average over agents and batch.
    kl_per_agent = kl.sum(dim=-1)
    return kl_per_agent.mean() + (0.0 * eps)
