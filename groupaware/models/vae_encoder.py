"""Group-aware variational encoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class VAEEncoderConfig:
    """Configuration for VAE encoder."""

    temporal_dim: int = 12
    context_dim: int = 8
    hidden_dim: int = 64
    latent_dim: int = 5
    eps: float = 1e-8


class GroupAwareVAEEncoder(nn.Module):
    """
    Encode temporal embedding + group context into latent posterior parameters.

    Inputs:
    - temporal_embedding: [B, N, D_t] or [N, D_t]
    - group_context: [B, N, D_c] or [N, D_c]
    Outputs:
    - mu: [B, N, Z] or [N, Z]
    - logvar: [B, N, Z] or [N, Z]
    - z: [B, N, Z] or [N, Z]
    """

    def __init__(self, config: VAEEncoderConfig) -> None:
        super().__init__()
        self.config = config
        in_dim = config.temporal_dim + config.context_dim
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(config.hidden_dim, config.latent_dim)
        self.logvar_head = nn.Linear(config.hidden_dim, config.latent_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample latent variable using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        temporal_embedding: torch.Tensor,
        group_context: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for latent posterior estimation."""
        squeeze_batch = False
        if temporal_embedding.dim() == 2 and group_context.dim() == 2:
            temporal_embedding = temporal_embedding.unsqueeze(0)
            group_context = group_context.unsqueeze(0)
            squeeze_batch = True

        if temporal_embedding.dim() != 3 or group_context.dim() != 3:
            raise ValueError(
                "Expected rank-3 tensors [B, N, D] or rank-2 [N, D] for both inputs."
            )
        if temporal_embedding.shape[:2] != group_context.shape[:2]:
            raise ValueError("temporal_embedding and group_context must share [B, N] dimensions.")

        x = torch.cat([temporal_embedding, group_context], dim=-1)  # [B, N, D_t + D_c]
        h = self.backbone(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = self.reparameterize(mu, logvar)

        if squeeze_batch:
            mu = mu.squeeze(0)
            logvar = logvar.squeeze(0)
            z = z.squeeze(0)

        return {"mu": mu, "logvar": logvar, "z": z}
