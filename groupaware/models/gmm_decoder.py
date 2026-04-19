"""Group-aware GMM decoder for multimodal trajectory prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GMMDecoderConfig:
    """Configuration for GMM decoder."""

    latent_dim: int = 5
    temporal_dim: int = 12
    context_dim: int = 8
    hidden_dim: int = 64
    pred_len: int = 12
    num_modes: int = 3
    min_std: float = 1e-3


class GroupAwareGMMDecoder(nn.Module):
    """
    Decode latent + temporal + group context into multimodal trajectory distribution.

    Outputs:
    - means: [B, N, M, T_pred, 2]
    - stds: [B, N, M, T_pred, 2] (diagonal covariance std)
    - mode_probs: [B, N, M]
    - deterministic_mode_id: [B, N]
    - deterministic_traj: [B, N, T_pred, 2]
    """

    def __init__(self, config: GMMDecoderConfig) -> None:
        super().__init__()
        self.config = config
        in_dim = config.latent_dim + config.temporal_dim + config.context_dim
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )

        traj_dim = config.num_modes * config.pred_len * 2
        self.mean_head = nn.Linear(config.hidden_dim, traj_dim)
        self.std_head = nn.Linear(config.hidden_dim, traj_dim)
        self.mode_head = nn.Linear(config.hidden_dim, config.num_modes)
        self.softplus = nn.Softplus()

    def forward(
        self,
        z: torch.Tensor,
        temporal_embedding: torch.Tensor,
        group_context: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward decode into GMM parameters and deterministic selected trajectory."""
        squeeze_batch = False
        if z.dim() == 2 and temporal_embedding.dim() == 2 and group_context.dim() == 2:
            z = z.unsqueeze(0)
            temporal_embedding = temporal_embedding.unsqueeze(0)
            group_context = group_context.unsqueeze(0)
            squeeze_batch = True

        if z.dim() != 3 or temporal_embedding.dim() != 3 or group_context.dim() != 3:
            raise ValueError("Expected rank-3 [B, N, D] or rank-2 [N, D] for all inputs.")
        if not (z.shape[:2] == temporal_embedding.shape[:2] == group_context.shape[:2]):
            raise ValueError("z, temporal_embedding, and group_context must share [B, N] dimensions.")

        b, n, _ = z.shape
        m = self.config.num_modes
        t = self.config.pred_len

        x = torch.cat([z, temporal_embedding, group_context], dim=-1)  # [B, N, D]
        h = self.backbone(x)

        means = self.mean_head(h).view(b, n, m, t, 2)
        stds = self.softplus(self.std_head(h)).view(b, n, m, t, 2) + self.config.min_std
        mode_logits = self.mode_head(h)
        mode_probs = torch.softmax(mode_logits, dim=-1)

        # Deterministic selection by highest mixture probability.
        det_mode = torch.argmax(mode_probs, dim=-1)  # [B, N]
        gather_idx = det_mode.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, t, 2)
        det_traj = torch.gather(means, dim=2, index=gather_idx).squeeze(2)  # [B, N, T, 2]

        out = {
            "means": means,
            "stds": stds,
            "mode_probs": mode_probs,
            "mode_logits": mode_logits,
            "deterministic_mode_id": det_mode,
            "deterministic_traj": det_traj,
            # Export-friendly aliases:
            "multimodal_trajs": means,
            "multimodal_probs": mode_probs,
        }

        if squeeze_batch:
            out = {k: v.squeeze(0) for k, v in out.items()}
        return out
