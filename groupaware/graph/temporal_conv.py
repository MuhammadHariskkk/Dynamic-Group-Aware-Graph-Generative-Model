"""Temporal convolution module for attended node sequences."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TemporalConvConfig:
    """Configuration for temporal convolution stack."""

    input_dim: int = 8
    hidden_dim: int = 12
    num_layers: int = 7
    kernel_size: int = 3
    dropout: float = 0.1


class TemporalConvStack(nn.Module):
    """
    1D temporal convolution over observed timesteps per node.

    Paper alignment:
    - default uses seven temporal convolution layers.
    - first layer projects 8 -> 12, subsequent layers keep 12.
    """

    def __init__(self, config: TemporalConvConfig) -> None:
        super().__init__()
        self.config = config
        if config.num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers: list[nn.Module] = []
        in_ch = config.input_dim
        for _ in range(config.num_layers):
            out_ch = config.hidden_dim
            layers.append(
                nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=config.kernel_size,
                    padding=config.kernel_size // 2,
                )
            )
            layers.append(nn.ReLU())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T_obs, V, F] or [T_obs, V, F]
        Returns:
            y: same rank as input with feature dim replaced by hidden_dim.
               - [B, T_obs, V, H] or [T_obs, V, H]
        """
        squeeze_batch = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_batch = True
        if x.dim() != 4:
            raise ValueError(f"Expected input rank 3 or 4, got shape {tuple(x.shape)}")

        b, t, v, f = x.shape
        x_perm = x.permute(0, 2, 3, 1).contiguous()  # [B, V, F, T]
        x_flat = x_perm.view(b * v, f, t)  # [B*V, F, T]
        y = self.net(x_flat)  # [B*V, H, T]
        h = y.shape[1]
        y = y.view(b, v, h, t).permute(0, 3, 1, 2).contiguous()  # [B, T, V, H]

        if squeeze_batch:
            return y.squeeze(0)
        return y
