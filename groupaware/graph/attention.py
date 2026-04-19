"""Enhanced spatial attention with behavioural context vector phi_ij."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class AttentionConfig:
    """Configuration for socially informed spatial attention."""

    input_dim: int
    hidden_dim: int = 16
    context_dim: int = 4  # [distance, velocity_diff, dir_align, normalized_conflict]
    negative_slope: float = 0.2
    eps: float = 1e-8


class EnhancedSpatialAttention(nn.Module):
    """
    Compute socially-informed attention over hybrid graph nodes.

    Paper alignment:
    - phi_ij = [distance, velocity difference, directional alignment, normalized conflict]
    - attention score uses node features and explicit behavioural context.
    """

    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.proj = nn.Linear(config.input_dim, config.hidden_dim, bias=False)
        self.score = nn.Linear((2 * config.hidden_dim) + config.context_dim, 1, bias=False)
        self.act = nn.LeakyReLU(negative_slope=config.negative_slope)

    def _build_phi(
        self,
        node_positions: torch.Tensor,
        node_velocities: torch.Tensor,
        node_types: torch.Tensor,
        num_ped_nodes: int,
        conflict_softmax_group: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Build phi tensor with shape [V, V, 4].
        """
        v = node_positions.shape[0]
        eps = self.config.eps

        pos_i = node_positions.unsqueeze(1).expand(v, v, 2)
        pos_j = node_positions.unsqueeze(0).expand(v, v, 2)
        vel_i = node_velocities.unsqueeze(1).expand(v, v, 2)
        vel_j = node_velocities.unsqueeze(0).expand(v, v, 2)

        dist = torch.norm(pos_i - pos_j, dim=-1)
        vel_diff = torch.norm(vel_i - vel_j, dim=-1)

        speed_i = torch.norm(vel_i, dim=-1, keepdim=True)
        speed_j = torch.norm(vel_j, dim=-1, keepdim=True)
        dir_i = vel_i / (speed_i + eps)
        dir_j = vel_j / (speed_j + eps)
        dir_align = torch.sum(dir_i * dir_j, dim=-1).clamp(min=-1.0, max=1.0)

        norm_conf = torch.zeros((v, v), dtype=node_positions.dtype, device=node_positions.device)
        if conflict_softmax_group is not None and conflict_softmax_group.numel() > 0:
            group_mask_i = node_types == 1
            group_mask_j = node_types == 1
            gi_idx = torch.where(group_mask_i)[0]
            gj_idx = torch.where(group_mask_j)[0]
            for i_node in gi_idx.tolist():
                for j_node in gj_idx.tolist():
                    gi = i_node - num_ped_nodes
                    gj = j_node - num_ped_nodes
                    if 0 <= gi < conflict_softmax_group.shape[0] and 0 <= gj < conflict_softmax_group.shape[1]:
                        norm_conf[i_node, j_node] = conflict_softmax_group[gi, gj]

        phi = torch.stack([dist, vel_diff, dir_align, norm_conf], dim=-1)
        return phi

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        node_positions: torch.Tensor,
        node_velocities: torch.Tensor,
        node_types: torch.Tensor,
        num_ped_nodes: int,
        conflict_softmax_group: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: [V, F]
            adjacency: [V, V] row-softmax graph adjacency (acts as structural prior mask/weight)
            node_positions: [V, 2]
            node_velocities: [V, 2]
            node_types: [V] (0 pedestrian, 1 group)
            num_ped_nodes: scalar
            conflict_softmax_group: [G, G] or None
        Returns:
            attended_features: [V, H]
            attention_weights: [V, V]
            phi: [V, V, 4]
        """
        v = node_features.shape[0]
        h = self.proj(node_features)  # [V, H]
        phi = self._build_phi(
            node_positions=node_positions,
            node_velocities=node_velocities,
            node_types=node_types,
            num_ped_nodes=num_ped_nodes,
            conflict_softmax_group=conflict_softmax_group,
        )  # [V, V, 4]

        h_i = h.unsqueeze(1).expand(v, v, h.shape[-1])
        h_j = h.unsqueeze(0).expand(v, v, h.shape[-1])
        pair = torch.cat([h_i, h_j, phi], dim=-1)  # [V, V, 2H+4]
        logits = self.act(self.score(pair).squeeze(-1))  # [V, V]

        # Use adjacency to preserve graph structure and reinforce dynamic edge prior.
        # If adjacency[i,j] == 0, suppress that edge in attention.
        suppress = torch.where(adjacency > 0, torch.zeros_like(logits), torch.full_like(logits, -1e9))
        attn = torch.softmax(logits + suppress, dim=1)  # [V, V]

        # Blend learned attention with adjacency prior.
        attn = attn * adjacency
        attn = attn / (attn.sum(dim=1, keepdim=True) + self.config.eps)

        attended = attn @ h  # [V, H]
        return attended, attn, phi
