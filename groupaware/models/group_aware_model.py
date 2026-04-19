"""Integrated group-aware trajectory prediction model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from groupaware.graph.attention import AttentionConfig, EnhancedSpatialAttention
from groupaware.graph.hybrid_graph import HybridGraphConfig, build_hybrid_graph_sequence
from groupaware.graph.temporal_conv import TemporalConvConfig, TemporalConvStack
from groupaware.grouping.group_features import GroupFeatureConfig, compute_dynamic_group_features
from groupaware.models.gmm_decoder import GMMDecoderConfig, GroupAwareGMMDecoder
from groupaware.models.group_context import build_group_context_sequence
from groupaware.models.vae_encoder import GroupAwareVAEEncoder, VAEEncoderConfig


@dataclass(frozen=True)
class GroupAwareModelConfig:
    """Top-level integration config."""

    obs_len: int = 8
    pred_len: int = 12
    latent_dim: int = 5
    gmm_num_modes: int = 3
    mlp_dims: tuple[int, int, int] = (16, 512, 8)
    temporal_conv_layers: int = 7
    gamma_spatial: float = 0.25
    gamma_vel: float = 0.25
    gamma_dir: float = 0.25
    gamma_conflict: float = 0.25
    distance_threshold_m: float = 1.0
    velocity_diff_threshold_mps: float = 0.2
    directional_coherence_threshold: float = 0.95

    @staticmethod
    def from_dict(cfg: dict[str, Any]) -> "GroupAwareModelConfig":
        """Build typed config from repository YAML dictionary."""
        return GroupAwareModelConfig(
            obs_len=int(cfg["data"]["obs_len"]),
            pred_len=int(cfg["data"]["pred_len"]),
            latent_dim=int(cfg["model"]["latent_dim"]),
            gmm_num_modes=int(cfg["model"]["gmm_num_modes"]),
            mlp_dims=tuple(int(v) for v in cfg["model"]["mlp_dims"]),
            temporal_conv_layers=int(cfg["model"]["temporal_conv_layers"]),
            gamma_spatial=float(cfg["graph"]["gamma_spatial"]),
            gamma_vel=float(cfg["graph"]["gamma_vel"]),
            gamma_dir=float(cfg["graph"]["gamma_dir"]),
            gamma_conflict=float(cfg["graph"]["gamma_conflict"]),
            distance_threshold_m=float(cfg["grouping"]["distance_threshold_m"]),
            velocity_diff_threshold_mps=float(cfg["grouping"]["velocity_diff_threshold_mps"]),
            directional_coherence_threshold=float(cfg["grouping"]["directional_coherence_threshold"]),
        )


class GroupAwareTrajectoryModel(nn.Module):
    """
    End-to-end integration of grouping, graph, attention, temporal conv, VAE, and GMM.

    Engineering assumption:
    - Uses the last temporal-convolution timestep as pedestrian embedding for decoder.
    """

    def __init__(self, config: GroupAwareModelConfig) -> None:
        super().__init__()
        self.cfg = config

        self.group_cfg = GroupFeatureConfig(
            distance_threshold_m=config.distance_threshold_m,
            velocity_diff_threshold_mps=config.velocity_diff_threshold_mps,
            directional_coherence_threshold=config.directional_coherence_threshold,
        )
        self.graph_cfg = HybridGraphConfig(
            gamma_spatial=config.gamma_spatial,
            gamma_vel=config.gamma_vel,
            gamma_dir=config.gamma_dir,
            gamma_conflict=config.gamma_conflict,
        )

        self.attention = EnhancedSpatialAttention(
            AttentionConfig(input_dim=5, hidden_dim=int(config.mlp_dims[0]))
        )
        self.spatial_mlp = nn.Sequential(
            nn.Linear(config.mlp_dims[0], config.mlp_dims[1]),
            nn.ReLU(),
            nn.Linear(config.mlp_dims[1], config.mlp_dims[2]),
        )
        self.temporal = TemporalConvStack(
            TemporalConvConfig(
                input_dim=config.mlp_dims[2],
                hidden_dim=12,
                num_layers=config.temporal_conv_layers,
            )
        )
        self.encoder = GroupAwareVAEEncoder(
            VAEEncoderConfig(temporal_dim=12, context_dim=8, latent_dim=config.latent_dim)
        )
        self.decoder = GroupAwareGMMDecoder(
            GMMDecoderConfig(
                latent_dim=config.latent_dim,
                temporal_dim=12,
                context_dim=8,
                pred_len=config.pred_len,
                num_modes=config.gmm_num_modes,
            )
        )

    @staticmethod
    def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
        return x if isinstance(x, np.ndarray) else x.detach().cpu().numpy()

    def _forward_single(
        self,
        observed_positions: np.ndarray,
        observed_velocities: np.ndarray,
        observed_valid_mask: np.ndarray,
        device: torch.device,
    ) -> dict[str, Any]:
        t_obs, n, _ = observed_positions.shape

        grouping_out = compute_dynamic_group_features(
            positions=observed_positions,
            velocities=observed_velocities,
            valid_mask=observed_valid_mask,
            cfg=self.group_cfg,
        )
        graph_out = build_hybrid_graph_sequence(
            positions=observed_positions,
            velocities=observed_velocities,
            valid_mask=observed_valid_mask,
            grouping_output=grouping_out,
            config=self.graph_cfg,
        )
        context_seq = build_group_context_sequence(
            positions=observed_positions,
            velocities=observed_velocities,
            valid_mask=observed_valid_mask,
            grouping_output=grouping_out,
        )

        ped_spatial_seq = torch.zeros((t_obs, n, self.cfg.mlp_dims[2]), dtype=torch.float32, device=device)
        attention_maps: list[torch.Tensor] = []
        phi_maps: list[torch.Tensor] = []

        for t in range(t_obs):
            g_t = graph_out["per_timestep"][t]
            node_features = torch.from_numpy(g_t["node_features"]).to(device=device, dtype=torch.float32)
            adjacency = torch.from_numpy(g_t["adjacency"]).to(device=device, dtype=torch.float32)
            node_positions = torch.from_numpy(g_t["node_positions"]).to(device=device, dtype=torch.float32)
            node_velocities = torch.from_numpy(g_t["node_velocities"]).to(device=device, dtype=torch.float32)
            node_types = torch.from_numpy(g_t["node_types"]).to(device=device, dtype=torch.long)
            conflict = torch.from_numpy(g_t["conflict_softmax"]).to(device=device, dtype=torch.float32)

            attended, attn_weights, phi = self.attention(
                node_features=node_features,
                adjacency=adjacency,
                node_positions=node_positions,
                node_velocities=node_velocities,
                node_types=node_types,
                num_ped_nodes=int(g_t["num_ped_nodes"]),
                conflict_softmax_group=conflict,
            )
            ped_count = int(g_t["num_ped_nodes"])
            ped_indices = g_t["ped_original_indices"]
            ped_spatial_seq[t, ped_indices] = self.spatial_mlp(attended[:ped_count])
            attention_maps.append(attn_weights)
            phi_maps.append(phi)

        temporal_out = self.temporal(ped_spatial_seq)  # [T, N, 12]
        temporal_embed = temporal_out[-1]  # [N, 12]
        context_last = torch.from_numpy(context_seq[-1]).to(device=device, dtype=torch.float32)  # [N, 8]

        enc_out = self.encoder(temporal_embed, context_last)
        dec_out = self.decoder(enc_out["z"], temporal_embed, context_last)

        return {
            "encoder": enc_out,
            "decoder": dec_out,
            "temporal_embeddings": temporal_out,
            "spatial_embeddings": ped_spatial_seq,
            "group_context": torch.from_numpy(context_seq).to(device=device, dtype=torch.float32),
            "attention_maps": attention_maps,
            "phi_maps": phi_maps,
            "grouping": grouping_out,
            "graph": graph_out,
        }

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """
        Forward pass for collated ETH/UCY batch (Phase 2 schema).
        """
        device = next(self.parameters()).device
        observed = batch["observed"]
        pos_b = self._to_numpy(observed["positions"])
        vel_b = self._to_numpy(observed["velocities"])
        mask_b = self._to_numpy(observed["valid_mask"]).astype(bool)

        per_sample: list[dict[str, Any]] = []
        for b in range(pos_b.shape[0]):
            per_sample.append(
                self._forward_single(
                    observed_positions=pos_b[b],
                    observed_velocities=vel_b[b],
                    observed_valid_mask=mask_b[b],
                    device=device,
                )
            )

        # Stack model outputs across batch. Group/graph metadata is returned as list.
        mu = torch.stack([s["encoder"]["mu"] for s in per_sample], dim=0)
        logvar = torch.stack([s["encoder"]["logvar"] for s in per_sample], dim=0)
        z = torch.stack([s["encoder"]["z"] for s in per_sample], dim=0)

        means = torch.stack([s["decoder"]["means"] for s in per_sample], dim=0)
        stds = torch.stack([s["decoder"]["stds"] for s in per_sample], dim=0)
        mode_probs = torch.stack([s["decoder"]["mode_probs"] for s in per_sample], dim=0)
        mode_logits = torch.stack([s["decoder"]["mode_logits"] for s in per_sample], dim=0)
        det_mode = torch.stack([s["decoder"]["deterministic_mode_id"] for s in per_sample], dim=0)
        det_traj = torch.stack([s["decoder"]["deterministic_traj"] for s in per_sample], dim=0)

        temporal_embeddings = torch.stack([s["temporal_embeddings"] for s in per_sample], dim=0)
        spatial_embeddings = torch.stack([s["spatial_embeddings"] for s in per_sample], dim=0)
        group_context = torch.stack([s["group_context"] for s in per_sample], dim=0)

        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "means": means,
            "stds": stds,
            "mode_probs": mode_probs,
            "mode_logits": mode_logits,
            "deterministic_mode_id": det_mode,
            "deterministic_traj": det_traj,
            # Export-ready aliases:
            "multimodal_trajs": means,
            "multimodal_probs": mode_probs,
            "temporal_embeddings": temporal_embeddings,
            "spatial_embeddings": spatial_embeddings,
            "group_context": group_context,
            "group_metadata": [s["grouping"] for s in per_sample],
            "graph_metadata": [s["graph"] for s in per_sample],
            "attention_maps": [s["attention_maps"] for s in per_sample],
            "phi_maps": [s["phi_maps"] for s in per_sample],
        }
