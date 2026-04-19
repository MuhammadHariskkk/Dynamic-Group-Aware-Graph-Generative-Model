"""Model modules for group-aware trajectory prediction."""

from groupaware.models.gmm_decoder import GMMDecoderConfig, GroupAwareGMMDecoder
from groupaware.models.vae_encoder import GroupAwareVAEEncoder, VAEEncoderConfig

__all__ = [
    "GMMDecoderConfig",
    "GroupAwareGMMDecoder",
    "GroupAwareVAEEncoder",
    "VAEEncoderConfig",
]
