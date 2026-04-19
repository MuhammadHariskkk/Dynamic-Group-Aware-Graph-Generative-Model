"""Loss functions for group-aware trajectory model."""

from groupaware.losses.gmm_nll import gmm_nll_loss
from groupaware.losses.kl_divergence import kl_divergence_standard_normal

__all__ = ["gmm_nll_loss", "kl_divergence_standard_normal"]
