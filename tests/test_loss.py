"""Tests for loss functions."""

from __future__ import annotations

import torch

from groupaware.losses import gmm_nll_loss, kl_divergence_standard_normal


def test_losses_are_finite() -> None:
    b, n, m, t = 2, 3, 3, 12
    means = torch.randn(b, n, m, t, 2)
    stds = torch.rand(b, n, m, t, 2) + 0.1
    probs = torch.softmax(torch.randn(b, n, m), dim=-1)
    tgt = torch.randn(b, n, t, 2)
    vm = torch.ones(b, n, t, dtype=torch.bool)
    nll = gmm_nll_loss(means, stds, probs, tgt, valid_mask=vm)
    kl = kl_divergence_standard_normal(torch.randn(b, n, 5), torch.randn(b, n, 5))
    assert torch.isfinite(nll)
    assert torch.isfinite(kl)
