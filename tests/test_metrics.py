"""Tests for ADE/FDE metrics."""

from __future__ import annotations

import torch

from groupaware.metrics import ade, best_of_m_ade, best_of_m_fde, fde


def test_metrics_basic_values() -> None:
    b, n, t = 1, 2, 12
    target = torch.zeros(b, n, t, 2)
    pred = torch.zeros_like(target)
    assert float(ade(pred, target)) == 0.0
    assert float(fde(pred, target)) == 0.0


def test_best_of_m_metrics() -> None:
    b, n, m, t = 1, 1, 3, 12
    target = torch.zeros(b, n, t, 2)
    preds = torch.ones(b, n, m, t, 2)
    preds[:, :, 1] = 0.0
    assert float(best_of_m_ade(preds, target)) == 0.0
    assert float(best_of_m_fde(preds, target)) == 0.0
