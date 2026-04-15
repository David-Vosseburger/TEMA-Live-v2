from __future__ import annotations

from typing import Sequence

import numpy as np


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def score_rf_probabilities(
    expected_alphas: Sequence[float],
    regime_probabilities: Sequence[float],
    alpha_weight: float = 1.0,
    regime_weight: float = 0.5,
    bias: float = 0.0,
) -> list[float]:
    """RF-like probability scorer with deterministic fallback math."""
    if len(expected_alphas) != len(regime_probabilities):
        raise ValueError("expected_alphas and regime_probabilities length mismatch")
    if len(expected_alphas) == 0:
        return []

    alphas = np.asarray(expected_alphas, dtype=float)
    regimes = np.asarray(regime_probabilities, dtype=float)
    alpha_scale = float(np.std(alphas)) or 1.0
    normalized_alpha = alphas / alpha_scale

    logits = alpha_weight * normalized_alpha + regime_weight * (regimes - 0.5) + bias
    return [_sigmoid(float(v)) for v in logits]
