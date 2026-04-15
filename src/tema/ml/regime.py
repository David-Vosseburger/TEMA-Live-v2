from __future__ import annotations

from typing import Sequence

import numpy as np


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def score_regime_probabilities(expected_alphas: Sequence[float], temperature: float = 5.0) -> list[float]:
    """Simple HMM-style regime proxy returning per-asset state probabilities.

    Uses a bounded transform over standardized alphas as a deterministic fallback.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if len(expected_alphas) == 0:
        return []
    arr = np.asarray(expected_alphas, dtype=float)
    std = float(np.std(arr))
    if std == 0.0:
        standardized = arr
    else:
        standardized = arr / std
    return [_sigmoid(float(x) / temperature) for x in standardized]
