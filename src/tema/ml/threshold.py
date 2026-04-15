from __future__ import annotations

from typing import Sequence


def threshold_probabilities(probabilities: Sequence[float], threshold: float = 0.0) -> list[float]:
    """Binary decision mask from probabilities."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    out: list[float] = []
    for p in probabilities:
        prob = float(p)
        out.append(1.0 if prob >= threshold else 0.0)
    return out
