from __future__ import annotations

from typing import Sequence


def compute_position_scalars(
    probabilities: Sequence[float],
    floor: float,
    ceiling: float,
    decisions: Sequence[float] | None = None,
) -> list[float]:
    """Map probabilities to scalar range and apply optional decision mask."""
    if floor < 0 or ceiling < 0:
        raise ValueError("floor and ceiling must be non-negative")
    if ceiling < floor:
        raise ValueError("ceiling must be >= floor")

    scalars: list[float] = []
    for i, p in enumerate(probabilities):
        prob = min(1.0, max(0.0, float(p)))
        scalar = float(floor + (ceiling - floor) * prob)
        if decisions is not None:
            if i >= len(decisions):
                raise ValueError("decisions length mismatch")
            scalar *= float(decisions[i])
        scalars.append(scalar)
    if decisions is not None and len(decisions) != len(probabilities):
        raise ValueError("decisions length mismatch")
    return scalars
