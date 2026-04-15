from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

STREAM_NAMES = ("tema_base", "ml_proxy", "risk_proxy")


def _annualized_sharpe(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    std = float(np.std(returns))
    if std <= 1e-12:
        return 0.0
    return float(np.mean(returns) / std * math.sqrt(252.0))


@dataclass(frozen=True)
class DynamicEnsembleConfig:
    enabled: bool = False
    lookback: int = 20
    ridge_shrink: float = 0.15
    min_weight: float = 0.05
    max_weight: float = 0.90
    regime_sensitivity: float = 0.40


def normalize_bounded_weights(weights: Sequence[float], min_weight: float, max_weight: float) -> np.ndarray:
    arr = np.asarray(weights, dtype=float)
    if arr.size == 0:
        raise ValueError("weights must not be empty")
    if min_weight < 0.0 or max_weight <= 0.0 or min_weight > max_weight:
        raise ValueError("invalid weight bounds")
    if min_weight * arr.size > 1.0 + 1e-12:
        raise ValueError("min_weight too large for number of streams")
    if max_weight * arr.size < 1.0 - 1e-12:
        raise ValueError("max_weight too small for number of streams")

    s = float(arr.sum())
    if not np.isfinite(s) or s <= 0.0:
        arr = np.ones_like(arr, dtype=float) / float(arr.size)
    else:
        arr = arr / s
    arr = np.clip(arr, min_weight, max_weight)

    for _ in range(16):
        total = float(arr.sum())
        gap = 1.0 - total
        if abs(gap) <= 1e-12:
            break
        if gap > 0:
            room = np.maximum(0.0, max_weight - arr)
            room_sum = float(room.sum())
            if room_sum <= 1e-12:
                break
            arr = arr + gap * (room / room_sum)
        else:
            room = np.maximum(0.0, arr - min_weight)
            room_sum = float(room.sum())
            if room_sum <= 1e-12:
                break
            arr = arr + gap * (room / room_sum)
        arr = np.clip(arr, min_weight, max_weight)

    arr = arr / float(arr.sum())
    return arr


def compute_dynamic_ensemble_weights(
    stream_returns: Mapping[str, Sequence[float]],
    cfg: DynamicEnsembleConfig,
    regime_score: float = 0.0,
    stream_names: Sequence[str] | None = None,
) -> dict[str, float]:
    names = tuple(stream_names) if stream_names is not None else STREAM_NAMES
    if not names:
        raise ValueError("stream_names must not be empty")
    regime = float(np.clip(regime_score, -1.0, 1.0))
    performance = []

    for name in names:
        history = np.asarray(stream_returns.get(name, []), dtype=float)
        if history.size > cfg.lookback:
            history = history[-cfg.lookback :]
        sharpe = max(0.0, _annualized_sharpe(history))
        score = sharpe + 1e-6
        if name == "risk_proxy":
            score *= 1.0 + cfg.regime_sensitivity * max(0.0, -regime)
        else:
            score *= 1.0 + cfg.regime_sensitivity * max(0.0, regime)
        performance.append(score)

    base = np.asarray(performance, dtype=float)
    base = base / float(base.sum())
    shrink = float(np.clip(cfg.ridge_shrink, 0.0, 1.0))
    equal = np.ones(len(names), dtype=float) / float(len(names))
    mixed = (1.0 - shrink) * base + shrink * equal
    bounded = normalize_bounded_weights(mixed, cfg.min_weight, cfg.max_weight)
    return {name: float(val) for name, val in zip(names, bounded)}


def combine_stream_signals(
    stream_signals: Mapping[str, Sequence[float]],
    weights: Mapping[str, float],
    stream_names: Sequence[str] | None = None,
) -> list[float]:
    names = tuple(stream_names) if stream_names is not None else STREAM_NAMES
    if not names:
        raise ValueError("stream_names must not be empty")
    lengths = {len(stream_signals.get(name, [])) for name in names}
    if len(lengths) != 1:
        raise ValueError("all stream signal vectors must have equal length")
    n = next(iter(lengths))
    out = np.zeros(n, dtype=float)
    for name in names:
        signal = np.asarray(stream_signals.get(name, []), dtype=float)
        out += float(weights.get(name, 0.0)) * signal
    return out.tolist()
