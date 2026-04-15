from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class OnlineLearningConfig:
    enabled: bool = False
    learning_rate: float = 0.10
    l2: float = 1e-4
    seed: int = 42


class OnlineLogisticLearner:
    """Tiny dependency-light online learner with deterministic reset."""

    def __init__(self, n_features: int, learning_rate: float = 0.10, l2: float = 1e-4, seed: int = 42):
        if n_features <= 0:
            raise ValueError("n_features must be > 0")
        self.n_features = int(n_features)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.weights = self._rng.normal(loc=0.0, scale=1e-3, size=self.n_features).astype(float)
        self.bias = float(self._rng.normal(loc=0.0, scale=1e-3))
        self.n_updates = 0

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.weights = self._rng.normal(loc=0.0, scale=1e-3, size=self.n_features).astype(float)
        self.bias = float(self._rng.normal(loc=0.0, scale=1e-3))
        self.n_updates = 0

    def _sigmoid(self, z: float) -> float:
        z = float(np.clip(z, -35.0, 35.0))
        return 1.0 / (1.0 + math.exp(-z))

    def predict_score(self, x: Sequence[float]) -> float:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.shape != (self.n_features,):
            raise ValueError(f"expected feature shape ({self.n_features},), got {x_arr.shape}")
        return self._sigmoid(float(np.dot(self.weights, x_arr) + self.bias))

    def partial_fit(self, x: Sequence[float], y: int | float) -> float:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.shape != (self.n_features,):
            raise ValueError(f"expected feature shape ({self.n_features},), got {x_arr.shape}")
        y_val = 1.0 if float(y) >= 0.5 else 0.0
        p = self.predict_score(x_arr)
        err = p - y_val
        grad_w = err * x_arr + self.l2 * self.weights
        grad_b = err
        self.weights = self.weights - self.learning_rate * grad_w
        self.bias = float(self.bias - self.learning_rate * grad_b)
        self.n_updates += 1
        return p

    def snapshot(self) -> dict[str, Any]:
        return {
            "n_features": self.n_features,
            "learning_rate": float(self.learning_rate),
            "l2": float(self.l2),
            "seed": int(self.seed),
            "n_updates": int(self.n_updates),
            "weights": self.weights.tolist(),
            "bias": float(self.bias),
        }
