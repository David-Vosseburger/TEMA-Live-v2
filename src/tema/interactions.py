from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class InteractionScore:
    feature_i: str
    feature_j: str
    score: float


def _safe_abs_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    xc = x - float(np.mean(x))
    yc = y - float(np.mean(y))
    denom = float(np.linalg.norm(xc) * np.linalg.norm(yc))
    if denom <= 1e-12:
        return 0.0
    return float(abs(np.dot(xc, yc) / denom))


def compute_pairwise_interaction_scores(
    feature_matrix: np.ndarray,
    target_proxy: np.ndarray,
    feature_names: Sequence[str] | None = None,
) -> list[InteractionScore]:
    x = np.asarray(feature_matrix, dtype=float)
    y = np.asarray(target_proxy, dtype=float).reshape(-1)
    if x.ndim != 2:
        raise ValueError("feature_matrix must be 2D")
    if y.ndim != 1:
        raise ValueError("target_proxy must be 1D")
    if x.shape[0] != y.shape[0]:
        raise ValueError("feature_matrix and target_proxy must have matching rows")

    n_features = int(x.shape[1])
    names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(n_features)]
    if len(names) != n_features:
        raise ValueError("feature_names length must match feature count")

    scores: list[InteractionScore] = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            cross = x[:, i] * x[:, j]
            score = _safe_abs_pearson(cross, y)
            scores.append(InteractionScore(feature_i=names[i], feature_j=names[j], score=score))

    scores.sort(key=lambda item: (-item.score, item.feature_i, item.feature_j))
    return scores


def select_top_k_interactions(scores: Sequence[InteractionScore], k: int) -> list[InteractionScore]:
    if k <= 0:
        return []
    return list(scores[:k])


def generate_feature_crosses(
    feature_matrix: np.ndarray,
    selected_pairs: Iterable[tuple[str, str] | InteractionScore],
    feature_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    x = np.asarray(feature_matrix, dtype=float)
    if x.ndim != 2:
        raise ValueError("feature_matrix must be 2D")

    n_features = int(x.shape[1])
    names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(n_features)]
    if len(names) != n_features:
        raise ValueError("feature_names length must match feature count")
    name_to_idx = {name: idx for idx, name in enumerate(names)}

    crosses: list[np.ndarray] = []
    cross_names: list[str] = []
    for pair in selected_pairs:
        left, right = (pair.feature_i, pair.feature_j) if isinstance(pair, InteractionScore) else pair
        if left not in name_to_idx or right not in name_to_idx:
            raise ValueError(f"Unknown feature in pair: ({left}, {right})")
        li, ri = name_to_idx[left], name_to_idx[right]
        crosses.append((x[:, li] * x[:, ri]).reshape(-1, 1))
        cross_names.append(f"{left}__x__{right}")

    if not crosses:
        return np.empty((x.shape[0], 0), dtype=float), []
    return np.hstack(crosses), cross_names
