from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PortfolioAllocationResult:
    weights: list[float]
    method: str
    used_fallback: bool
    diagnostics: dict


def _normalize_long_only(weights: np.ndarray, w_min: float = 0.0, w_max: float = 1.0) -> np.ndarray:
    x = np.asarray(weights, dtype=float).reshape(-1)
    if x.size == 0:
        return x
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, max(0.0, w_min), w_max)
    total = float(x.sum())
    if total <= 1e-12:
        return np.full_like(x, 1.0 / float(x.size))
    return x / total


def _covariance_with_shrinkage(returns_window: np.ndarray | None, n_assets: int, shrinkage: float) -> np.ndarray:
    if returns_window is None or returns_window.size == 0:
        return np.eye(n_assets, dtype=float)
    x = np.asarray(returns_window, dtype=float)
    if x.ndim != 2 or x.shape[1] != n_assets:
        return np.eye(n_assets, dtype=float)
    cov = np.cov(x, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=float)
    diag = np.diag(np.diag(cov))
    shrink = min(max(float(shrinkage), 0.0), 1.0)
    return (1.0 - shrink) * cov + shrink * diag + 1e-8 * np.eye(n_assets)


def _black_litterman_like(
    expected_alphas: np.ndarray,
    cov: np.ndarray,
    signals: np.ndarray | None,
    tau: float,
    risk_aversion: float,
    view_confidence: float,
) -> np.ndarray:
    n_assets = expected_alphas.size
    pi = np.zeros(n_assets, dtype=float)
    if signals is not None and signals.size == n_assets:
        pi = 0.5 * np.asarray(signals, dtype=float)
    q = np.asarray(expected_alphas, dtype=float)
    p = np.eye(n_assets, dtype=float)

    tau_cov = max(float(tau), 1e-6) * cov
    confidence = min(max(float(view_confidence), 1e-3), 0.999)
    omega_scale = (1.0 - confidence) / confidence
    omega = omega_scale * np.diag(np.diag(cov) + 1e-8)

    inv_tau_cov = np.linalg.pinv(tau_cov)
    inv_omega = np.linalg.pinv(omega)
    lhs = inv_tau_cov + p.T @ inv_omega @ p
    rhs = inv_tau_cov @ pi + p.T @ inv_omega @ q
    posterior = np.linalg.pinv(lhs) @ rhs
    w = np.linalg.pinv(risk_aversion * cov) @ posterior
    return w.reshape(-1)


def _mean_variance_fallback(
    expected_alphas: np.ndarray,
    cov: np.ndarray,
    risk_aversion: float,
) -> np.ndarray:
    target = expected_alphas.reshape(-1)
    w = np.linalg.pinv(risk_aversion * cov) @ target
    return w.reshape(-1)


def hrp_allocation_hook(expected_alphas: Sequence[float]) -> PortfolioAllocationResult:
    n_assets = len(expected_alphas)
    if n_assets == 0:
        return PortfolioAllocationResult(weights=[], method="hrp-hook", used_fallback=True, diagnostics={"hook": "empty"})
    w = [1.0 / n_assets for _ in range(n_assets)]
    return PortfolioAllocationResult(
        weights=w,
        method="hrp-hook",
        used_fallback=True,
        diagnostics={"hook": "placeholder-deterministic-equal-weight"},
    )


def nco_allocation_hook(expected_alphas: Sequence[float]) -> PortfolioAllocationResult:
    n_assets = len(expected_alphas)
    if n_assets == 0:
        return PortfolioAllocationResult(weights=[], method="nco-hook", used_fallback=True, diagnostics={"hook": "empty"})
    abs_alpha = np.abs(np.asarray(expected_alphas, dtype=float))
    total = float(abs_alpha.sum())
    if total <= 1e-12:
        w = [1.0 / n_assets for _ in range(n_assets)]
    else:
        w = (abs_alpha / total).tolist()
    return PortfolioAllocationResult(
        weights=w,
        method="nco-hook",
        used_fallback=True,
        diagnostics={"hook": "placeholder-deterministic-abs-alpha"},
    )


def allocate_portfolio_weights(
    expected_alphas: Sequence[float],
    returns_window: np.ndarray | None,
    signals: Sequence[float] | None = None,
    method: str = "bl",
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    view_confidence: float = 0.65,
    cov_shrinkage: float = 0.15,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> PortfolioAllocationResult:
    alpha = np.asarray(expected_alphas, dtype=float).reshape(-1)
    n_assets = int(alpha.size)
    if n_assets == 0:
        return PortfolioAllocationResult(weights=[], method=method, used_fallback=True, diagnostics={"reason": "no-assets"})

    signal_vec = None if signals is None else np.asarray(signals, dtype=float).reshape(-1)
    cov = _covariance_with_shrinkage(returns_window=returns_window, n_assets=n_assets, shrinkage=cov_shrinkage)

    selected_method = (method or "bl").lower()
    used_fallback = False
    diagnostics: dict = {"n_assets": n_assets}
    try:
        if selected_method == "hrp":
            return hrp_allocation_hook(alpha.tolist())
        if selected_method == "nco":
            return nco_allocation_hook(alpha.tolist())
        if selected_method in ("mv", "mean_variance"):
            raw = _mean_variance_fallback(alpha, cov=cov, risk_aversion=max(risk_aversion, 1e-6))
            selected_method = "mean_variance"
        else:
            raw = _black_litterman_like(
                expected_alphas=alpha,
                cov=cov,
                signals=signal_vec,
                tau=tau,
                risk_aversion=max(risk_aversion, 1e-6),
                view_confidence=view_confidence,
            )
            selected_method = "black_litterman_like"
    except Exception as exc:
        used_fallback = True
        diagnostics["fallback_reason"] = str(exc)
        raw = _mean_variance_fallback(alpha, cov=np.eye(n_assets, dtype=float), risk_aversion=max(risk_aversion, 1e-6))
        selected_method = "mean_variance_fallback"

    weights = _normalize_long_only(raw, w_min=min_weight, w_max=max_weight)
    diagnostics["sum_weights"] = float(weights.sum())
    diagnostics["min_weight"] = float(weights.min()) if weights.size else 0.0
    diagnostics["max_weight"] = float(weights.max()) if weights.size else 0.0
    return PortfolioAllocationResult(
        weights=weights.tolist(),
        method=selected_method,
        used_fallback=used_fallback,
        diagnostics=diagnostics,
    )
