from __future__ import annotations

from typing import Any

import numpy as np


def _to_returns_array(returns: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(returns, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("returns must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError("returns must be finite")
    return arr


def compute_scenario_metrics(returns: list[float] | np.ndarray) -> dict[str, float]:
    arr = _to_returns_array(returns)
    equity_curve = np.cumprod(1.0 + arr)
    running_peak = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve / np.maximum(running_peak, 1e-12) - 1.0
    return {
        "return": float(equity_curve[-1] - 1.0),
        "vol": float(np.std(arr) * np.sqrt(252.0)),
        "max_drawdown_proxy": float(np.min(drawdown)),
    }


def historical_shock_scenarios(returns: list[float] | np.ndarray) -> dict[str, np.ndarray]:
    base = _to_returns_array(returns)
    scenarios = {
        "equity_crash": base.copy(),
        "vol_spike": base * 1.75,
        "spread_widening_proxy": base - 0.0015 - 0.20 * np.abs(base),
    }
    crash_n = max(1, min(5, base.size))
    scenarios["equity_crash"][:crash_n] -= 0.08
    return scenarios


def sample_scenario_paths(
    returns: list[float] | np.ndarray,
    n_paths: int = 100,
    horizon: int = 20,
    seed: int = 42,
    method: str = "bootstrap",
) -> np.ndarray:
    if n_paths <= 0:
        raise ValueError("n_paths must be > 0")
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    base = _to_returns_array(returns)
    rng = np.random.default_rng(seed)
    if method == "bootstrap":
        idx = rng.integers(0, base.size, size=(n_paths, horizon))
        return base[idx]
    if method == "monte_carlo":
        mu = float(np.mean(base))
        sigma = float(np.std(base))
        return rng.normal(loc=mu, scale=sigma, size=(n_paths, horizon))
    raise ValueError(f"unknown method: {method}")


def evaluate_stress_scenarios(
    returns: list[float] | np.ndarray,
    seed: int = 42,
    n_paths: int = 200,
    horizon: int = 20,
) -> dict[str, Any]:
    base = _to_returns_array(returns)
    historical = {
        name: compute_scenario_metrics(series)
        for name, series in historical_shock_scenarios(base).items()
    }

    bootstrap_paths = sample_scenario_paths(base, n_paths=n_paths, horizon=horizon, seed=seed, method="bootstrap")
    monte_carlo_paths = sample_scenario_paths(base, n_paths=n_paths, horizon=horizon, seed=seed + 1, method="monte_carlo")
    bootstrap_metrics = [compute_scenario_metrics(path) for path in bootstrap_paths]
    monte_carlo_metrics = [compute_scenario_metrics(path) for path in monte_carlo_paths]

    def _summary(items: list[dict[str, float]]) -> dict[str, float]:
        arr_return = np.array([x["return"] for x in items], dtype=float)
        arr_vol = np.array([x["vol"] for x in items], dtype=float)
        arr_mdd = np.array([x["max_drawdown_proxy"] for x in items], dtype=float)
        return {
            "return_mean": float(np.mean(arr_return)),
            "return_p05": float(np.quantile(arr_return, 0.05)),
            "vol_mean": float(np.mean(arr_vol)),
            "max_drawdown_proxy_p95": float(np.quantile(arr_mdd, 0.95)),
        }

    return {
        "seed": int(seed),
        "n_paths": int(n_paths),
        "horizon": int(horizon),
        "historical": historical,
        "sampling": {
            "bootstrap": _summary(bootstrap_metrics),
            "monte_carlo": _summary(monte_carlo_metrics),
        },
    }
