from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from typing import Any

import numpy as np
from .ensemble import DynamicEnsembleConfig, compute_dynamic_ensemble_weights
from .interactions import (
    compute_pairwise_interaction_scores,
    generate_feature_crosses,
    select_top_k_interactions,
)
from .online_learning import OnlineLearningConfig, OnlineLogisticLearner
from .stress import evaluate_stress_scenarios


def _annualized_sharpe(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    std = float(np.std(returns))
    if std <= 1e-12:
        return 0.0
    return float(np.mean(returns) / std * math.sqrt(252.0))


def compute_objective(
    train_sharpe: float,
    val_sharpe: float,
    annualized_turnover: float,
    turnover_penalty_lambda: float,
    overfit_guard_lambda: float,
) -> float:
    overfit_gap = max(0.0, float(train_sharpe) - float(val_sharpe))
    return float(val_sharpe - turnover_penalty_lambda * annualized_turnover - overfit_guard_lambda * overfit_gap)


@dataclass(frozen=True)
class SearchSpace:
    ema_fast: int
    ema_slow: int
    rebalance_threshold: float
    turnover_penalty_lambda: float
    ml_scalar: float
    vol_target: float


def _simulate_metrics(
    params: SearchSpace,
    seed: int = 42,
    n_periods: int = 192,
    train_ratio: float = 0.70,
    ensemble_cfg: DynamicEnsembleConfig | None = None,
    online_cfg: OnlineLearningConfig | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0004, scale=0.010, size=n_periods)

    ema_fast = 0.0
    ema_slow = 0.0
    alpha_fast = 2.0 / (params.ema_fast + 1.0)
    alpha_slow = 2.0 / (params.ema_slow + 1.0)

    weights = np.zeros(n_periods, dtype=float)
    strategy_returns = np.zeros(n_periods, dtype=float)
    stream_returns = {
        "tema_base": np.zeros(n_periods, dtype=float),
        "ml_proxy": np.zeros(n_periods, dtype=float),
        "risk_proxy": np.zeros(n_periods, dtype=float),
    }
    ensemble_weights_trace = {
        "tema_base": np.zeros(n_periods, dtype=float),
        "ml_proxy": np.zeros(n_periods, dtype=float),
        "risk_proxy": np.zeros(n_periods, dtype=float),
    }
    eff_online_cfg = online_cfg if online_cfg is not None else OnlineLearningConfig(enabled=False)
    online_learner: OnlineLogisticLearner | None = None
    if eff_online_cfg.enabled:
        stream_returns["online_learning"] = np.zeros(n_periods, dtype=float)
        ensemble_weights_trace["online_learning"] = np.zeros(n_periods, dtype=float)
        online_learner = OnlineLogisticLearner(
            n_features=4,
            learning_rate=eff_online_cfg.learning_rate,
            l2=eff_online_cfg.l2,
            seed=eff_online_cfg.seed,
        )

    prev_w = 0.0
    fee_proxy = 0.0006 * (1.0 + 0.15 * params.turnover_penalty_lambda)
    leverage = max(0.25, min(3.0, params.vol_target / 0.10))
    eff_cfg = ensemble_cfg if ensemble_cfg is not None else DynamicEnsembleConfig(enabled=False)
    stream_names = list(stream_returns.keys())
    equal_w = 1.0 / float(len(stream_names))
    equal_weights = {name: equal_w for name in stream_names}

    for t, ret in enumerate(returns):
        ema_fast = alpha_fast * ret + (1.0 - alpha_fast) * ema_fast
        ema_slow = alpha_slow * ret + (1.0 - alpha_slow) * ema_slow
        base_signal = ema_fast - ema_slow
        ml_signal = base_signal * params.ml_scalar
        risk_signal = -abs(ret)
        online_signal = 0.0
        online_features = np.array([base_signal, ml_signal, risk_signal, ret], dtype=float)
        if online_learner is not None:
            online_signal = 2.0 * online_learner.predict_score(online_features) - 1.0
        stream_returns["tema_base"][t] = float(np.tanh(base_signal * leverage * 14.0) * ret)
        stream_returns["ml_proxy"][t] = float(np.tanh(ml_signal * leverage * 14.0) * ret)
        stream_returns["risk_proxy"][t] = float(np.tanh(risk_signal * leverage * 8.0) * ret)
        if online_learner is not None:
            stream_returns["online_learning"][t] = float(np.tanh(online_signal * leverage * 10.0) * ret)

        if eff_cfg.enabled and t > 0:
            regime_window = returns[max(0, t - eff_cfg.lookback) : t]
            regime_score = float(np.tanh(np.mean(regime_window) * 100.0)) if regime_window.size else 0.0
            dyn_weights = compute_dynamic_ensemble_weights(
                stream_returns={k: v[:t].tolist() for k, v in stream_returns.items()},
                cfg=eff_cfg,
                regime_score=regime_score,
                stream_names=stream_names,
            )
        else:
            dyn_weights = equal_weights
        for name in ensemble_weights_trace:
            ensemble_weights_trace[name][t] = float(dyn_weights[name])

        ensemble_signal = (
            float(dyn_weights["tema_base"]) * base_signal
            + float(dyn_weights["ml_proxy"]) * ml_signal
            + float(dyn_weights["risk_proxy"]) * risk_signal
        )
        if online_learner is not None:
            ensemble_signal += float(dyn_weights["online_learning"]) * online_signal
        raw_signal = ensemble_signal * leverage * 14.0
        target_w = float(np.tanh(raw_signal))

        if abs(target_w - prev_w) < params.rebalance_threshold:
            w = prev_w
            turnover_step = 0.0
        else:
            w = target_w
            turnover_step = abs(w - prev_w)

        strategy_returns[t] = w * ret - turnover_step * fee_proxy
        weights[t] = w
        prev_w = w
        if online_learner is not None:
            online_learner.partial_fit(online_features, 1.0 if ret > 0.0 else 0.0)

    diffs = np.diff(weights, prepend=0.0)
    annualized_turnover = float(np.sum(np.abs(diffs)) / max(1, n_periods) * 252.0)

    split = max(1, min(n_periods - 1, int(n_periods * train_ratio)))
    train_sharpe = _annualized_sharpe(strategy_returns[:split])
    val_sharpe = _annualized_sharpe(strategy_returns[split:])
    return {
        "train_sharpe": train_sharpe,
        "val_sharpe": val_sharpe,
        "annualized_turnover": annualized_turnover,
        "ensemble_avg_weights": {k: float(np.mean(v)) for k, v in ensemble_weights_trace.items()},
        "ensemble_enabled": bool(eff_cfg.enabled),
        "online_learning_enabled": bool(eff_online_cfg.enabled),
        "online_learning_updates": int(online_learner.n_updates) if online_learner is not None else 0,
    }


def _sample_space_numpy(rng: np.random.Generator) -> SearchSpace:
    ema_fast = int(rng.integers(3, 18))
    ema_slow = int(rng.integers(ema_fast + 3, 64))
    return SearchSpace(
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        rebalance_threshold=float(rng.uniform(0.0005, 0.03)),
        turnover_penalty_lambda=float(rng.uniform(0.0, 0.25)),
        ml_scalar=float(rng.uniform(0.50, 2.00)),
        vol_target=float(rng.uniform(0.06, 0.20)),
    )


def run_bayesian_optimization(
    budget: int = 16,
    seed: int = 42,
    prefer_optuna: bool = True,
    overfit_guard_lambda: float = 0.15,
    ensemble_enabled: bool = False,
    ensemble_lookback: int = 20,
    ensemble_ridge_shrink: float = 0.15,
    ensemble_min_weight: float = 0.05,
    ensemble_max_weight: float = 0.90,
    ensemble_regime_sensitivity: float = 0.40,
    interaction_discovery_enabled: bool = False,
    interaction_top_k: int = 5,
    interaction_generate_crosses: bool = False,
    online_learning_enabled: bool = False,
    online_learning_learning_rate: float = 0.10,
    online_learning_l2: float = 1e-4,
    online_learning_seed: int = 42,
    stress_enabled: bool = False,
    stress_n_paths: int = 200,
    stress_horizon: int = 20,
) -> dict[str, Any]:
    if budget <= 0:
        raise ValueError("budget must be > 0")

    try:
        import optuna  # type: ignore
    except Exception:
        optuna = None

    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    ensemble_cfg = DynamicEnsembleConfig(
        enabled=ensemble_enabled,
        lookback=ensemble_lookback,
        ridge_shrink=ensemble_ridge_shrink,
        min_weight=ensemble_min_weight,
        max_weight=ensemble_max_weight,
        regime_sensitivity=ensemble_regime_sensitivity,
    )
    online_cfg = OnlineLearningConfig(
        enabled=online_learning_enabled,
        learning_rate=online_learning_learning_rate,
        l2=online_learning_l2,
        seed=online_learning_seed,
    )

    if prefer_optuna and optuna is not None:
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def _objective(trial: Any) -> float:
            ema_fast = int(trial.suggest_int("ema_fast", 3, 17))
            ema_slow = int(trial.suggest_int("ema_slow", ema_fast + 3, 64))
            params = SearchSpace(
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                rebalance_threshold=float(trial.suggest_float("rebalance_threshold", 0.0005, 0.03)),
                turnover_penalty_lambda=float(trial.suggest_float("turnover_penalty_lambda", 0.0, 0.25)),
                ml_scalar=float(trial.suggest_float("ml_scalar", 0.50, 2.00)),
                vol_target=float(trial.suggest_float("vol_target", 0.06, 0.20)),
            )
            metrics = _simulate_metrics(params=params, seed=seed, ensemble_cfg=ensemble_cfg, online_cfg=online_cfg)
            objective = compute_objective(
                train_sharpe=metrics["train_sharpe"],
                val_sharpe=metrics["val_sharpe"],
                annualized_turnover=metrics["annualized_turnover"],
                turnover_penalty_lambda=params.turnover_penalty_lambda,
                overfit_guard_lambda=overfit_guard_lambda,
            )
            record = {
                "trial": int(trial.number),
                "params": params.__dict__,
                "metrics": metrics,
                "objective": objective,
            }
            trials.append(record)
            return objective

        study.optimize(_objective, n_trials=budget)
        best = max(trials, key=lambda x: x["objective"])
        backend = "optuna"
    else:
        rng = np.random.default_rng(seed)
        for idx in range(budget):
            params = _sample_space_numpy(rng)
            metrics = _simulate_metrics(params=params, seed=seed, ensemble_cfg=ensemble_cfg, online_cfg=online_cfg)
            objective = compute_objective(
                train_sharpe=metrics["train_sharpe"],
                val_sharpe=metrics["val_sharpe"],
                annualized_turnover=metrics["annualized_turnover"],
                turnover_penalty_lambda=params.turnover_penalty_lambda,
                overfit_guard_lambda=overfit_guard_lambda,
            )
            trials.append(
                {
                    "trial": idx,
                    "params": params.__dict__,
                    "metrics": metrics,
                    "objective": objective,
                }
            )
        best = max(trials, key=lambda x: x["objective"])
        backend = "random-search"

    result = {
        "backend": backend,
        "seed": int(seed),
        "budget": int(budget),
        "overfit_guard_lambda": float(overfit_guard_lambda),
        "ensemble": {
            "enabled": bool(ensemble_cfg.enabled),
            "lookback": int(ensemble_cfg.lookback),
            "ridge_shrink": float(ensemble_cfg.ridge_shrink),
            "min_weight": float(ensemble_cfg.min_weight),
            "max_weight": float(ensemble_cfg.max_weight),
            "regime_sensitivity": float(ensemble_cfg.regime_sensitivity),
        },
        "online_learning": {
            "enabled": bool(online_cfg.enabled),
            "learning_rate": float(online_cfg.learning_rate),
            "l2": float(online_cfg.l2),
            "seed": int(online_cfg.seed),
        },
        "best": best,
        "trials": trials,
    }
    if interaction_discovery_enabled and trials:
        param_names = sorted(trials[0]["params"].keys())
        feature_matrix = np.array([[float(t["params"][name]) for name in param_names] for t in trials], dtype=float)
        target_proxy = np.array([float(t["objective"]) for t in trials], dtype=float)
        ranked = compute_pairwise_interaction_scores(
            feature_matrix=feature_matrix,
            target_proxy=target_proxy,
            feature_names=param_names,
        )
        top = select_top_k_interactions(ranked, int(interaction_top_k))
        interaction_payload: dict[str, Any] = {
            "feature_names": param_names,
            "n_samples": int(feature_matrix.shape[0]),
            "n_features": int(feature_matrix.shape[1]),
            "top_k": int(interaction_top_k),
            "top_interactions": [
                {"feature_i": item.feature_i, "feature_j": item.feature_j, "score": float(item.score)} for item in top
            ],
        }
        if interaction_generate_crosses:
            crosses, cross_names = generate_feature_crosses(
                feature_matrix=feature_matrix,
                selected_pairs=[(item.feature_i, item.feature_j) for item in top],
                feature_names=param_names,
            )
            interaction_payload["cross_features"] = {
                "names": cross_names,
                "shape": [int(crosses.shape[0]), int(crosses.shape[1])],
            }
        result["feature_interactions"] = interaction_payload
    if stress_enabled and best is not None:
        stress_seed = int(seed) + 1000
        best_params = SearchSpace(**best["params"])
        best_sim = _simulate_metrics(params=best_params, seed=seed, ensemble_cfg=ensemble_cfg, online_cfg=online_cfg)
        # Synthetic deterministic proxy series from metrics to keep this lightweight.
        proxy_returns = np.array(
            [
                0.0015 * best_sim["train_sharpe"],
                0.0015 * best_sim["val_sharpe"],
                -0.0008 * best_sim["annualized_turnover"],
            ]
            * max(2, stress_horizon),
            dtype=float,
        )[: max(3, stress_horizon)]
        result["stress_scenarios"] = evaluate_stress_scenarios(
            returns=proxy_returns,
            seed=stress_seed,
            n_paths=stress_n_paths,
            horizon=stress_horizon,
        )
    return result


def run_and_write_optimization(
    out_root: str = "outputs",
    run_id: str | None = None,
    budget: int = 16,
    seed: int = 42,
    prefer_optuna: bool = True,
    overfit_guard_lambda: float = 0.15,
    ensemble_enabled: bool = False,
    ensemble_lookback: int = 20,
    ensemble_ridge_shrink: float = 0.15,
    ensemble_min_weight: float = 0.05,
    ensemble_max_weight: float = 0.90,
    ensemble_regime_sensitivity: float = 0.40,
    interaction_discovery_enabled: bool = False,
    interaction_top_k: int = 5,
    interaction_generate_crosses: bool = False,
    online_learning_enabled: bool = False,
    online_learning_learning_rate: float = 0.10,
    online_learning_l2: float = 1e-4,
    online_learning_seed: int = 42,
    stress_enabled: bool = False,
    stress_n_paths: int = 200,
    stress_horizon: int = 20,
) -> dict[str, Any]:
    if run_id is None:
        run_id = datetime.utcnow().strftime("opt-%Y%m%dT%H%M%SZ")
    if run_id in (".", ".."):
        raise ValueError("Invalid run_id")
    if not all(ch.isalnum() or ch in "._-" for ch in run_id):
        raise ValueError("Invalid run_id; only A-Za-z0-9._- allowed")

    root_abs = os.path.abspath(out_root)
    out_dir_abs = os.path.abspath(os.path.join(root_abs, run_id))
    if not (out_dir_abs == root_abs or out_dir_abs.startswith(root_abs + os.sep)):
        raise ValueError("Invalid run_id; resolved path escapes out_root")

    os.makedirs(out_dir_abs, exist_ok=True)
    result = run_bayesian_optimization(
        budget=budget,
        seed=seed,
        prefer_optuna=prefer_optuna,
        overfit_guard_lambda=overfit_guard_lambda,
        ensemble_enabled=ensemble_enabled,
        ensemble_lookback=ensemble_lookback,
        ensemble_ridge_shrink=ensemble_ridge_shrink,
        ensemble_min_weight=ensemble_min_weight,
        ensemble_max_weight=ensemble_max_weight,
        ensemble_regime_sensitivity=ensemble_regime_sensitivity,
        interaction_discovery_enabled=interaction_discovery_enabled,
        interaction_top_k=interaction_top_k,
        interaction_generate_crosses=interaction_generate_crosses,
        online_learning_enabled=online_learning_enabled,
        online_learning_learning_rate=online_learning_learning_rate,
        online_learning_l2=online_learning_l2,
        online_learning_seed=online_learning_seed,
        stress_enabled=stress_enabled,
        stress_n_paths=stress_n_paths,
        stress_horizon=stress_horizon,
    )

    artifact_path = os.path.join(out_dir_abs, "optimization_result.json")
    with open(artifact_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    return {
        "artifact_path": artifact_path,
        "out_dir": out_dir_abs,
        "result": result,
    }
