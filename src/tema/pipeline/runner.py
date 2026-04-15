from typing import List, Sequence, Optional
from ..config import BacktestConfig
from ..turnover import apply_rebalance_gating
from ..ensemble import DynamicEnsembleConfig, combine_stream_signals, compute_dynamic_ensemble_weights
from ..online_learning import OnlineLogisticLearner
from ..stress import evaluate_stress_scenarios
import json
import os
from datetime import datetime


def _portfolio_stage() -> tuple[Sequence[float], Sequence[float], Sequence[float]]:
    """Simplified BL/portfolio stage producing current, candidate, and expected alphas.
    In real code this would call into portfolio/optimization modules. Here we keep
    deterministic, small arrays so orchestration can be tested.
    """
    current = [0.30, 0.40, 0.30]
    candidate = [0.25, 0.45, 0.30]
    expected_alphas = [0.01, 0.02, 0.005]
    return current, candidate, expected_alphas


def _ml_filter_and_scalar(cfg: BacktestConfig, expected_alphas: Sequence[float]) -> dict:
    """Minimal ML stage: optionally adjusts expected_alphas or returns a scalar.
    We return a small dict describing ML decisions to include in the manifest.
    """
    ml_info = {
        "ml_enabled": bool(cfg.ml_enabled),
        "scalar": [1.0 for _ in expected_alphas],
        "notes": "simple-pass-through scalar for Wave 2 smoke runner",
    }
    return ml_info


def _scaling_stage(weights: Sequence[float], ml_info: dict, cfg: BacktestConfig) -> List[float]:
    """Apply ml scalar and a naive vol-target style normalization.
    This keeps deterministic behavior while demonstrating the interface.
    """
    scalar = ml_info.get("scalar", [1.0] * len(weights))
    # validate lengths to avoid silently dropping assets
    if len(scalar) != len(weights):
        raise ValueError(f"Scalar length {len(scalar)} does not match weights length {len(weights)}")
    scaled = [w * s for w, s in zip(weights, scalar)]
    # ensure no negative and normalize to sum 1 unless all zeros
    total = sum(abs(x) for x in scaled)
    if total == 0:
        return list(scaled)
    normalized = [x / total for x in scaled]
    return normalized


def _ensemble_stage(
    cfg: BacktestConfig,
    current: Sequence[float],
    candidate_weights: Sequence[float],
    expected_alphas: Sequence[float],
    ml_info: dict,
) -> tuple[list[float], dict]:
    if not cfg.ensemble_enabled:
        return list(expected_alphas), {"enabled": False, "weights": None}

    ml_scalar = ml_info.get("scalar", [1.0] * len(expected_alphas))
    risk_proxy = [max(0.0, 1.0 - abs(nw - cw)) * 0.01 for cw, nw in zip(current, candidate_weights)]
    stream_signals = {
        "tema_base": list(expected_alphas),
        "ml_proxy": [a * s for a, s in zip(expected_alphas, ml_scalar)],
        "risk_proxy": risk_proxy,
    }
    if cfg.online_learning_enabled:
        learner = OnlineLogisticLearner(
            n_features=3,
            learning_rate=cfg.online_learning_learning_rate,
            l2=cfg.online_learning_l2,
            seed=cfg.online_learning_seed,
        )
        online_signal = []
        for alpha, ml, risk in zip(expected_alphas, stream_signals["ml_proxy"], risk_proxy):
            feat = [alpha, ml, risk]
            score = learner.predict_score(feat)
            online_signal.append(2.0 * score - 1.0)
            learner.partial_fit(feat, 1.0 if alpha > 0.0 else 0.0)
        stream_signals["online_learning"] = online_signal
    stream_returns = {
        "tema_base": [0.7 * x for x in expected_alphas] + [1.0 * x for x in expected_alphas] + [1.2 * x for x in expected_alphas],
        "ml_proxy": [0.6 * x for x in stream_signals["ml_proxy"]]
        + [1.0 * x for x in stream_signals["ml_proxy"]]
        + [1.1 * x for x in stream_signals["ml_proxy"]],
        "risk_proxy": [0.9 * x for x in risk_proxy] + [1.0 * x for x in risk_proxy] + [1.05 * x for x in risk_proxy],
    }
    if cfg.online_learning_enabled:
        stream_returns["online_learning"] = (
            [0.8 * x for x in stream_signals["online_learning"]]
            + [1.0 * x for x in stream_signals["online_learning"]]
            + [1.15 * x for x in stream_signals["online_learning"]]
        )
    stream_names = list(stream_signals.keys())

    ensemble_cfg = DynamicEnsembleConfig(
        enabled=True,
        lookback=cfg.ensemble_lookback,
        ridge_shrink=cfg.ensemble_ridge_shrink,
        min_weight=cfg.ensemble_min_weight,
        max_weight=cfg.ensemble_max_weight,
        regime_sensitivity=cfg.ensemble_regime_sensitivity,
    )
    regime_score = float(sum(expected_alphas) - sum(abs(nw - cw) for cw, nw in zip(current, candidate_weights)))
    weights = compute_dynamic_ensemble_weights(
        stream_returns=stream_returns,
        cfg=ensemble_cfg,
        regime_score=regime_score,
        stream_names=stream_names,
    )
    combined = combine_stream_signals(stream_signals, weights, stream_names=stream_names)
    info = {
        "enabled": True,
        "online_learning_enabled": bool(cfg.online_learning_enabled),
        "regime_score": regime_score,
        "weights": weights,
        "stream_signals": stream_signals,
        "combined_expected_alphas": combined,
    }
    return combined, info


def run_pipeline(run_id: Optional[str] = None, cfg: Optional[BacktestConfig] = None, out_root: str = "outputs") -> dict:
    """Execute Wave 2 simplified pipeline and write artifacts under outputs/{run_id}/.

    Returns a dict summary which is also written to manifest.json.
    """
    if run_id is None:
        run_id = datetime.utcnow().strftime("run-%Y%m%dT%H%M%SZ")
    if cfg is None:
        cfg = BacktestConfig()

    # sanitize run_id to avoid path traversal
    import re as _re
    # basic token check
    if not _re.match(r'^[A-Za-z0-9._-]+$', run_id):
        raise ValueError("Invalid run_id; only A-Za-z0-9._- allowed")
    # reject single or double-dot ids which can escape directories
    if run_id in ('.', '..'):
        raise ValueError("Invalid run_id; '.' and '..' are not allowed")
    # ensure resolved path remains under out_root to prevent path traversal
    out_root_abs = os.path.abspath(out_root)
    candidate = os.path.abspath(os.path.join(out_root_abs, run_id))
    if not (candidate == out_root_abs or candidate.startswith(out_root_abs + os.sep)):
        raise ValueError("Invalid run_id; resolved path escapes out_root")

    out_dir = os.path.join(out_root, run_id)
    os.makedirs(out_dir, exist_ok=True)

    # Stage 1: Portfolio (BL-like)
    current, candidate, expected_alphas = _portfolio_stage()

    # Stage 2: ML filter / scaler
    ml_info = _ml_filter_and_scalar(cfg, expected_alphas)

    # Stage 3: Optional dynamic ensemble (feature-flagged)
    ensemble_alphas, ensemble_info = _ensemble_stage(cfg, current, candidate, expected_alphas, ml_info)

    # Stage 4: Turnover / cost gate
    gated = apply_rebalance_gating(current, candidate, ensemble_alphas, cfg)

    # Stage 5: Scaling stage
    final_weights = _scaling_stage(gated, ml_info, cfg)

    # Stage 6: Reporting artifacts
    artifacts = {
        "current_weights": current,
        "candidate_weights": candidate,
        "expected_alphas": expected_alphas,
        "ensemble_info": ensemble_info,
        "effective_expected_alphas": ensemble_alphas,
        "gated_weights": gated,
        "final_weights": final_weights,
        "ml_info": ml_info,
    }
    if cfg.stress_enabled:
        artifacts["stress_scenarios"] = evaluate_stress_scenarios(
            returns=list(ensemble_alphas),
            seed=cfg.stress_seed,
            n_paths=cfg.stress_n_paths,
            horizon=cfg.stress_horizon,
        )

    manifest = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "artifacts": list(artifacts.keys()),
    }

    # write artifacts
    for name, value in artifacts.items():
        path = os.path.join(out_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2)

    # write manifest
    mf_path = os.path.join(out_dir, "manifest.json")
    with open(mf_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {"manifest_path": mf_path, "out_dir": out_dir, "manifest": manifest}


if __name__ == "__main__":
    # quick CLI for ad-hoc local runs
    import argparse
    parser = argparse.ArgumentParser("tema-pipeline-runner")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    print(run_pipeline(run_id=args.run_id))
