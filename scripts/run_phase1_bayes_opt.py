#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tema.optimization import run_and_write_optimization


def main(argv=None):
    parser = argparse.ArgumentParser("run_phase1_bayes_opt")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-root", default="outputs")
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-optuna", action="store_true")
    parser.add_argument("--overfit-guard-lambda", type=float, default=0.15)
    parser.add_argument("--ensemble-enabled", action="store_true")
    parser.add_argument("--ensemble-lookback", type=int, default=20)
    parser.add_argument("--ensemble-ridge-shrink", type=float, default=0.15)
    parser.add_argument("--ensemble-min-weight", type=float, default=0.05)
    parser.add_argument("--ensemble-max-weight", type=float, default=0.90)
    parser.add_argument("--ensemble-regime-sensitivity", type=float, default=0.40)
    parser.add_argument("--online-learning-enabled", action="store_true")
    parser.add_argument("--online-learning-learning-rate", type=float, default=0.10)
    parser.add_argument("--online-learning-l2", type=float, default=1e-4)
    parser.add_argument("--online-learning-seed", type=int, default=42)
    parser.add_argument("--stress-enabled", action="store_true")
    parser.add_argument("--stress-n-paths", type=int, default=200)
    parser.add_argument("--stress-horizon", type=int, default=20)
    args = parser.parse_args(argv)

    res = run_and_write_optimization(
        out_root=args.out_root,
        run_id=args.run_id,
        budget=args.budget,
        seed=args.seed,
        prefer_optuna=not args.no_optuna,
        overfit_guard_lambda=args.overfit_guard_lambda,
        ensemble_enabled=args.ensemble_enabled,
        ensemble_lookback=args.ensemble_lookback,
        ensemble_ridge_shrink=args.ensemble_ridge_shrink,
        ensemble_min_weight=args.ensemble_min_weight,
        ensemble_max_weight=args.ensemble_max_weight,
        ensemble_regime_sensitivity=args.ensemble_regime_sensitivity,
        online_learning_enabled=args.online_learning_enabled,
        online_learning_learning_rate=args.online_learning_learning_rate,
        online_learning_l2=args.online_learning_l2,
        online_learning_seed=args.online_learning_seed,
        stress_enabled=args.stress_enabled,
        stress_n_paths=args.stress_n_paths,
        stress_horizon=args.stress_horizon,
    )
    print(
        {
            "artifact_path": res["artifact_path"],
            "backend": res["result"]["backend"],
            "best_objective": res["result"]["best"]["objective"],
        }
    )
    return res


if __name__ == "__main__":
    main()
