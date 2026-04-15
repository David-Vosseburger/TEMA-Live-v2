#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tema.config import BacktestConfig
from tema.pipeline import run_pipeline


def main(argv=None):
    parser = argparse.ArgumentParser("run_phase1_dynamic_ensemble")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-root", default="outputs")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--ridge-shrink", type=float, default=0.15)
    parser.add_argument("--min-weight", type=float, default=0.05)
    parser.add_argument("--max-weight", type=float, default=0.90)
    parser.add_argument("--regime-sensitivity", type=float, default=0.40)
    args = parser.parse_args(argv)

    cfg = BacktestConfig(
        ensemble_enabled=True,
        ensemble_lookback=args.lookback,
        ensemble_ridge_shrink=args.ridge_shrink,
        ensemble_min_weight=args.min_weight,
        ensemble_max_weight=args.max_weight,
        ensemble_regime_sensitivity=args.regime_sensitivity,
    )
    res = run_pipeline(run_id=args.run_id, cfg=cfg, out_root=args.out_root)
    print({"manifest_path": res["manifest_path"], "out_dir": res["out_dir"], "ensemble_enabled": True})
    return res


if __name__ == "__main__":
    main()
