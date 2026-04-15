#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tema.stress import evaluate_stress_scenarios


def main(argv=None):
    parser = argparse.ArgumentParser("run_stress_scenarios")
    parser.add_argument("--run-id", default="stress-smoke")
    parser.add_argument("--out-root", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-paths", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--n-returns", type=int, default=252)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    base_returns = rng.normal(loc=0.0005, scale=0.012, size=args.n_returns)
    result = evaluate_stress_scenarios(
        returns=base_returns,
        seed=args.seed,
        n_paths=args.n_paths,
        horizon=args.horizon,
    )

    out_dir = Path(args.out_root) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / "stress_scenarios.json"
    with open(artifact_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    payload = {"artifact_path": str(artifact_path), "run_id": args.run_id, "seed": int(args.seed)}
    print(payload)
    return payload


if __name__ == "__main__":
    main()
