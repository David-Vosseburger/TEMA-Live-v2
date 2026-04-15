#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tema.optimization import run_and_write_optimization


def main(argv=None):
    parser = argparse.ArgumentParser("run_phase4_feature_interactions")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-root", default="outputs")
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-optuna", action="store_true")
    parser.add_argument("--include-crosses", action="store_true")
    args = parser.parse_args(argv)

    res = run_and_write_optimization(
        out_root=args.out_root,
        run_id=args.run_id,
        budget=args.budget,
        seed=args.seed,
        prefer_optuna=not args.no_optuna,
        interaction_discovery_enabled=True,
        interaction_top_k=args.top_k,
        interaction_generate_crosses=args.include_crosses,
    )
    interaction_payload = res["result"].get("feature_interactions", {})
    artifact_path = Path(res["out_dir"]) / "interaction_discovery.json"
    with open(artifact_path, "w", encoding="utf-8") as fh:
        json.dump(interaction_payload, fh, indent=2)

    print({"artifact_path": str(artifact_path), "top_interactions": len(interaction_payload.get("top_interactions", []))})
    return {"artifact_path": str(artifact_path), "out_dir": res["out_dir"]}


if __name__ == "__main__":
    main()
