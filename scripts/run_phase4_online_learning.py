#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tema.online_learning import OnlineLogisticLearner


def main(argv=None):
    parser = argparse.ArgumentParser("run_phase4_online_learning")
    parser.add_argument("--run-id", default="phase4-online-learning-smoke")
    parser.add_argument("--out-root", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.10)
    parser.add_argument("--l2", type=float, default=1e-4)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_root) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    learner = OnlineLogisticLearner(
        n_features=3,
        learning_rate=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
    )
    stream = [
        ([0.010, 0.009, 0.004], 1),
        ([-0.006, -0.004, 0.003], 0),
        ([0.012, 0.011, 0.002], 1),
        ([-0.008, -0.005, 0.004], 0),
    ]
    scores = []
    for feat, y in stream:
        before = learner.predict_score(feat)
        learner.partial_fit(feat, y)
        after = learner.predict_score(feat)
        scores.append({"feature": feat, "label": y, "score_before": before, "score_after": after})

    artifact = {
        "run_id": args.run_id,
        "seed": int(args.seed),
        "n_observations": len(stream),
        "scores": scores,
        "state": learner.snapshot(),
    }
    artifact_path = out_dir / "online_learning_artifact.json"
    with open(artifact_path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2)
    print({"artifact_path": str(artifact_path), "n_updates": learner.n_updates})
    return {"artifact_path": str(artifact_path), "out_dir": str(out_dir)}


if __name__ == "__main__":
    main()
