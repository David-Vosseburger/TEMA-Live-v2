import json
import subprocess
from pathlib import Path

import numpy as np

from tema.optimization import run_bayesian_optimization
from tema.online_learning import OnlineLogisticLearner


def test_incremental_update_changes_model_state():
    learner = OnlineLogisticLearner(n_features=3, seed=123, learning_rate=0.2)
    x = [0.01, -0.02, 0.005]
    w_before = learner.weights.copy()
    b_before = learner.bias
    learner.partial_fit(x, 1)
    assert learner.n_updates == 1
    assert not np.allclose(learner.weights, w_before)
    assert learner.bias != b_before


def test_deterministic_behavior_with_seed_and_reset():
    stream = [([0.01, 0.02, 0.0], 1), ([-0.01, 0.01, 0.02], 0), ([0.005, -0.003, 0.001], 1)]
    a = OnlineLogisticLearner(n_features=3, seed=7)
    b = OnlineLogisticLearner(n_features=3, seed=7)
    out_a = []
    out_b = []
    for feat, y in stream:
        out_a.append(a.partial_fit(feat, y))
        out_b.append(b.partial_fit(feat, y))
    assert np.allclose(a.weights, b.weights)
    assert a.bias == b.bias
    assert np.allclose(out_a, out_b)
    a.reset(seed=7)
    c = OnlineLogisticLearner(n_features=3, seed=7)
    assert np.allclose(a.weights, c.weights)
    assert a.bias == c.bias
    assert a.n_updates == 0


def test_integration_path_when_enabled():
    res = run_bayesian_optimization(
        budget=4,
        seed=11,
        prefer_optuna=False,
        ensemble_enabled=True,
        online_learning_enabled=True,
        online_learning_seed=11,
    )
    assert res["online_learning"]["enabled"] is True
    assert "online_learning" in res["best"]["metrics"]["ensemble_avg_weights"]
    assert res["best"]["metrics"]["online_learning_updates"] > 0


def test_online_learning_entrypoint_smoke(tmp_path):
    root = Path(__file__).resolve().parent.parent
    out_root = tmp_path / "outputs"
    run_id = "smoke-online"
    cmd = [
        "python",
        str(root / "scripts" / "run_phase4_online_learning.py"),
        "--run-id",
        run_id,
        "--out-root",
        str(out_root),
        "--seed",
        "9",
    ]
    subprocess.check_call(cmd, cwd=str(root))
    artifact = out_root / run_id / "online_learning_artifact.json"
    assert artifact.exists()
    payload = json.loads(artifact.read_text())
    assert payload["n_observations"] == 4
    assert payload["state"]["n_updates"] == 4
