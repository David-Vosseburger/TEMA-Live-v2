import json
import subprocess
from pathlib import Path

from tema.optimization import compute_objective, run_bayesian_optimization


def test_objective_penalizes_turnover_and_overfit_gap():
    score = compute_objective(
        train_sharpe=1.20,
        val_sharpe=1.00,
        annualized_turnover=2.0,
        turnover_penalty_lambda=0.10,
        overfit_guard_lambda=0.50,
    )
    # 1.00 - (0.1*2.0) - (0.5*(1.2-1.0)) = 0.70
    assert abs(score - 0.70) < 1e-9


def test_fallback_search_is_deterministic():
    res_a = run_bayesian_optimization(budget=6, seed=123, prefer_optuna=False)
    res_b = run_bayesian_optimization(budget=6, seed=123, prefer_optuna=False)
    assert res_a["backend"] == "random-search"
    assert res_a["best"]["objective"] == res_b["best"]["objective"]
    assert res_a["best"]["params"] == res_b["best"]["params"]


def test_optimization_entrypoint_smoke(tmp_path):
    root = Path(__file__).resolve().parent.parent
    run_id = "smoke-opt"
    out_root = tmp_path / "outputs"
    cmd = [
        "python",
        str(root / "scripts" / "run_phase1_bayes_opt.py"),
        "--run-id",
        run_id,
        "--out-root",
        str(out_root),
        "--budget",
        "3",
        "--seed",
        "7",
        "--no-optuna",
    ]
    subprocess.check_call(cmd, cwd=str(root))
    artifact = out_root / run_id / "optimization_result.json"
    assert artifact.exists()
    payload = json.loads(artifact.read_text())
    assert payload["budget"] == 3
    assert payload["seed"] == 7
