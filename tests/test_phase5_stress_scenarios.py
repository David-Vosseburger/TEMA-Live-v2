import json
import subprocess
from pathlib import Path

import numpy as np

from tema.config import BacktestConfig
from tema.optimization import run_bayesian_optimization
from tema.pipeline import run_pipeline
from tema.stress import compute_scenario_metrics, evaluate_stress_scenarios, sample_scenario_paths


def test_scenario_metrics_expected_shape_and_keys():
    returns = np.array([0.01, -0.02, 0.015, -0.005], dtype=float)
    result = evaluate_stress_scenarios(returns=returns, seed=11, n_paths=20, horizon=4)
    assert set(result["historical"].keys()) == {"equity_crash", "vol_spike", "spread_widening_proxy"}
    for metrics in result["historical"].values():
        assert set(metrics.keys()) == {"return", "vol", "max_drawdown_proxy"}


def test_bootstrap_sampling_is_deterministic_with_seed():
    returns = np.array([0.01, -0.01, 0.005, -0.002], dtype=float)
    a = sample_scenario_paths(returns=returns, n_paths=5, horizon=6, seed=7, method="bootstrap")
    b = sample_scenario_paths(returns=returns, n_paths=5, horizon=6, seed=7, method="bootstrap")
    assert np.allclose(a, b)


def test_metric_summary_on_known_path():
    # (1.10 * 0.90 * 1.05) - 1 = 0.0395
    metrics = compute_scenario_metrics([0.10, -0.10, 0.05])
    assert abs(metrics["return"] - 0.0395) < 1e-12
    assert metrics["max_drawdown_proxy"] <= 0.0


def test_pipeline_stress_flag_writes_artifact(tmp_path):
    out_root = str(tmp_path / "outputs")
    cfg = BacktestConfig(stress_enabled=True, stress_seed=5, stress_n_paths=10, stress_horizon=6)
    res = run_pipeline(run_id="stress-pipeline", cfg=cfg, out_root=out_root)
    stress_file = Path(res["out_dir"]) / "stress_scenarios.json"
    assert stress_file.exists()
    payload = json.loads(stress_file.read_text())
    assert payload["seed"] == 5


def test_optimization_stress_default_off_and_optional_on():
    baseline = run_bayesian_optimization(budget=3, seed=13, prefer_optuna=False)
    assert "stress_scenarios" not in baseline
    stressed = run_bayesian_optimization(
        budget=3,
        seed=13,
        prefer_optuna=False,
        stress_enabled=True,
        stress_n_paths=10,
        stress_horizon=5,
    )
    assert "stress_scenarios" in stressed
    assert stressed["stress_scenarios"]["horizon"] == 5


def test_stress_script_smoke(tmp_path):
    root = Path(__file__).resolve().parent.parent
    run_id = "stress-script-smoke"
    out_root = tmp_path / "outputs"
    cmd = [
        "python",
        str(root / "scripts" / "run_stress_scenarios.py"),
        "--run-id",
        run_id,
        "--out-root",
        str(out_root),
        "--seed",
        "17",
        "--n-paths",
        "12",
        "--horizon",
        "8",
        "--n-returns",
        "64",
    ]
    subprocess.check_call(cmd, cwd=str(root))
    artifact = out_root / run_id / "stress_scenarios.json"
    assert artifact.exists()
    payload = json.loads(artifact.read_text())
    assert payload["seed"] == 17
