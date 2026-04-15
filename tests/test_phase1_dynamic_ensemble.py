import json
import subprocess
from pathlib import Path

from tema.ensemble import DynamicEnsembleConfig, compute_dynamic_ensemble_weights


def test_dynamic_weights_normalized_and_bounded():
    cfg = DynamicEnsembleConfig(enabled=True, lookback=8, ridge_shrink=0.2, min_weight=0.10, max_weight=0.80)
    returns = {
        "tema_base": [0.01, 0.005, -0.002, 0.007, 0.004, 0.006],
        "ml_proxy": [0.003, 0.002, -0.001, 0.003, 0.001, 0.004],
        "risk_proxy": [0.001, 0.0012, 0.0008, 0.0010, 0.0011, 0.0009],
    }
    w = compute_dynamic_ensemble_weights(returns, cfg, regime_score=0.1)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(cfg.min_weight <= v <= cfg.max_weight for v in w.values())


def test_regime_and_performance_adapt_weights():
    cfg = DynamicEnsembleConfig(enabled=True, lookback=10, ridge_shrink=0.1, regime_sensitivity=0.7)
    returns = {
        "tema_base": [0.01, 0.011, 0.008, 0.012, 0.010],
        "ml_proxy": [0.002, 0.001, 0.003, 0.002, 0.0015],
        "risk_proxy": [0.004, 0.0042, 0.0041, 0.0043, 0.0040],
    }
    w_pos = compute_dynamic_ensemble_weights(returns, cfg, regime_score=0.8)
    w_neg = compute_dynamic_ensemble_weights(returns, cfg, regime_score=-0.8)
    assert w_pos["tema_base"] > w_pos["ml_proxy"]
    assert w_neg["risk_proxy"] > w_pos["risk_proxy"]


def test_dynamic_ensemble_entrypoint_smoke(tmp_path):
    root = Path(__file__).resolve().parent.parent
    out_root = tmp_path / "outputs"
    run_id = "smoke-dyn-ens"
    cmd = [
        "python",
        str(root / "scripts" / "run_phase1_dynamic_ensemble.py"),
        "--run-id",
        run_id,
        "--out-root",
        str(out_root),
    ]
    subprocess.check_call(cmd, cwd=str(root))
    ensemble_artifact = out_root / run_id / "ensemble_info.json"
    assert ensemble_artifact.exists()
    payload = json.loads(ensemble_artifact.read_text())
    assert payload["enabled"] is True
    assert abs(sum(payload["weights"].values()) - 1.0) < 1e-9
