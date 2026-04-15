import json
from pathlib import Path

import numpy as np

from tema.backtest import build_weight_schedule_from_signals, run_return_equity_simulation
from tema.config import BacktestConfig
from tema.pipeline import run_pipeline


def test_run_return_equity_simulation_computes_core_metrics():
    asset_returns = np.array([[0.01], [-0.02], [0.03]], dtype=float)
    target_weights = np.array([[1.0], [1.0], [1.0]], dtype=float)
    res = run_return_equity_simulation(asset_returns, target_weights, freq="D")

    expected_sharpe = float(np.mean(asset_returns[:, 0]) / np.std(asset_returns[:, 0], ddof=0) * np.sqrt(252.0))
    expected_vol = float(np.std(asset_returns[:, 0], ddof=0) * np.sqrt(252.0))
    expected_equity = np.cumprod(1.0 + asset_returns[:, 0])
    expected_mdd = float(np.min(expected_equity / np.maximum.accumulate(expected_equity) - 1.0))

    assert len(res.periodic_returns) == 3
    assert abs(res.metrics["sharpe"] - expected_sharpe) < 1e-10
    assert abs(res.metrics["annual_vol"] - expected_vol) < 1e-10
    assert abs(res.metrics["max_drawdown"] - expected_mdd) < 1e-10
    assert "annualized_turnover" in res.metrics


def test_build_weight_schedule_from_signals_has_safe_fallback():
    import pandas as pd

    signal_df = pd.DataFrame([[1.0, 0.0], [0.0, 0.0], [-1.0, 2.0]], columns=["a", "b"])
    sched = build_weight_schedule_from_signals(signal_df, fallback_weights=[0.6, 0.4])
    assert sched.shape == (3, 2)
    assert abs(float(np.sum(sched[0])) - 1.0) < 1e-12
    assert abs(float(np.sum(sched[1])) - 1.0) < 1e-12
    assert abs(sched[1, 0] - 0.6) < 1e-12 and abs(sched[1, 1] - 0.4) < 1e-12


def test_pipeline_includes_performance_artifact_with_fallback(tmp_path):
    out_root = tmp_path / "outputs"
    cfg = BacktestConfig(
        modular_data_signals_enabled=True,
        data_path=str(tmp_path / "missing-data-dir"),
    )
    res = run_pipeline(run_id="perf-fallback-test", cfg=cfg, out_root=str(out_root))
    perf_path = Path(res["out_dir"]) / "performance.json"
    manifest = json.loads((Path(res["out_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    performance = json.loads(perf_path.read_text(encoding="utf-8"))

    assert perf_path.exists()
    assert "performance" in manifest["artifacts"]
    assert performance["fallback_used"] is True
    for key in ("sharpe", "annual_return", "annual_vol", "max_drawdown", "annualized_turnover", "turnover_proxy"):
        assert key in performance
