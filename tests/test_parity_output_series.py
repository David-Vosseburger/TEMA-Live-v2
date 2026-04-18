import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure repo root is on sys.path so we can import run_pipeline.py
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_pipeline


def _load_csv(path: Path):
    return pd.read_csv(path)


def _compare_series_df(produced: pd.DataFrame, fixture: pd.DataFrame, value_col: str):
    # Ensure same length and datetime alignment
    assert list(produced.columns)[0] == "datetime"
    assert list(fixture.columns)[0] == "datetime"
    assert produced["datetime"].tolist() == fixture["datetime"].tolist()
    # Numeric comparison
    a = produced[value_col].to_numpy(dtype=float)
    b = fixture[value_col].to_numpy(dtype=float)
    assert a.shape == b.shape
    # Tight tolerance
    assert np.allclose(a, b, atol=1e-12, rtol=0.0)


def test_parity_baseline_and_ml_and_meta(tmp_path, monkeypatch):
    # Use vendored fixtures by ignoring Template/ dir
    monkeypatch.setenv("TEMA_IGNORE_TEMPLATE_DIR", "1")

    out_root = str(tmp_path / "outputs")

    # Baseline (no ML overlays)
    res_base = run_pipeline.run_modular(
        run_id="parity-base",
        out_root=out_root,
        modular_data_signals_enabled=True,
        modular_portfolio_enabled=True,
        template_default_universe=True,
        ml_modular_path_enabled=False,
        ml_template_overlay=False,
    )
    out_dir = Path(res_base["out_dir"])
    produced_base = out_dir / "portfolio_test_returns.csv"
    assert produced_base.exists(), f"baseline CSV not produced: {produced_base}"
    perf_base = json.loads((out_dir / "performance.json").read_text(encoding="utf-8"))
    assert perf_base["source"].get("overlay_derived") is not True

    fixture_dir = Path(run_pipeline.ROOT) / "src" / "tema" / "benchmarks" / "template_default_universe"
    fixture_base = fixture_dir / "portfolio_test_returns.csv"
    assert fixture_base.exists(), f"fixture missing: {fixture_base}"

    df_prod_base = _load_csv(produced_base)
    df_fix_base = _load_csv(fixture_base)
    _compare_series_df(df_prod_base, df_fix_base, "portfolio_return")

    # ML overlay
    res_ml = run_pipeline.run_modular(
        run_id="parity-ml",
        out_root=out_root,
        modular_data_signals_enabled=True,
        modular_portfolio_enabled=True,
        template_default_universe=True,
        ml_modular_path_enabled=True,
        ml_template_overlay=True,
    )
    out_dir_ml = Path(res_ml["out_dir"])
    produced_ml = out_dir_ml / "portfolio_test_returns_ml.csv"
    assert produced_ml.exists(), f"ml CSV not produced: {produced_ml}"
    perf_ml = json.loads((out_dir_ml / "performance.json").read_text(encoding="utf-8"))
    overlay_ml = json.loads((out_dir_ml / "template_ml_overlay.json").read_text(encoding="utf-8"))
    assert perf_ml["overlay_performance_promoted"] is True
    assert perf_ml["overlay_performance_source"] == "template_ml_overlay.ml_test_metrics"
    assert perf_ml["source"]["overlay_source"] == "template_ml_overlay.ml_test_metrics"
    assert np.isclose(perf_ml["sharpe"], overlay_ml["ml_test_metrics"]["sharpe"])
    assert np.isclose(perf_ml["annual_return"], overlay_ml["ml_test_metrics"]["annual_return"])
    assert np.isclose(perf_ml["max_drawdown"], overlay_ml["ml_test_metrics"]["max_drawdown"])

    fixture_ml = fixture_dir / "portfolio_test_returns_ml.csv"
    assert fixture_ml.exists(), f"fixture missing: {fixture_ml}"

    df_prod_ml = _load_csv(produced_ml)
    df_fix_ml = _load_csv(fixture_ml)
    _compare_series_df(df_prod_ml, df_fix_ml, "portfolio_return_ml")

    # ML_META overlay (requires both overlays enabled)
    res_meta = run_pipeline.run_modular(
        run_id="parity-ml-meta",
        out_root=out_root,
        modular_data_signals_enabled=True,
        modular_portfolio_enabled=True,
        template_default_universe=True,
        ml_modular_path_enabled=True,
        ml_template_overlay=True,
        ml_meta_overlay=True,
    )
    out_dir_meta = Path(res_meta["out_dir"])
    produced_meta = out_dir_meta / "portfolio_test_returns_ml_meta.csv"
    assert produced_meta.exists(), f"ml_meta CSV not produced: {produced_meta}"
    perf_meta = json.loads((out_dir_meta / "performance.json").read_text(encoding="utf-8"))
    overlay_meta = json.loads((out_dir_meta / "template_ml_meta_overlay.json").read_text(encoding="utf-8"))
    assert perf_meta["overlay_performance_promoted"] is True
    assert perf_meta["overlay_performance_source"] == "template_ml_meta_overlay.test_metrics"
    assert perf_meta["source"]["overlay_source"] == "template_ml_meta_overlay.test_metrics"
    assert np.isclose(perf_meta["sharpe"], overlay_meta["test_metrics"]["sharpe"])
    assert np.isclose(perf_meta["annual_return"], overlay_meta["test_metrics"]["annual_return"])
    assert np.isclose(perf_meta["max_drawdown"], overlay_meta["test_metrics"]["max_drawdown"])

    fixture_meta = fixture_dir / "portfolio_test_returns_ml_meta.csv"
    assert fixture_meta.exists(), f"fixture missing: {fixture_meta}"

    df_prod_meta = _load_csv(produced_meta)
    df_fix_meta = _load_csv(fixture_meta)
    _compare_series_df(df_prod_meta, df_fix_meta, "portfolio_return_ml_meta")


def test_overlay_failure_keeps_baseline_performance(tmp_path, monkeypatch):
    monkeypatch.setenv("TEMA_IGNORE_TEMPLATE_DIR", "1")

    import tema.ml.template_overlay as template_overlay_mod

    def _raise_overlay(**kwargs):
        raise RuntimeError("overlay failed")

    monkeypatch.setattr(template_overlay_mod, "compute_template_ml_overlay", _raise_overlay)

    out_root = str(tmp_path / "outputs")
    res = run_pipeline.run_modular(
        run_id="parity-ml-fail",
        out_root=out_root,
        modular_data_signals_enabled=True,
        modular_portfolio_enabled=True,
        template_default_universe=True,
        ml_modular_path_enabled=True,
        ml_template_overlay=True,
    )

    out_dir = Path(res["out_dir"])
    performance = json.loads((out_dir / "performance.json").read_text(encoding="utf-8"))
    returns_csv_info = json.loads((out_dir / "returns_csv_info.json").read_text(encoding="utf-8"))
    overlay_info = json.loads((out_dir / "template_ml_overlay.json").read_text(encoding="utf-8"))

    assert overlay_info["enabled"] is False
    assert "overlay failed" in overlay_info["error"]
    assert performance.get("overlay_performance_promoted") is not True
    assert returns_csv_info["performance_overlay_promotion_error"] == "overlay_metrics_unavailable"
