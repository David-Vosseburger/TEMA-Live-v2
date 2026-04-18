from pathlib import Path


def test_regime_report_written_for_template_ml_overlay(tmp_path, monkeypatch):
    monkeypatch.setenv("TEMA_IGNORE_TEMPLATE_DIR", "1")

    from tema.config import BacktestConfig
    from tema.pipeline import run_pipeline

    out_root = str(tmp_path / "outputs")
    cfg = BacktestConfig(
        modular_data_signals_enabled=True,
        portfolio_modular_enabled=True,
        template_default_universe=True,
        ml_enabled=True,
        ml_modular_path_enabled=True,
        ml_template_overlay_enabled=True,
    )

    res = run_pipeline(run_id="regime-report-test", cfg=cfg, out_root=out_root)
    out_dir = Path(res["out_dir"])

    # Should be produced when ML template overlay is enabled and series payload includes HMM bull prob.
    assert (out_dir / "regime_report_base_test.csv").exists()
    assert (out_dir / "regime_report_ml_test.csv").exists()

    # Also recorded in artifacts
    info_path = out_dir / "regime_report_info.json"
    assert info_path.exists()
