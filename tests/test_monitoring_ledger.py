import sqlite3
from pathlib import Path


def test_monitoring_ledger_writes_row(tmp_path, monkeypatch):
    monkeypatch.setenv("TEMA_IGNORE_TEMPLATE_DIR", "1")

    from tema.config import BacktestConfig
    from tema.pipeline import run_pipeline

    ledger_path = tmp_path / "tema_ledger.sqlite"
    out_root = str(tmp_path / "outputs")

    cfg = BacktestConfig(
        modular_data_signals_enabled=True,
        portfolio_modular_enabled=True,
        template_default_universe=True,
        ml_modular_path_enabled=False,
        monitoring_ledger_enabled=True,
        monitoring_ledger_path=str(ledger_path),
    )

    res = run_pipeline(run_id="ledger-test", cfg=cfg, out_root=out_root)
    out_dir = Path(res["out_dir"])

    assert ledger_path.exists(), "ledger sqlite file not created"

    conn = sqlite3.connect(str(ledger_path))
    try:
        cur = conn.cursor()
        cur.execute("SELECT out_dir, run_id FROM runs WHERE out_dir = ?", (str(out_dir),))
        row = cur.fetchone()
        assert row is not None
        assert row[1] == "ledger-test"
    finally:
        conn.close()
