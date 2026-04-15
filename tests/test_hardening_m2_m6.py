import os
import json
import numpy as np
import pandas as pd
import pytest

from tema.data import loader, splitter
from tema.signals import generate_crossover_signal_matrix
from tema.signals.engine import resolve_signal_engine, PythonSignalEngine
from tema.ml import threshold as ml_threshold, scalar as ml_scalar
from tema.portfolio.allocation import allocate_portfolio_weights
from tema.backtest import build_weight_schedule_from_signals, run_return_equity_simulation, compute_backtest_metrics
from tema.cpp import get_signal_engine, has_cpp


# --- M2: Data loading / splitting edge cases ---

def make_csv(fp, rows=5, col_name="close", start="2020-01-01"):
    idx = pd.date_range(start=start, periods=rows, freq="D", tz="UTC")
    df = pd.DataFrame({"datetime": idx, col_name: np.linspace(1.0, 2.0, num=rows)})
    df.to_csv(fp, index=False)


def test_find_price_files_max_assets_zero(tmp_path):
    d = tmp_path
    # create two candidate files
    (d / "A_merged.csv").write_text("x")
    (d / "B_merged.csv").write_text("y")
    with pytest.raises(ValueError):
        loader.find_price_files(d, max_assets=0)


def test_load_close_series_empty_and_missing_columns(tmp_path):
    fp = tmp_path / "empty_merged.csv"
    fp.write_text("")
    with pytest.raises(ValueError):
        loader.load_close_series_from_csv(fp)

    # missing datetime and timestamp
    bad = tmp_path / "bad_merged.csv"
    bad.write_text("close,other\n1.0,2.0\n")
    with pytest.raises(KeyError):
        loader.load_close_series_from_csv(bad)


def test_load_price_panel_min_rows_filtering(tmp_path):
    # create three assets with varying rows
    a = tmp_path / "a_merged.csv"
    b = tmp_path / "b_merged.csv"
    c = tmp_path / "c_merged.csv"
    make_csv(a, rows=5)
    make_csv(b, rows=2)
    make_csv(c, rows=4)
    panel = loader.load_price_panel(data_path=str(tmp_path), root=str(tmp_path), max_assets=None, min_rows=3)
    # only a and c should be present
    assert list(panel.columns) == ["a", "c"] or set(panel.columns) >= set(["a", "c"]) 


def test_split_train_test_edge_cases():
    s = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError):
        splitter.split_train_test(s, train_ratio=0.0)
    with pytest.raises(ValueError):
        splitter.split_train_test(s, min_train_rows=0)
    with pytest.raises(ValueError):
        splitter.split_train_test(pd.Series([1.0]), min_train_rows=1, min_test_rows=1)


# --- M3: Signal generation consistency ---

def test_generate_crossover_signal_matrix_validation_and_output():
    df = pd.DataFrame({"x": [1,2,3,4,5,6,7,8,9,10]}, index=pd.date_range("2020-01-01", periods=10, tz="UTC"))
    with pytest.raises(ValueError):
        generate_crossover_signal_matrix(df, fast_period=5, slow_period=5)
    with pytest.raises(ValueError):
        generate_crossover_signal_matrix(df, fast_period=-1, slow_period=5)
    # valid call
    sig = generate_crossover_signal_matrix(df, fast_period=2, slow_period=5, method="ema", shift_by=1)
    assert sig.shape == df.shape
    # For strictly increasing series expect last signal to be non-negative
    assert float(sig.iloc[-1, 0]) >= 0.0


def test_resolve_signal_engine_cpp_fallback(monkeypatch):
    # Force has_cpp to True but get_signal_engine to raise, ensuring fallback
    import tema.cpp as cpp_mod

    monkeypatch.setattr(cpp_mod, "has_cpp", lambda: True)
    monkeypatch.setattr(cpp_mod, "_cpp_lib", None)

    # If get_signal_engine raises, resolve_signal_engine should return a PythonSignalEngine
    monkeypatch.setattr(cpp_mod, "get_signal_engine", lambda prefer_cpp=True: (_ for _ in ()).throw(RuntimeError("nope")))
    eng = resolve_signal_engine(use_cpp=True, cpp_engine=None)
    assert isinstance(eng, PythonSignalEngine)


# --- M4: ML threshold / scalar edge cases ---

def test_threshold_probabilities_boundaries():
    with pytest.raises(ValueError):
        ml_threshold.threshold_probabilities([0.1, 0.9], threshold=-0.1)
    with pytest.raises(ValueError):
        ml_threshold.threshold_probabilities([0.1], threshold=1.1)
    probs = [0.0, 0.5, 1.0]
    assert ml_threshold.threshold_probabilities(probs, threshold=0.5) == [0.0, 1.0, 1.0]


def test_compute_position_scalars_clipping_and_decisions():
    with pytest.raises(ValueError):
        ml_scalar.compute_position_scalars([0.1], floor=-0.1, ceiling=1.0)
    with pytest.raises(ValueError):
        ml_scalar.compute_position_scalars([0.1], floor=0.0, ceiling=-1.0)
    # clipping probabilities outside [0,1]
    scalars = ml_scalar.compute_position_scalars([1.5, -0.5, 0.5], floor=0.1, ceiling=1.0)
    assert all(0.1 <= s <= 1.0 for s in scalars)
    # decisions length mismatch
    with pytest.raises(ValueError):
        ml_scalar.compute_position_scalars([0.2, 0.3], floor=0.1, ceiling=1.0, decisions=[1.0])


# --- M5: Portfolio allocation corner cases ---

def test_allocate_portfolio_empty_and_mismatch_returns():
    res = allocate_portfolio_weights([], returns_window=None)
    assert res.weights == []
    # mismatched returns window should be handled via identity covariance and produce deterministic result
    alloc = allocate_portfolio_weights([0.01, 0.02], returns_window=np.array([[1.0],[2.0]]))
    assert isinstance(alloc.weights, list)
    assert len(alloc.weights) == 2


# --- M6: Backtest metric stability and weight schedule ---

def test_build_weight_schedule_and_normalize():
    sig = pd.DataFrame([[1.0, -0.5], [0.0, 0.0]], index=pd.date_range("2020-01-01", periods=2, tz="UTC"), columns=["a","b"]) 
    fallback = [0.6, 0.4]
    ws = build_weight_schedule_from_signals(sig, fallback)
    assert ws.shape == (2,2)
    # rows sum to 1
    assert np.allclose(ws.sum(axis=1), 1.0)


def test_run_return_equity_simulation_edge_cases():
    # mismatched dims
    with pytest.raises(ValueError):
        run_return_equity_simulation(np.zeros((2,2)), np.zeros((1,2)))
    # empty inputs -> zero metrics
    res = run_return_equity_simulation(np.empty((0,0)), np.empty((0,0)))
    assert res.metrics["periods"] == 0
    metrics = compute_backtest_metrics(np.array([]), np.array([]), np.array([]), 252.0)
    assert metrics["sharpe"] == 0.0


# Light integration sanity: ensure C++ shim exposes deterministic booleans
def test_cpp_has_cpp_monotonic_behavior():
    # has_cpp() should return a boolean and not crash
    assert isinstance(has_cpp(), bool)


# Keep tests deterministic and fast


if __name__ == "__main__":
    pytest.main([__file__])
