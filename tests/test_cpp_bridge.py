import pandas as pd
import numpy as np

import tema.cpp as cpp
from tema.signals.engine import PythonSignalEngine


def make_price_df(n_rows=10, n_cols=3):
    idx = pd.date_range("2020-01-01", periods=n_rows, freq="D")
    data = np.arange(n_rows * n_cols, dtype=float).reshape(n_rows, n_cols) + 1.0
    cols = [f"A{i}" for i in range(n_cols)]
    return pd.DataFrame(data, index=idx, columns=cols)


def test_cpp_detection_returns_bool():
    assert isinstance(cpp.has_cpp(), bool)


def test_get_signal_engine_has_generate():
    eng = cpp.get_signal_engine(prefer_cpp=False)
    assert hasattr(eng, "generate")


def test_smoke_generate_fallback():
    df = make_price_df()
    eng = cpp.get_signal_engine(prefer_cpp=True)
    out = eng.generate(df, fast_period=3, slow_period=5, method="ema")
    assert isinstance(out, pd.DataFrame)
    # shape should match input
    assert out.shape == df.shape
    # values deterministic for the fallback path: when prefer_cpp=True but no C++ available,
    # engine should fall back to Python. Ensure output is finite
    assert out.notnull().values.all()
