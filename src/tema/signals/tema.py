from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    if period <= 0:
        raise ValueError("period must be > 0")
    s = pd.Series(series, copy=False).astype(float)
    return s.ewm(span=period, adjust=False).mean()


def tema(series: pd.Series, period: int) -> pd.Series:
    e1 = ema(series, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    return 3.0 * e1 - 3.0 * e2 + e3


def generate_crossover_signal_matrix(
    price_df: pd.DataFrame,
    fast_period: int = 5,
    slow_period: int = 20,
    method: str = "ema",
    shift_by: int = 1,
) -> pd.DataFrame:
    if fast_period <= 0 or slow_period <= 0:
        raise ValueError("fast_period and slow_period must be > 0")
    if fast_period >= slow_period:
        raise ValueError("fast_period must be smaller than slow_period")
    if shift_by < 0:
        raise ValueError("shift_by must be >= 0")
    if method not in {"ema", "tema"}:
        raise ValueError("method must be one of {'ema', 'tema'}")
    if price_df.empty:
        raise ValueError("price_df must not be empty")

    signal_df = pd.DataFrame(index=price_df.index, columns=price_df.columns, dtype=float)
    smoother = ema if method == "ema" else tema

    for col in price_df.columns:
        s = pd.to_numeric(price_df[col], errors="coerce")
        fast = smoother(s, fast_period)
        slow = smoother(s, slow_period)
        signal = np.sign((fast - slow).to_numpy(dtype=float))
        signal_df[col] = signal

    if shift_by:
        signal_df = signal_df.shift(shift_by).fillna(0.0)

    return signal_df.astype(float)
