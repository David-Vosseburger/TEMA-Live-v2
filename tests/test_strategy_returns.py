import pandas as pd
import numpy as np

from tema.strategy_returns import build_strategy_returns


def test_simple_long_no_costs():
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]}, index=[0, 1, 2])
    # signal 1 means fully long; build_strategy_returns will generate signals if None
    returns = build_strategy_returns(prices, signal_df=pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=prices.index), fee_rate=0.0, slippage_rate=0.0)
    expected = prices.pct_change().fillna(0.0)["A"].values
    assert np.allclose(returns["A"].values, expected)


def test_signal_flip_with_costs():
    prices = pd.DataFrame({"A": [100.0, 110.0, 105.0]}, index=[0, 1, 2])
    signals = pd.DataFrame({"A": [1.0, -1.0, -1.0]}, index=prices.index)
    cost = 0.01
    returns = build_strategy_returns(prices, signal_df=signals, fee_rate=cost, slippage_rate=0.0)
    pct = prices.pct_change().fillna(0.0)["A"].values
    # manual expected
    # t0: pos=1, prev=0 => turnover=1 => ret = 1*0 - 1*cost
    e0 = -1 * cost
    # t1: pos=-1, prev=1 => turnover=2 => ret = -1*pct1 - 2*cost
    e1 = -1 * pct[1] - 2 * cost
    # t2: pos=-1, prev=-1 => turnover=0 => ret = -1*pct2
    e2 = -1 * pct[2]
    expected = np.array([e0, e1, e2])
    assert np.allclose(returns["A"].values, expected)


def test_empty_prices_returns_empty():
    prices = pd.DataFrame({}, index=pd.Index([]))
    out = build_strategy_returns(prices)
    assert out.empty


def test_mismatched_signal_columns_raises():
    prices = pd.DataFrame({"A": [100.0, 101.0]}, index=[0, 1])
    signals = pd.DataFrame({"B": [1.0, 1.0]}, index=[0, 1])
    try:
        build_strategy_returns(prices, signal_df=signals)
        assert False, "Expected ValueError for mismatched columns"
    except ValueError:
        pass
