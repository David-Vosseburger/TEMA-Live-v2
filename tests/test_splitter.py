import pandas as pd

from tema.data import split_panel_per_asset


def test_split_panel_per_asset_basic():
    idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"])
    a = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx, name="a")
    b = pd.Series([10.0, 11.0, 12.0], index=idx[:3], name="b")
    df = pd.DataFrame({"a": a, "b": b})

    train_df, test_df = split_panel_per_asset(df, train_ratio=0.5, min_train_rows=2, min_test_rows=1)

    # a has 4 rows -> split 2/2
    assert a.index[0] in train_df.index
    assert len(train_df['a'].dropna()) == 2
    assert len(test_df['a'].dropna()) == 2

    # b has 3 rows -> deterministic split should give 2 train, 1 test
    assert len(train_df['b'].dropna()) == 2
    assert len(test_df['b'].dropna()) == 1


def test_split_panel_per_asset_short_and_empty():
    idx = pd.to_datetime(["2020-01-01", "2020-01-02"])
    # c: single-row series
    c = pd.Series([100.0], index=[idx[0]], name="c")
    # d: empty series (all NaN)
    d = pd.Series([pd.NA, pd.NA], index=idx, name="d")
    df = pd.DataFrame({"c": c, "d": d})

    train_df, test_df = split_panel_per_asset(df, train_ratio=0.5, min_train_rows=1, min_test_rows=1)

    # c has 1 row and min_train_rows=1,min_test_rows=1 -> should go all into train
    assert len(train_df['c'].dropna()) == 1
    assert len(test_df['c'].dropna()) == 0

    # d has no valid rows -> both partitions empty (column present but all NaN)
    assert train_df['d'].dropna().empty
    assert test_df['d'].dropna().empty
