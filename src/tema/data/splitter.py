from __future__ import annotations

import pandas as pd


def split_train_test(
    data: pd.Series | pd.DataFrame,
    train_ratio: float = 0.7,
    min_train_rows: int = 2,
    min_test_rows: int = 1,
) -> tuple[pd.Series | pd.DataFrame, pd.Series | pd.DataFrame]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be strictly between 0 and 1")
    if min_train_rows <= 0 or min_test_rows <= 0:
        raise ValueError("min_train_rows and min_test_rows must be > 0")

    n_rows = len(data)
    min_total = min_train_rows + min_test_rows
    if n_rows < min_total:
        raise ValueError(f"Not enough rows ({n_rows}) for split; need at least {min_total}")

    split_idx = int(n_rows * train_ratio)
    split_idx = max(min_train_rows, min(split_idx, n_rows - min_test_rows))
    train = data.iloc[:split_idx].copy()
    test = data.iloc[split_idx:].copy()
    return train, test
