from __future__ import annotations

import pandas as pd


def split_train_test(
    data: pd.Series | pd.DataFrame,
    train_ratio: float = 0.7,
    min_train_rows: int = 2,
    min_test_rows: int = 1,
) -> tuple[pd.Series | pd.DataFrame, pd.Series | pd.DataFrame]:
    """Chronological global split across the panel or series (backwards compatible).

    This existing function keeps the original behaviour: for a DataFrame, rows
    are split globally by index position so all assets share the same train/test
    window boundaries.
    """
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


def _split_series_per_asset(
    s: pd.Series,
    train_ratio: float,
    min_train_rows: int,
    min_test_rows: int,
) -> tuple[pd.Series, pd.Series]:
    """Split a single (non-null) series deterministically, tolerating short series.

    Rules:
    - If the series is empty: return two empty series with the same name/dtype.
    - If the series has enough rows for the configured minima, behave like
      split_train_test.
    - If the series is shorter than min_train + min_test, fall back to a
      deterministic policy that prefers putting as much as possible into train
      while guaranteeing min_test_rows when feasible. This avoids raising for
      short/absent assets in per-asset mode.
    """
    n_rows = len(s)
    empty = s.iloc[0:0].copy()
    if n_rows == 0:
        return empty, empty

    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be strictly between 0 and 1")
    if min_train_rows <= 0 or min_test_rows <= 0:
        raise ValueError("min_train_rows and min_test_rows must be > 0")

    tentative = int(n_rows * train_ratio)
    split_idx = tentative

    if n_rows >= (min_train_rows + min_test_rows):
        # Enough rows for the normal constraints
        split_idx = max(min_train_rows, min(tentative, n_rows - min_test_rows))
    else:
        # Short series: deterministic fallback
        if n_rows <= min_train_rows:
            split_idx = n_rows  # all train
        elif n_rows <= min_test_rows:
            split_idx = 0  # all test
        else:
            # Try to respect minima where possible, otherwise bias to train.
            if tentative < min_train_rows:
                if n_rows - min_train_rows >= min_test_rows:
                    split_idx = min_train_rows
                else:
                    split_idx = n_rows
            elif n_rows - tentative < min_test_rows:
                if tentative >= min_test_rows:
                    split_idx = n_rows - min_test_rows
                else:
                    split_idx = 0

    # safety clamp
    split_idx = max(0, min(split_idx, n_rows))
    train = s.iloc[:split_idx].copy()
    test = s.iloc[split_idx:].copy()
    return train, test


def split_panel_per_asset(
    panel: pd.DataFrame,
    train_ratio: float = 0.7,
    min_train_rows: int = 2,
    min_test_rows: int = 1,
    as_frames: bool = True,
) -> tuple[dict | pd.DataFrame, dict | pd.DataFrame]:
    """Split each asset (column) independently by chronological order.

    - For each column: drop NaNs, split that series independently using the
      deterministic short-series policy above.
    - Returns either two dicts (asset -> Series) when as_frames=False, or two
      DataFrames (train_df, test_df) when as_frames=True. The DataFrames are
      aligned on the union of timestamps and will contain NaNs where an asset
      has no value for that timestamp or is not present in that partition.

    This API is intended to be used by template-style workflows that split
    each asset separately and align later.
    """
    if not isinstance(panel, pd.DataFrame):
        raise ValueError("panel must be a pandas DataFrame with assets as columns")

    train_dict: dict[str, pd.Series] = {}
    test_dict: dict[str, pd.Series] = {}

    for col in panel.columns:
        col_series = panel[col].dropna()
        train_s, test_s = _split_series_per_asset(
            col_series, train_ratio=train_ratio, min_train_rows=min_train_rows, min_test_rows=min_test_rows
        )
        # keep the original column name
        train_s.name = col
        test_s.name = col
        train_dict[col] = train_s
        test_dict[col] = test_s

    if as_frames:
        # pd.DataFrame from a dict of Series aligns on the union of indices
        train_df = pd.DataFrame(train_dict, dtype=float)
        test_df = pd.DataFrame(test_dict, dtype=float)
        # Ensure columns order matches input
        train_df = train_df.reindex(columns=panel.columns)
        test_df = test_df.reindex(columns=panel.columns)
        return train_df, test_df
    return train_dict, test_dict


# convenience alias
split_train_test_per_asset = split_panel_per_asset
