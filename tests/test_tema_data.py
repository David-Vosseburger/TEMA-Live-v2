from pathlib import Path

import pandas as pd
import pytest

from tema.data import load_close_series_from_csv, load_price_panel, split_train_test


def _write_csv(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_load_close_series_from_csv_success(tmp_path):
    fp = tmp_path / "asset_d1_merged.csv"
    _write_csv(
        fp,
        "timestamp,close_mid\n"
        "1677628800000,10\n"
        "1677715200000,11\n"
        "1677801600000,12\n",
    )
    s = load_close_series_from_csv(fp)
    assert s.name == "asset"
    assert list(s.astype(float)) == [10.0, 11.0, 12.0]


def test_load_close_series_from_csv_missing_close_column(tmp_path):
    fp = tmp_path / "asset_d1_merged.csv"
    _write_csv(fp, "timestamp,open_mid\n1677628800000,10\n")
    with pytest.raises(KeyError, match="Required close column missing"):
        load_close_series_from_csv(fp)


def test_load_price_panel_and_split(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    _write_csv(
        d / "a_d1_merged.csv",
        "timestamp,close_mid\n1677628800000,1\n1677715200000,2\n1677801600000,3\n1678060800000,4\n",
    )
    _write_csv(
        d / "b_d1_merged.csv",
        "timestamp,close_mid\n1677628800000,2\n1677715200000,3\n1677801600000,4\n1678060800000,5\n",
    )

    panel = load_price_panel(data_path=d, min_rows=3)
    assert list(panel.columns) == ["a", "b"]
    train, test = split_train_test(panel, train_ratio=0.5, min_train_rows=2, min_test_rows=1)
    assert len(train) == 2
    assert len(test) == 2


def test_split_train_test_invalid_ratio():
    df = pd.DataFrame({"x": [1, 2, 3, 4]})
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        split_train_test(df, train_ratio=1.0)
