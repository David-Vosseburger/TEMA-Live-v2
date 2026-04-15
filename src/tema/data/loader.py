from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_DATA_CANDIDATES: tuple[str, ...] = ("merged_d1", "merged")
_CLOSE_CANDIDATES: tuple[str, ...] = ("close_mid", "close", "Close", "close_bid", "close_ask")


def resolve_data_dir(data_path: str | Path | None = None, root: str | Path | None = None) -> Path:
    base = Path.cwd() if root is None else Path(root)
    if data_path is not None:
        candidate = Path(data_path)
        if not candidate.is_absolute():
            candidate = base / candidate
        if not candidate.exists() or not candidate.is_dir():
            raise FileNotFoundError(f"Data path does not exist or is not a directory: {candidate}")
        return candidate

    for rel in DEFAULT_DATA_CANDIDATES:
        candidate = base / rel
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"No data directory found. Tried: {', '.join(str(base / c) for c in DEFAULT_DATA_CANDIDATES)}"
    )


def find_price_files(data_dir: str | Path, max_assets: int | None = None) -> list[Path]:
    root = Path(data_dir)
    files = sorted(root.glob("*_merged.csv"))
    if max_assets is not None:
        if max_assets <= 0:
            raise ValueError("max_assets must be > 0 when provided")
        files = files[:max_assets]
    return files


def _parse_datetime_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    if "datetime" in df.columns:
        idx = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    elif "timestamp" in df.columns:
        idx = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
    else:
        raise KeyError("Required datetime column missing: expected one of ['datetime', 'timestamp']")
    return pd.DatetimeIndex(idx)


def _detect_close_column(columns: Iterable[str]) -> str:
    for col in _CLOSE_CANDIDATES:
        if col in columns:
            return col
    raise KeyError(
        f"Required close column missing: expected one of {list(_CLOSE_CANDIDATES)}"
    )


def _parse_asset_name(file_path: Path) -> str:
    name = file_path.name
    if name.endswith("_d1_merged.csv"):
        return name.removesuffix("_d1_merged.csv")
    if name.endswith("_merged.csv"):
        return name.removesuffix("_merged.csv")
    return file_path.stem


def load_close_series_from_csv(file_path: str | Path) -> pd.Series:
    fp = Path(file_path)
    if not fp.exists():
        raise FileNotFoundError(f"CSV file not found: {fp}")

    df = pd.read_csv(fp)
    if df.empty:
        raise ValueError(f"CSV is empty: {fp}")

    close_col = _detect_close_column(df.columns)
    idx = _parse_datetime_index(df)
    close = pd.to_numeric(df[close_col], errors="coerce")

    series = pd.Series(close.to_numpy(dtype=float), index=idx, name=_parse_asset_name(fp))
    series = series[~series.index.isna()].sort_index()
    series = series[~series.index.duplicated(keep="last")]
    series = series.dropna()
    if series.empty:
        raise ValueError(f"No valid close values after parsing: {fp}")
    return series


def load_price_panel(
    data_path: str | Path | None = None,
    root: str | Path | None = None,
    max_assets: int | None = None,
    min_rows: int = 3,
) -> pd.DataFrame:
    if min_rows <= 0:
        raise ValueError("min_rows must be > 0")

    data_dir = resolve_data_dir(data_path=data_path, root=root)
    files = find_price_files(data_dir, max_assets=max_assets)
    if not files:
        raise FileNotFoundError(f"No '*_merged.csv' files found in {data_dir}")

    series_list: list[pd.Series] = []
    for fp in files:
        s = load_close_series_from_csv(fp)
        if len(s) >= min_rows:
            series_list.append(s)

    if not series_list:
        raise ValueError("No assets with sufficient rows after loading and filtering")

    panel = pd.concat(series_list, axis=1).sort_index()
    panel = panel.dropna(how="all")
    if panel.empty:
        raise ValueError("Loaded panel is empty after alignment")
    return panel
