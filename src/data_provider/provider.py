from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import SETTINGS


_OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _as_series(value: pd.Series | pd.DataFrame, name: str) -> pd.Series:
    if isinstance(value, pd.Series):
        return value
    if isinstance(value, pd.DataFrame):
        if value.shape[1] == 0:
            raise ValueError(f"Column {name} is empty")
        return value.iloc[:, 0]
    raise TypeError(f"Unsupported type for {name}: {type(value)!r}")


def _pick_col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return _as_series(df[name], name)

    lower_name = name.lower()
    for col in df.columns:
        if str(col).lower() == lower_name:
            return _as_series(df[col], name)

    if isinstance(df.columns, pd.MultiIndex):
        for level in range(df.columns.nlevels):
            values = df.columns.get_level_values(level)
            mask = values.astype(str).str.lower() == lower_name
            if mask.any():
                return _as_series(df.loc[:, mask], name)

    for col in df.columns:
        if isinstance(col, tuple) and any(str(part).lower() == lower_name for part in col):
            return _as_series(df[col], name)

    raise KeyError(f"Missing column {name}")


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = pd.DataFrame(index=df.index)
    for name in _OHLCV_COLS:
        out[name] = pd.to_numeric(_pick_col(df, name), errors="coerce")

    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    out["Volume"] = out["Volume"].fillna(0.0)

    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    return out


def _download_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        progress=False,
        auto_adjust=False,
    )
    return _normalize_ohlcv(raw)


def _to_stooq_symbol(ticker: str) -> str:
    t = ticker.strip().lower()
    if "." in t:
        return t
    return f"{t}.us"


def _download_stooq(ticker: str, start: str, end: str) -> pd.DataFrame:
    symbol = _to_stooq_symbol(ticker)
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    raw = pd.read_csv(url)
    if raw.empty:
        return raw
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"]).set_index("Date")
    out = _normalize_ohlcv(raw)

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    out = out.loc[(out.index >= start_ts) & (out.index <= end_ts)]
    return out


def get_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    cache_dir = Path(SETTINGS.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{ticker}_{start}_{end}.parquet"

    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        return _normalize_ohlcv(df).copy()

    errors: list[str] = []

    try:
        df = _download_yfinance(ticker, start, end)
        if not df.empty:
            df.to_parquet(cache_file)
            return df.copy()
        errors.append("yfinance: empty")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"yfinance: {exc}")

    try:
        df = _download_stooq(ticker, start, end)
        if not df.empty:
            df.to_parquet(cache_file)
            return df.copy()
        errors.append("stooq: empty")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"stooq: {exc}")

    msg = f"No data for ticker={ticker}. Tried providers: {'; '.join(errors)}"
    raise ValueError(msg)
