from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import SETTINGS


def get_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    cache_dir = Path(SETTINGS.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{ticker}_{start}_{end}.parquet"

    if cache_file.exists():
        df = pd.read_parquet(cache_file)
    else:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            progress=False,
            auto_adjust=False,
        )
        if df.empty:
            msg = f"No data for ticker={ticker}"
            raise ValueError(msg)
        df = df.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]]
        df.to_parquet(cache_file)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    return df.copy()
