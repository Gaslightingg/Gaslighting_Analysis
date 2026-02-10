from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(".cache")


def get_ohlcv_daily(ticker: str, start: str, end: str) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{ticker}_{start}_{end}.parquet"
    if cache_file.exists():
        df = pd.read_parquet(cache_file)
    else:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        if df.empty:
            raise ValueError(f"No data for {ticker}")
        df = df.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]]
        df.to_parquet(cache_file)
    return df.sort_index().copy()
