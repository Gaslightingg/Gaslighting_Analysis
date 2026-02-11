from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import SETTINGS

_OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


@dataclass(slots=True)
class DataProviderError(Exception):
    reason: str
    message: str

    def __str__(self) -> str:
        return self.message


def _today_utc() -> date:
    return datetime.now(timezone.utc).date()


def _cache_path(ticker: str) -> Path:
    cache_dir = Path(SETTINGS.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_ticker = ticker.strip().upper().replace("/", "_")
    return cache_dir / f"{safe_ticker}.parquet"


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
        return pd.DataFrame(columns=_OHLCV_COLS)

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


def _to_stooq_symbol(ticker: str) -> str:
    t = ticker.strip().lower()
    if "." in t:
        return t
    return f"{t}.us"


def fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        progress=False,
        auto_adjust=False,
    )
    return _normalize_ohlcv(raw)


def fetch_stooq(ticker: str, start: str, end: str) -> pd.DataFrame:
    symbol = _to_stooq_symbol(ticker)
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    raw = pd.read_csv(url)
    if raw.empty:
        return pd.DataFrame(columns=_OHLCV_COLS)

    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"]).set_index("Date")
    out = _normalize_ohlcv(raw)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return out.loc[(out.index >= start_ts) & (out.index <= end_ts)]


def load_cache(ticker: str) -> pd.DataFrame:
    path = _cache_path(ticker)
    if not path.exists():
        return pd.DataFrame(columns=_OHLCV_COLS)
    return _normalize_ohlcv(pd.read_parquet(path))


def save_cache(ticker: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    path = _cache_path(ticker)
    _normalize_ohlcv(df).to_parquet(path)


def get_cached_range(ticker: str) -> dict | None:
    df = load_cache(ticker)
    if df.empty:
        return None
    return {
        "min_date": df.index.min().date().isoformat(),
        "max_date": df.index.max().date().isoformat(),
        "bars_count": int(len(df)),
    }


def _is_network_error(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = ["timeout", "connection", "temporar", "name resolution", "dns", "network", "ssl"]
    return any(m in text for m in markers)


def _merge(base: pd.DataFrame, add: pd.DataFrame) -> pd.DataFrame:
    merged = pd.concat([base, add]) if not base.empty else add.copy()
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged.sort_index()


def get_ohlcv_with_meta(ticker: str, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    start_d = date.fromisoformat(start)
    requested_end = date.fromisoformat(end)
    today = _today_utc()
    effective_end = min(requested_end, today)
    end_trimmed = requested_end > today

    if start_d >= effective_end:
        raise DataProviderError("future_period", "future_period: start_date >= effective_end")

    cache = load_cache(ticker)
    source = "cache"
    fetched_source = "cache"
    errors: list[str] = []
    reason = ""

    start_ts = pd.Timestamp(start_d)
    end_ts = pd.Timestamp(effective_end)

    if cache.empty or cache.index.min() > start_ts or cache.index.max() < end_ts:
        fetch_start = start_d.isoformat()
        fetch_end = effective_end.isoformat()

        got = pd.DataFrame(columns=_OHLCV_COLS)
        try:
            got = fetch_yfinance(ticker, fetch_start, fetch_end)
            fetched_source = "yf"
        except Exception as exc:  # noqa: BLE001
            errors.append(f"yf:{exc}")
            if _is_network_error(exc):
                reason = "network_error"

        if got.empty:
            try:
                got = fetch_stooq(ticker, fetch_start, fetch_end)
                fetched_source = "stooq"
            except Exception as exc:  # noqa: BLE001
                errors.append(f"stooq:{exc}")
                if _is_network_error(exc):
                    reason = "network_error"

        if got.empty and cache.empty:
            if reason == "network_error":
                raise DataProviderError("network_error", f"network_error: {'; '.join(errors) or 'all providers failed'}")
            raise DataProviderError("provider_empty", f"provider_empty: {'; '.join(errors) or 'providers returned empty'}")

        if not got.empty:
            cache = _merge(cache, got)
            save_cache(ticker, cache)
            source = fetched_source

    sliced = cache.loc[(cache.index >= start_ts) & (cache.index <= end_ts)].copy()
    if sliced.empty:
        raise DataProviderError("provider_empty", "provider_empty: cache/providers do not cover requested range")

    meta = {
        "min_date": sliced.index.min().date().isoformat(),
        "max_date": sliced.index.max().date().isoformat(),
        "bars_count": int(len(sliced)),
        "source": source,
        "end_trimmed": bool(end_trimmed),
        "requested_end": requested_end.isoformat(),
        "effective_end": effective_end.isoformat(),
        "reason": reason or ("future_period" if end_trimmed else ""),
    }
    return sliced, meta


def get_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    df, _meta = get_ohlcv_with_meta(ticker, start, end)
    return df
