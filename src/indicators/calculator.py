from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLS = ("Open", "High", "Low", "Close", "Volume")


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

    if isinstance(df.columns, pd.MultiIndex):
        for level in range(df.columns.nlevels):
            mask = df.columns.get_level_values(level) == name
            if mask.any():
                return _as_series(df.loc[:, mask], name)

    for col in df.columns:
        if isinstance(col, tuple) and name in col:
            return _as_series(df[col], name)

    raise KeyError(f"Required column is missing: {name}")


def add_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    out = df.copy()

    close = _pick_col(out, "Close")
    high = _pick_col(out, "High")
    low = _pick_col(out, "Low")
    volume = _pick_col(out, "Volume")

    ema_fast = int(config.get("ema_fast", 12))
    ema_slow = int(config.get("ema_slow", 26))
    out["ema_fast"] = close.ewm(span=ema_fast, adjust=False).mean()
    out["ema_slow"] = close.ewm(span=ema_slow, adjust=False).mean()

    rsi_period = int(config.get("rsi_period", 14))
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / rsi_period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / rsi_period, adjust=False).mean()
    rs = gain / loss.replace(0.0, np.nan)
    out["rsi"] = 100 - 100 / (1 + rs)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr_period = int(config.get("atr_period", 14))
    out["atr"] = tr.ewm(alpha=1 / atr_period, adjust=False).mean()

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / out["atr"].replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / out["atr"].replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    out["adx"] = _as_series(dx, "adx").ewm(alpha=1 / 14, adjust=False).mean()

    bb_period = int(config.get("bb_period", 20))
    bb_std = float(config.get("bb_std", 2.0))
    bb_mean = close.rolling(bb_period, min_periods=bb_period).mean()
    bb_sigma = close.rolling(bb_period, min_periods=bb_period).std(ddof=0)
    out["bb_upper"] = bb_mean + bb_std * bb_sigma
    out["bb_lower"] = bb_mean - bb_std * bb_sigma

    don_period = int(config.get("donchian_period", 20))
    out["donchian_high"] = high.rolling(don_period, min_periods=don_period).max()
    out["donchian_low"] = low.rolling(don_period, min_periods=don_period).min()

    direction = np.sign(close.diff().fillna(0.0))
    out["obv"] = (direction * volume).cumsum()

    return out
