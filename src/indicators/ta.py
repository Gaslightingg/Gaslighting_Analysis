from __future__ import annotations

import pandas as pd


def add_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]
    volume = out["Volume"]

    ema_fast = int(config.get("ema_fast", 12))
    ema_slow = int(config.get("ema_slow", 26))
    out["ema_fast"] = close.ewm(span=ema_fast, adjust=False).mean()
    out["ema_slow"] = close.ewm(span=ema_slow, adjust=False).mean()

    rsi_period = int(config.get("rsi_period", 14))
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs = gain / loss.replace(0, pd.NA)
    out["rsi"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()

    tr = pd.concat(
        [(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    atr_period = int(config.get("atr_period", 14))
    out["atr"] = tr.rolling(atr_period).mean()

    plus_dm = (high.diff()).clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_di = 100 * (plus_dm.rolling(14).mean() / out["atr"].replace(0, pd.NA))
    minus_di = 100 * (minus_dm.rolling(14).mean() / out["atr"].replace(0, pd.NA))
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, pd.NA)) * 100
    out["adx"] = dx.rolling(14).mean()

    bb_period = int(config.get("bb_period", 20))
    bb_std = float(config.get("bb_std", 2.0))
    basis = close.rolling(bb_period).mean()
    sigma = close.rolling(bb_period).std()
    out["bb_upper"] = basis + bb_std * sigma
    out["bb_lower"] = basis - bb_std * sigma

    dc_period = int(config.get("donchian_period", 20))
    out["donchian_high"] = high.rolling(dc_period).max()
    out["donchian_low"] = low.rolling(dc_period).min()

    direction = close.diff().fillna(0).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    out["obv"] = (direction * volume).cumsum()
    return out
