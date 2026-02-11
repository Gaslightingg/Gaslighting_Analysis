from __future__ import annotations

import pandas as pd


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


def generate_positions(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    ema_fast = _pick_col(df, "ema_fast")
    ema_slow = _pick_col(df, "ema_slow")
    rsi = _pick_col(df, "rsi")
    close = _pick_col(df, "Close")
    bb_lower = _pick_col(df, "bb_lower")
    bb_upper = _pick_col(df, "bb_upper")
    adx = _pick_col(df, "adx")

    ema_cross = pd.Series(0, index=df.index, dtype=int)
    ema_cross[ema_fast > ema_slow] = 1
    ema_cross[ema_fast < ema_slow] = -1

    buy_below = float(config["buy_below"])
    sell_above = float(config["sell_above"])

    rsi_signal = pd.Series(0, index=df.index, dtype=int)
    rsi_signal[rsi < buy_below] = 1
    rsi_signal[rsi > sell_above] = -1

    bb_signal = pd.Series(0, index=df.index, dtype=int)
    bb_signal[close < bb_lower] = 1
    bb_signal[close > bb_upper] = -1

    regime_ok = adx >= float(config["adx_min"])

    # Enter is EMA trend confirmation + (RSI OR BB) trigger.
    entry_ok = (ema_cross == 1) & ((rsi_signal == 1) | (bb_signal == 1)) & regime_ok

    # Exit is weak score OR trend reversal OR regime loss.
    score = ema_cross + rsi_signal + bb_signal
    exit_long = int(config["exit_long"])
    exit_ok = (score <= exit_long) | (ema_cross == -1) | (~regime_ok)

    signal = pd.Series(0, index=df.index, dtype=int)
    position = pd.Series(0, index=df.index, dtype=int)

    current = 0
    for i, _ in enumerate(df.index):
        if i == 0:
            continue
        if current == 0 and bool(entry_ok.iloc[i]):
            current = 1
            signal.iloc[i] = 1
        elif current == 1 and bool(exit_ok.iloc[i]):
            current = 0
            signal.iloc[i] = -1
        position.iloc[i] = current

    out = pd.DataFrame(index=df.index)
    out["score"] = score
    out["regime_ok"] = regime_ok
    out["ema_cross"] = ema_cross
    out["rsi_signal"] = rsi_signal
    out["bb_signal"] = bb_signal
    out["signal"] = signal
    out["position"] = position
    return out
