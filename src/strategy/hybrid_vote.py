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

    ema_vote = (ema_fast > ema_slow).astype(int).replace({0: -1})

    buy_below = float(config["buy_below"])
    sell_above = float(config["sell_above"])
    rsi_vote = pd.Series(0, index=df.index)
    rsi_vote[rsi < buy_below] = 1
    rsi_vote[rsi > sell_above] = -1

    bb_vote = pd.Series(0, index=df.index)
    bb_vote[close < bb_lower] = 1
    bb_vote[close > bb_upper] = -1

    score = ema_vote + rsi_vote + bb_vote
    regime_ok = adx >= float(config["adx_min"])

    enter_long = int(config["enter_long"])
    exit_long = int(config["exit_long"])

    signal = pd.Series(0, index=df.index, dtype=int)
    position = pd.Series(0, index=df.index, dtype=int)

    current = 0
    for i, _ in enumerate(df.index):
        if i == 0:
            continue
        if current == 0 and score.iloc[i] >= enter_long and bool(regime_ok.iloc[i]):
            current = 1
            signal.iloc[i] = 1
        elif current == 1 and (score.iloc[i] <= exit_long or not bool(regime_ok.iloc[i])):
            current = 0
            signal.iloc[i] = -1
        position.iloc[i] = current

    out = pd.DataFrame(index=df.index)
    out["score"] = score
    out["regime_ok"] = regime_ok
    out["signal"] = signal
    out["position"] = position
    return out
