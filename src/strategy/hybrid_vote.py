from __future__ import annotations

import pandas as pd


# Explicit event codes for compatibility with logs/diagnostics.
SIGNAL_HOLD = 0
SIGNAL_ENTER_LONG = 1
SIGNAL_ENTER_SHORT = -1
SIGNAL_EXIT_LONG = 2
SIGNAL_EXIT_SHORT = -2


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

    buy_below = float(config["buy_below"])
    sell_above = float(config["sell_above"])

    ema_trend_up = ema_fast > ema_slow
    ema_trend_down = ema_fast < ema_slow
    rsi_buy = rsi < buy_below
    rsi_sell = rsi > sell_above
    bb_buy = close < bb_lower
    bb_sell = close > bb_upper

    regime_mode = str(config.get("regime_mode", "on"))
    if regime_mode == "off":
        regime_ok = pd.Series(True, index=df.index)
    else:
        regime_ok = adx >= float(config["adx_min"])

    entry_votes = (
        ema_trend_up.astype(int)
        + rsi_buy.astype(int)
        + bb_buy.astype(int)
        + regime_ok.astype(int)
    )
    exit_votes = (
        ema_trend_down.astype(int)
        + rsi_sell.astype(int)
        + bb_sell.astype(int)
        + (~regime_ok).astype(int)
    )

    enter_required = int(config.get("enter_long", 2))
    exit_required = int(config.get("exit_long", 1))
    entry_ok = entry_votes >= enter_required
    exit_ok = exit_votes >= exit_required

    signal = pd.Series(SIGNAL_HOLD, index=df.index, dtype=int)
    position = pd.Series(0, index=df.index, dtype=int)
    enter_long = pd.Series(0, index=df.index, dtype=int)
    exit_long = pd.Series(0, index=df.index, dtype=int)
    enter_short = pd.Series(0, index=df.index, dtype=int)
    exit_short = pd.Series(0, index=df.index, dtype=int)

    allow_short = bool(config.get("allow_short", True))

    current = 0  # -1 short, 0 flat, 1 long
    for i, _ in enumerate(df.index):
        if i == 0:
            continue

        want_long = bool(entry_ok.iloc[i])
        want_short = bool(exit_ok.iloc[i])

        if current == 0:
            if want_long and not want_short:
                current = 1
                signal.iloc[i] = SIGNAL_ENTER_LONG
                enter_long.iloc[i] = 1
            elif allow_short and want_short and not want_long:
                current = -1
                signal.iloc[i] = SIGNAL_ENTER_SHORT
                enter_short.iloc[i] = 1
        elif current == 1:
            if allow_short and want_short and not want_long:
                # FLIP long -> short on same signal bar (engine executes next bar)
                current = -1
                signal.iloc[i] = SIGNAL_ENTER_SHORT
                exit_long.iloc[i] = 1
                enter_short.iloc[i] = 1
            elif want_short:
                current = 0
                signal.iloc[i] = SIGNAL_EXIT_LONG
                exit_long.iloc[i] = 1
        else:  # current == -1
            if want_long and not want_short:
                # FLIP short -> long
                current = 1
                signal.iloc[i] = SIGNAL_ENTER_LONG
                exit_short.iloc[i] = 1
                enter_long.iloc[i] = 1
            elif want_long:
                current = 0
                signal.iloc[i] = SIGNAL_EXIT_SHORT
                exit_short.iloc[i] = 1

        if not allow_short and current < 0:
            current = 0
        position.iloc[i] = current

    out = pd.DataFrame(index=df.index)
    out["regime_ok"] = regime_ok
    out["entry_votes"] = entry_votes
    out["exit_votes"] = exit_votes
    out["entry_ok"] = entry_ok.astype(int)
    out["exit_ok"] = exit_ok.astype(int)
    out["enter_long"] = enter_long
    out["exit_long"] = exit_long
    out["enter_short"] = enter_short
    out["exit_short"] = exit_short
    out["signal"] = signal
    out["position"] = position
    out["entry_vote_components"] = (
        "ema_up=" + ema_trend_up.astype(int).astype(str)
        + ",rsi_buy=" + rsi_buy.astype(int).astype(str)
        + ",bb_buy=" + bb_buy.astype(int).astype(str)
        + ",regime_ok=" + regime_ok.astype(int).astype(str)
    )
    out["exit_vote_components"] = (
        "ema_down=" + ema_trend_down.astype(int).astype(str)
        + ",rsi_sell=" + rsi_sell.astype(int).astype(str)
        + ",bb_sell=" + bb_sell.astype(int).astype(str)
        + ",regime_bad=" + (~regime_ok).astype(int).astype(str)
    )
    return out
