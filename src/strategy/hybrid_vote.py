from __future__ import annotations

import pandas as pd


def generate_positions(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    ema_vote = (df["ema_fast"] > df["ema_slow"]).astype(int).replace({0: -1})

    buy_below = float(config["buy_below"])
    sell_above = float(config["sell_above"])
    rsi_vote = pd.Series(0, index=df.index)
    rsi_vote[df["rsi"] < buy_below] = 1
    rsi_vote[df["rsi"] > sell_above] = -1

    bb_vote = pd.Series(0, index=df.index)
    bb_vote[df["Close"] < df["bb_lower"]] = 1
    bb_vote[df["Close"] > df["bb_upper"]] = -1

    score = ema_vote + rsi_vote + bb_vote
    regime_ok = df["adx"] >= float(config["adx_min"])

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
