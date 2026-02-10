from __future__ import annotations

import pandas as pd


def generate_hybrid_signals(df: pd.DataFrame, config: dict) -> pd.Series:
    ema_vote = (df["ema_fast"] > df["ema_slow"]).astype(int).replace({0: -1})

    rsi_buy_below = float(config.get("rsi_buy_below", 30))
    rsi_sell_above = float(config.get("rsi_sell_above", 70))
    rsi_vote = pd.Series(0, index=df.index)
    rsi_vote = rsi_vote.mask(df["rsi"] < rsi_buy_below, 1)
    rsi_vote = rsi_vote.mask(df["rsi"] > rsi_sell_above, -1)

    bb_vote = pd.Series(0, index=df.index)
    bb_vote = bb_vote.mask(df["Close"] < df["bb_lower"], 1)
    bb_vote = bb_vote.mask(df["Close"] > df["bb_upper"], -1)

    score = ema_vote + rsi_vote + bb_vote
    adx_min = float(config.get("adx_min", 20))
    regime_ok = df["adx"] >= adx_min

    enter_long = int(config.get("enter_long", 1))
    exit_long = int(config.get("exit_long", 0))

    position = pd.Series(0, index=df.index, dtype="int64")
    current = 0
    for i, idx in enumerate(df.index):
        if i == 0:
            position.loc[idx] = 0
            continue
        if current == 0 and score.iloc[i] >= enter_long and regime_ok.iloc[i]:
            current = 1
        elif current == 1 and (score.iloc[i] <= exit_long or not regime_ok.iloc[i]):
            current = 0
        position.loc[idx] = current

    return position
