from __future__ import annotations

import pandas as pd

from src.strategy.hybrid_vote import generate_positions


def test_strategy_callable() -> None:
    assert callable(generate_positions)


def test_generate_positions_handles_multiindex_close_column() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    base = pd.DataFrame(
        {
            "Close": [100, 101, 99, 103, 105],
            "ema_fast": [100, 101, 100, 102, 104],
            "ema_slow": [100, 100, 100, 101, 102],
            "rsi": [30, 35, 25, 80, 60],
            "bb_lower": [99, 100, 100, 101, 102],
            "bb_upper": [101, 102, 103, 104, 106],
            "adx": [30, 30, 30, 30, 30],
        },
        index=idx,
    )

    multi_close = pd.concat(
        [base.drop(columns=["Close"]), pd.concat({"SPY": base[["Close"]]}, axis=1)],
        axis=1,
    )

    out = generate_positions(
        multi_close,
        {
            "buy_below": 40,
            "sell_above": 70,
            "adx_min": 20,
            "enter_long": 1,
            "exit_long": 0,
        },
    )

    required = {"regime_ok", "signal", "position", "entry_votes", "exit_votes", "entry_ok", "exit_ok"}
    assert required.issubset(set(out.columns))
    assert len(out) == len(base)
