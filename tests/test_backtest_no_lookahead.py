from __future__ import annotations

import pandas as pd

from src.backtester.engine import run_backtest


def test_no_lookahead_position_shift() -> None:
    df = pd.DataFrame(
        {
            "Open": [100, 110, 121, 133.1],
            "High": [101, 111, 122, 134],
            "Low": [99, 109, 120, 132],
            "Close": [100, 110, 121, 133],
            "Volume": [1000, 1000, 1000, 1000],
        }
    )
    signal_df = pd.DataFrame({"signal": [0, 1, 0, -1], "position": [0, 1, 1, 0]})
    bt, _metrics, _trades = run_backtest(
        df,
        signal_df,
        commission_bps=0,
        slippage_bps=0,
        initial_cash=10000,
        position_size_pct=0.01,
    )
    assert bt["executed_position"].iloc[1] == 0
    assert bt["executed_position"].iloc[2] == 1


def test_missing_votes_columns_does_not_crash() -> None:
    df = pd.DataFrame(
        {
            "Open": [100, 101, 102, 103, 104],
            "High": [101, 102, 103, 104, 105],
            "Low": [99, 100, 101, 102, 103],
            "Close": [100, 101, 102, 103, 104],
            "Volume": [1000, 1000, 1000, 1000, 1000],
        }
    )
    # baseline-like signal frame without entry/exit votes columns
    signal_df = pd.DataFrame({"signal": [0, 1, 0, -1, 0], "position": [0, 1, 1, 0, 0]})

    bt, metrics, trades = run_backtest(
        df,
        signal_df,
        commission_bps=0,
        slippage_bps=0,
        initial_cash=10000,
        position_size_pct=0.01,
    )

    assert "entry_votes" in bt.columns
    assert "exit_votes" in bt.columns
    assert "entry_vote_components" in bt.columns
    assert "exit_vote_components" in bt.columns
    assert metrics["final_equity"] == float(bt["equity"].iloc[-1])
    assert isinstance(trades, list)
