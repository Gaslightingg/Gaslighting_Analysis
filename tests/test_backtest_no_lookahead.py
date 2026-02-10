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
    bt, _metrics, _trades = run_backtest(df, signal_df, commission_bps=0, slippage_bps=0)
    assert bt["executed_position"].iloc[1] == 0
    assert bt["executed_position"].iloc[2] == 1
