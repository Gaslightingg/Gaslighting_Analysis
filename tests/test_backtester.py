from __future__ import annotations

import pandas as pd

from src.backtester.engine import run_backtest


def test_backtest_basic_metrics() -> None:
    df = pd.DataFrame(
        {
            "Open": [100, 101, 102, 103, 104],
            "High": [101, 102, 103, 104, 105],
            "Low": [99, 100, 101, 102, 103],
            "Close": [100, 101, 102, 103, 104],
            "Volume": [1000, 1000, 1000, 1000, 1000],
        }
    )
    pos = pd.Series([0, 1, 1, 0, 0])
    bt, metrics = run_backtest(df, pos)

    assert "equity" in bt.columns
    assert metrics["trades_count"] >= 1
    assert isinstance(metrics["score"], float)
