from __future__ import annotations

import pandas as pd

from src.backtester.engine import run_backtest
from src.reporter.report import normalize_equity_df


def test_next_open_execution_and_tp_sl_fields() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    df = pd.DataFrame(
        {
            "Open": [100, 100, 101, 102, 103, 104],
            "High": [101, 102, 103, 104, 105, 106],
            "Low": [99, 99, 100, 101, 102, 103],
            "Close": [100, 101, 102, 103, 104, 105],
            "Volume": [1000] * 6,
        },
        index=idx,
    )
    signal = pd.DataFrame({"signal": [0, 1, 0, 0, -1, 0], "position": [0, 1, 1, 1, 0, 0]}, index=idx)
    bt, metrics, trades = run_backtest(
        df,
        signal,
        commission_bps=0,
        slippage_bps=0,
        initial_cash=10000,
        position_size_pct=0.01,
        sl_pct=0.01,
        tp_pct=0.03,
    )
    assert "equity" in bt.columns
    assert metrics["final_equity"] == float(bt["equity"].iloc[-1])
    if trades:
        assert "exit_reason" in trades[0]
        assert "R" in trades[0]


def test_equity_normalization_series() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    s = pd.Series([10000, 10010, 10020], index=idx)
    out = normalize_equity_df(s)
    assert "equity" in out.columns
    assert out["equity"].isna().sum() == 0


def test_no_entry_columns_no_exception_and_no_entries_invariant() -> None:
    idx = pd.date_range("2024-02-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "Open": [100, 101, 102, 103, 104],
            "High": [101, 102, 103, 104, 105],
            "Low": [99, 100, 101, 102, 103],
            "Close": [100, 101, 102, 103, 104],
            "Volume": [1000] * 5,
        },
        index=idx,
    )
    signal_df = pd.DataFrame(index=idx)

    bt, metrics, trades = run_backtest(
        df,
        signal_df,
        commission_bps=0,
        slippage_bps=0,
        initial_cash=10000,
        position_size_pct=0.01,
    )

    assert not bt.empty
    assert len(trades) == 0
    assert metrics["trades_count"] == 0
    assert metrics["final_equity"] == 10000.0
    assert metrics["profit_$"] == 0.0
    assert metrics["profit_%"] == 0.0
