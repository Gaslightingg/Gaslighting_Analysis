from __future__ import annotations

import pandas as pd

from src.backtester.engine import run_backtest


def _price_df(values: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.DataFrame(
        {
            "Open": values,
            "High": [v * 1.001 for v in values],
            "Low": [v * 0.999 for v in values],
            "Close": values,
            "Volume": [1000] * len(values),
        },
        index=idx,
    )


def test_long_enter_exit_counts_match_closed_trades() -> None:
    df = _price_df([100, 101, 102, 103, 104, 105])
    idx = df.index
    sig = pd.DataFrame(index=idx)
    sig["position"] = [0, 1, 1, 0, 0, 0]
    sig["enter_long"] = [0, 1, 0, 0, 0, 0]
    sig["exit_long"] = [0, 0, 0, 1, 0, 0]

    _bt, metrics, trades = run_backtest(df, sig, 0, 0, 10000, 0.1)

    assert metrics["enter_count"] == 1
    assert metrics["entry_events_count"] == 1
    assert metrics["exit_count"] == 1
    assert metrics["exit_events_count"] == 1
    assert metrics["closed_trades_count"] == 1
    assert len(trades) == 1
    assert trades[0]["side"] == "long"


def test_short_enter_exit_counts_match_closed_trades() -> None:
    df = _price_df([105, 104, 103, 102, 101, 100])
    idx = df.index
    sig = pd.DataFrame(index=idx)
    sig["position"] = [0, -1, -1, 0, 0, 0]
    sig["enter_short"] = [0, 1, 0, 0, 0, 0]
    sig["exit_short"] = [0, 0, 0, 1, 0, 0]

    _bt, metrics, trades = run_backtest(df, sig, 0, 0, 10000, 0.1)

    assert metrics["enter_count"] == 1
    assert metrics["entry_events_count"] == 1
    assert metrics["exit_count"] == 1
    assert metrics["exit_events_count"] == 1
    assert metrics["closed_trades_count"] == 1
    assert len(trades) == 1
    assert trades[0]["side"] == "short"


def test_forced_exit_counts_as_exit_event() -> None:
    df = _price_df([100, 101, 102, 103])
    idx = df.index
    sig = pd.DataFrame(index=idx)
    sig["position"] = [0, 1, 1, 1]
    sig["enter_long"] = [0, 1, 0, 0]

    _bt, metrics, trades = run_backtest(df, sig, 0, 0, 10000, 0.1)

    assert len(trades) == 1
    assert trades[0].get("forced_exit") is True
    assert metrics["forced_exit_count"] == 1
    assert metrics["exits_forced_end"] == 1
    assert metrics["exit_count"] == 1


def test_short_positive_pnl_on_downtrend_long_negative() -> None:
    df = _price_df([110, 108, 106, 104, 102, 100])
    idx = df.index

    sig_short = pd.DataFrame(index=idx)
    sig_short["position"] = [0, -1, -1, -1, 0, 0]
    sig_short["enter_short"] = [0, 1, 0, 0, 0, 0]
    sig_short["exit_short"] = [0, 0, 0, 0, 1, 0]
    _bt_s, metrics_s, trades_s = run_backtest(df, sig_short, 0, 0, 10000, 0.1)

    sig_long = pd.DataFrame(index=idx)
    sig_long["position"] = [0, 1, 1, 1, 0, 0]
    sig_long["enter_long"] = [0, 1, 0, 0, 0, 0]
    sig_long["exit_long"] = [0, 0, 0, 0, 1, 0]
    _bt_l, metrics_l, trades_l = run_backtest(df, sig_long, 0, 0, 10000, 0.1)

    assert trades_s[0]["pnl_$"] > 0
    assert trades_l[0]["pnl_$"] < 0
    assert metrics_s["profit_$"] > 0
    assert metrics_l["profit_$"] < 0


def test_no_double_entry_same_side_without_pyramiding() -> None:
    df = _price_df([100, 101, 102, 103, 104])
    idx = df.index
    sig = pd.DataFrame(index=idx)
    sig["position"] = [0, 1, 1, 1, 0]
    sig["enter_long"] = [0, 1, 1, 1, 0]
    sig["exit_long"] = [0, 0, 0, 0, 1]

    _bt, metrics, trades = run_backtest(df, sig, 0, 0, 10000, 0.1)

    assert metrics["enter_count"] == 1
    assert len(trades) == 1


def test_flip_long_to_short_creates_close_and_new_open() -> None:
    df = _price_df([100, 101, 102, 101, 100, 99])
    idx = df.index
    sig = pd.DataFrame(index=idx)
    sig["position"] = [0, 1, -1, -1, 0, 0]
    sig["enter_long"] = [0, 1, 0, 0, 0, 0]
    sig["enter_short"] = [0, 0, 1, 0, 0, 0]
    sig["exit_short"] = [0, 0, 0, 0, 1, 0]

    _bt, metrics, trades = run_backtest(df, sig, 0, 0, 10000, 0.1)

    assert metrics["enter_count"] == 2
    assert metrics["exit_count"] == 2
    assert len(trades) == 2
    assert trades[0]["side"] == "long"
    assert trades[1]["side"] == "short"


def test_long_short_symmetry_on_reversed_market() -> None:
    df_up = _price_df([100, 102, 104, 106, 108, 110])
    idx = df_up.index
    sig_long = pd.DataFrame(index=idx)
    sig_long["position"] = [0, 1, 1, 1, 0, 0]
    sig_long["enter_long"] = [0, 1, 0, 0, 0, 0]
    sig_long["exit_long"] = [0, 0, 0, 0, 1, 0]
    _bt_a, metrics_a, trades_a = run_backtest(df_up, sig_long, 0, 0, 10000, 0.1)

    df_down = _price_df([110, 108, 106, 104, 102, 100])
    sig_short = pd.DataFrame(index=df_down.index)
    sig_short["position"] = [0, -1, -1, -1, 0, 0]
    sig_short["enter_short"] = [0, 1, 0, 0, 0, 0]
    sig_short["exit_short"] = [0, 0, 0, 0, 1, 0]
    _bt_b, metrics_b, trades_b = run_backtest(df_down, sig_short, 0, 0, 10000, 0.1)

    assert trades_a[0]["pnl_$"] > 0
    assert trades_b[0]["pnl_$"] > 0
    assert abs(metrics_a["profit_$"] - metrics_b["profit_$"]) < 1e-6


def test_flip_short_to_long_creates_close_and_new_open() -> None:
    df = _price_df([105, 104, 103, 104, 105, 106])
    idx = df.index
    sig = pd.DataFrame(index=idx)
    sig["position"] = [0, -1, 1, 1, 0, 0]
    sig["enter_short"] = [0, 1, 0, 0, 0, 0]
    sig["enter_long"] = [0, 0, 1, 0, 0, 0]
    sig["exit_long"] = [0, 0, 0, 0, 1, 0]

    _bt, metrics, trades = run_backtest(df, sig, 0, 0, 10000, 0.1)

    assert metrics["enter_count"] == 2
    assert metrics["exit_count"] == 2
    assert len(trades) == 2
    assert trades[0]["side"] == "short"
    assert trades[1]["side"] == "long"


def test_regression_exit_summary_not_zero_when_exits_exist() -> None:
    df = _price_df([100, 101, 102, 103, 102, 101, 100])
    idx = df.index
    sig = pd.DataFrame(index=idx)
    sig["position"] = [0, 1, 1, 0, -1, -1, 0]
    sig["enter_long"] = [0, 1, 0, 0, 0, 0, 0]
    sig["exit_long"] = [0, 0, 0, 1, 0, 0, 0]
    sig["enter_short"] = [0, 0, 0, 0, 1, 0, 0]
    sig["exit_short"] = [0, 0, 0, 0, 0, 0, 1]

    _bt, metrics, trades = run_backtest(df, sig, 0, 0, 10000, 0.1)

    assert len(trades) == 2
    assert metrics["exit_events_count"] == 2
    assert metrics["signals_count_exit"] >= 2
    assert metrics["exits_by_rule"] >= 2
    assert metrics["exit_events_count"] <= metrics["entry_events_count"]
