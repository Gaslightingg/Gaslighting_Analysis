from __future__ import annotations

import pandas as pd

from src.analytics.metrics import QualityThresholds, compute_diagnostic_metrics, quality_flags


def test_profit_factor_with_losses() -> None:
    trades = [{"pnl_$": 10.0, "R": 1.0}, {"pnl_$": -5.0, "R": -1.0}, {"pnl_$": 5.0, "R": 0.5}]
    eq = pd.Series([10000, 10010, 10005, 10010])
    m = compute_diagnostic_metrics(trades, eq)
    assert abs(m["profit_factor"] - 3.0) < 1e-12


def test_profit_factor_infinite_without_losses() -> None:
    trades = [{"pnl_$": 10.0, "R": 1.0}, {"pnl_$": 5.0, "R": 0.5}]
    eq = pd.Series([10000, 10015])
    m = compute_diagnostic_metrics(trades, eq)
    assert m["profit_factor"] == float("inf")


def test_max_drawdown_synthetic() -> None:
    eq = pd.Series([100, 120, 90, 95])
    m = compute_diagnostic_metrics([], eq)
    assert abs(m["max_drawdown"] - (-0.25)) < 1e-12
    assert m["dd_duration"] == 2


def test_top3_contribution() -> None:
    trades = [{"pnl_$": 10, "R": 1}, {"pnl_$": 8, "R": 1}, {"pnl_$": 6, "R": 1}, {"pnl_$": -4, "R": -1}]
    eq = pd.Series([10000, 10020])
    m = compute_diagnostic_metrics(trades, eq)
    assert abs(m["top3_contribution"] - (24 / 20)) < 1e-12


def test_consecutive_wins_losses() -> None:
    trades = [
        {"pnl_$": 1, "R": 1},
        {"pnl_$": 2, "R": 1},
        {"pnl_$": -1, "R": -1},
        {"pnl_$": -2, "R": -1},
        {"pnl_$": -3, "R": -1},
        {"pnl_$": 4, "R": 1},
    ]
    m = compute_diagnostic_metrics(trades, pd.Series([10000, 10001]))
    assert m["max_consecutive_wins"] == 2
    assert m["max_consecutive_losses"] == 3


def test_flags_threshold_switching() -> None:
    metrics = {
        "n_trades": 29,
        "profit_factor": 1.0,
        "expectancy$": 0.0,
        "max_drawdown": -0.11,
        "top3_contribution": 0.7,
        "total_pnl$": 100,
        "short_trades": 0,
        "allow_short": True,
    }
    flags = quality_flags(metrics, QualityThresholds(n_min_trades=30, pf_min=1.1, max_dd_min=-0.1, top3_max=0.6))
    assert flags["too_few_trades"] is True
    assert flags["pf_bad"] is True
    assert flags["expectancy_bad"] is True
    assert flags["dd_too_high"] is True
    assert flags["top3_dominates"] is True
    assert flags["no_short_when_allowed"] is True

    metrics["n_trades"] = 50
    metrics["profit_factor"] = 1.4
    metrics["expectancy$"] = 2.0
    metrics["max_drawdown"] = -0.05
    metrics["top3_contribution"] = 0.3
    metrics["short_trades"] = 2
    flags2 = quality_flags(metrics, QualityThresholds(n_min_trades=30, pf_min=1.1, max_dd_min=-0.1, top3_max=0.6))
    assert not any(flags2.values())
