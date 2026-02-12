from __future__ import annotations

import json

import pandas as pd

from src.reporter.diagnostics import build_diagnostic_summary, save_diagnostic_artifacts


def _bt(prices: list[float], qty: list[float] | None = None) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="D")
    q = qty if qty is not None else [0.0] * len(prices)
    cash = [10000.0] * len(prices)
    equity = [cash[i] + q[i] * prices[i] for i in range(len(prices))]
    return pd.DataFrame({"Close": prices, "qty": q, "cash": cash, "equity": equity}, index=idx)


def _trade(side: str, ep: float, xp: float, forced: bool = False, reason: str = "signal") -> dict:
    return {
        "side": side,
        "entry_date": "2024-01-02",
        "exit_date": "2024-01-03",
        "entry_price": ep,
        "exit_price": xp,
        "qty": 1.0,
        "pnl_$": (xp - ep) if side == "long" else (ep - xp),
        "pnl": ((xp - ep) / ep) if side == "long" else ((ep - xp) / ep),
        "R": 1.0 if ((xp - ep) if side == "long" else (ep - xp)) > 0 else -1.0,
        "holding_days": 1,
        "holding_bars": 1,
        "exit_reason": reason,
        "forced_exit": forced,
    }


def test_long_win_metrics() -> None:
    bt = _bt([100, 101, 102])
    trades = [_trade("long", 100, 102)]
    s = build_diagnostic_summary(bt, trades, 10000)
    assert s["model"]["trades_closed"] == 1
    assert s["trade_metrics"]["winrate"] == 1.0


def test_long_loss_metrics() -> None:
    bt = _bt([100, 99, 98])
    trades = [_trade("long", 100, 98)]
    s = build_diagnostic_summary(bt, trades, 10000)
    assert s["trade_metrics"]["winrate"] == 0.0
    assert s["trade_metrics"]["profit_factor"] == 0.0


def test_short_profit_on_downtrend() -> None:
    bt = _bt([110, 108, 106])
    trades = [_trade("short", 110, 106)]
    s = build_diagnostic_summary(bt, trades, 10000)
    assert s["trade_metrics"]["avg_win_$"] > 0


def test_forced_exit_count() -> None:
    bt = _bt([100, 101, 102])
    trades = [_trade("long", 100, 102, forced=True, reason="forced_eod")]
    s = build_diagnostic_summary(bt, trades, 10000)
    assert s["model"]["forced_exit_count"] == 1


def test_flip_reason_distribution() -> None:
    bt = _bt([100, 101, 102])
    trades = [_trade("long", 100, 101, reason="flip")]
    s = build_diagnostic_summary(bt, trades, 10000)
    assert s["trade_metrics"]["exit_reason_distribution"]["flip"] == 1


def test_exposure_always_in_market() -> None:
    bt = _bt([100, 100, 100], qty=[1, 1, 1])
    s = build_diagnostic_summary(bt, [], 10000)
    assert s["equity_metrics"]["exposure"] == 1.0


def test_exposure_always_flat() -> None:
    bt = _bt([100, 100, 100], qty=[0, 0, 0])
    s = build_diagnostic_summary(bt, [], 10000)
    assert s["equity_metrics"]["exposure"] == 0.0


def test_drawdown_known_series() -> None:
    bt = _bt([100, 120, 90, 95], qty=[0, 0, 0, 0])
    bt["equity"] = [100, 120, 90, 95]
    s = build_diagnostic_summary(bt, [], 100)
    assert abs(s["equity_metrics"]["max_drawdown"] - 0.25) < 1e-9


def test_equity_last_equals_final_invariant() -> None:
    bt = _bt([100, 101, 102])
    s = build_diagnostic_summary(bt, [], 10000)
    assert s["invariants"]["equity_last_equals_final"] is True


def test_summary_json_and_csv_artifacts(tmp_path) -> None:
    bt = _bt([100, 101, 102])
    trades = [_trade("long", 100, 102)]
    summary, artifacts = save_diagnostic_artifacts(tmp_path, bt, trades, 10000)
    data = json.loads((tmp_path / "summary.json").read_text())
    assert "model" in data and "trade_metrics" in data and "equity_metrics" in data
    assert (tmp_path / "trades.csv").exists()
    assert (tmp_path / "equity.csv").exists()
    assert summary["model"]["trades_closed"] == 1
    assert artifacts.summary_json.endswith("summary.json")
