from src.optimizer.optuna_runner import _ensure_trial_metrics


def test_trial_metrics_preserve_event_counts_from_source() -> None:
    metrics = {
        "signals_count_enter": 3,
        "signals_count_exit": 2,
        "entry_events_count": 3,
        "exit_events_count": 2,
        "final_equity": 10100.0,
    }
    trades = [{"pnl_$": 10.0}, {"pnl_$": -5.0}]

    out = _ensure_trial_metrics(metrics, initial_cash=10000.0, trades=trades)

    assert out["trades_count"] == 2
    assert out["entry_events_count"] == 3
    assert out["exit_events_count"] == 2
    assert out["closed_trades_count"] == 2
    assert out["final_equity"] == 10100.0


def test_zero_trades_equity_resets_to_start_cash() -> None:
    metrics = {"signals_count_enter": 0, "final_equity": 9500.0}

    out = _ensure_trial_metrics(metrics, initial_cash=10000.0, trades=[])

    assert out["trades_count"] == 0
    assert out["closed_trades_count"] == 0
    assert out["final_equity"] == 10000.0
    assert out["profit_$"] == 0.0
    assert out["profit_%"] == 0.0
