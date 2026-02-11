from src.optimizer.optuna_runner import _ensure_trial_metrics


def test_signals_count_enter_not_less_than_trades_count() -> None:
    metrics = {"signals_count_enter": 0, "signals_count_exit": 0, "final_equity": 10100.0}
    trades = [{"pnl_$": 10.0}, {"pnl_$": -5.0}]

    out = _ensure_trial_metrics(metrics, initial_cash=10000.0, trades=trades)

    assert out["trades_count"] == 2
    assert out["signals_count_enter"] >= out["trades_count"]
    assert out["final_equity"] == 10100.0


def test_zero_trades_equity_resets_to_start_cash() -> None:
    metrics = {"signals_count_enter": 0, "final_equity": 9500.0}

    out = _ensure_trial_metrics(metrics, initial_cash=10000.0, trades=[])

    assert out["trades_count"] == 0
    assert out["final_equity"] == 10000.0
    assert out["profit_$"] == 0.0
    assert out["profit_%"] == 0.0
