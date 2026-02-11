from src.backtester.engine import run_backtest


def test_backtester_callable() -> None:
    assert callable(run_backtest)
