from __future__ import annotations

import pandas as pd

from src.optimizer.walk_forward import _aggregate_metrics


def test_aggregate_metrics_fills_equity_nans_at_start_and_inside() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    combined = pd.DataFrame(
        {
            "equity": [float("nan"), 1.0, float("nan"), 1.1, 1.2],
            "strategy_ret": [0.0, 0.01, -0.01, 0.02, 0.0],
        },
        index=idx,
    )

    metrics = _aggregate_metrics(combined, trades=[])

    equity_filled = combined["equity"].ffill().bfill().fillna(1.0)
    assert equity_filled.isna().sum() == 0
    assert float(equity_filled.iloc[0]) == 1.0
    assert float(metrics["final_equity"]) == float(equity_filled.iloc[-1])
