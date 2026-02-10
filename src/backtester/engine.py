from __future__ import annotations

import pandas as pd


def run_backtest(
    df: pd.DataFrame,
    desired_position: pd.Series,
    commission_bps: float = 2.0,
    slippage_bps: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    bt = df.copy()
    bt["desired_pos"] = desired_position.reindex(bt.index).fillna(0)
    bt["position"] = bt["desired_pos"].shift(1).fillna(0)
    bt["open_ret_next"] = bt["Open"].shift(-1) / bt["Open"] - 1

    turn = (bt["position"] - bt["position"].shift(1).fillna(0)).abs()
    cost = turn * ((commission_bps + slippage_bps) / 10000)
    bt["strategy_ret"] = bt["position"] * bt["open_ret_next"].fillna(0) - cost
    bt["equity"] = (1 + bt["strategy_ret"]).cumprod()

    total_days = max(len(bt), 1)
    years = total_days / 252
    cagr = bt["equity"].iloc[-1] ** (1 / years) - 1 if years > 0 else 0
    rolling_max = bt["equity"].cummax()
    drawdown = bt["equity"] / rolling_max - 1
    max_dd = abs(drawdown.min())
    trade_count = int((turn > 0).sum())

    penalty = 0.25 if trade_count < 20 else 0.0
    score = cagr - 0.5 * max_dd - penalty

    metrics = {
        "cagr": float(cagr),
        "max_drawdown": float(max_dd),
        "trades_count": trade_count,
        "score": float(score),
        "final_equity": float(bt["equity"].iloc[-1]),
    }
    return bt, metrics
