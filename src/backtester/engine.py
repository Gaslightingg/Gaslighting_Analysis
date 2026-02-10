from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(
    df: pd.DataFrame,
    signal_df: pd.DataFrame,
    commission_bps: float,
    slippage_bps: float,
) -> tuple[pd.DataFrame, dict, list[dict]]:
    bt = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    bt["signal"] = signal_df["signal"].reindex(bt.index).fillna(0)
    bt["desired_position"] = signal_df["position"].reindex(bt.index).fillna(0)

    bt["executed_position"] = bt["desired_position"].shift(1).fillna(0)
    bt["ret_open_to_open"] = bt["Open"].shift(-1) / bt["Open"] - 1

    turnover = bt["executed_position"].diff().abs().fillna(0)
    cost = turnover * ((commission_bps + slippage_bps) / 10000)

    bt["strategy_ret"] = bt["executed_position"] * bt["ret_open_to_open"].fillna(0) - cost
    bt["equity"] = (1 + bt["strategy_ret"]).cumprod()

    trades = _extract_trades(bt)
    metrics = _compute_metrics(bt, trades)
    return bt, metrics, trades


def _extract_trades(bt: pd.DataFrame) -> list[dict]:
    trades: list[dict] = []
    in_pos = False
    entry_date = None
    entry_price = None

    idx = bt.index.tolist()
    for i in range(1, len(bt) - 1):
        prev_pos = int(bt["executed_position"].iloc[i - 1])
        pos = int(bt["executed_position"].iloc[i])
        if not in_pos and prev_pos == 0 and pos == 1:
            in_pos = True
            entry_date = idx[i]
            entry_price = float(bt["Open"].iloc[i])
        elif in_pos and prev_pos == 1 and pos == 0:
            exit_date = idx[i]
            exit_price = float(bt["Open"].iloc[i])
            pnl = (exit_price / entry_price) - 1 if entry_price else 0.0
            trades.append(
                {
                    "entry_date": str(entry_date),
                    "exit_date": str(exit_date),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                }
            )
            in_pos = False
            entry_date = None
            entry_price = None
    return trades


def _compute_metrics(bt: pd.DataFrame, trades: list[dict]) -> dict:
    equity = bt["equity"].ffill().fillna(1.0)
    total_days = max(len(bt), 1)
    years = total_days / 252
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0

    rolling_max = equity.cummax()
    dd = equity / rolling_max - 1
    max_dd = float(abs(dd.min())) if np.isfinite(dd.min()) else 0.0

    ret_std = bt["strategy_ret"].std(ddof=0)
    sharpe = float((bt["strategy_ret"].mean() / ret_std) * np.sqrt(252)) if ret_std and ret_std > 0 else 0.0

    trades_count = len(trades)
    win_rate = float(sum(1 for t in trades if t["pnl"] > 0) / trades_count) if trades_count else 0.0

    penalty = 2.0 if trades_count < 20 else 0.0
    score = cagr - 0.5 * max_dd - penalty
    if not np.isfinite(score):
        score = -9999.0

    return {
        "score": float(score),
        "cagr": float(cagr if np.isfinite(cagr) else 0.0),
        "max_dd": float(max_dd if np.isfinite(max_dd) else 0.0),
        "sharpe": float(sharpe if np.isfinite(sharpe) else 0.0),
        "trades_count": trades_count,
        "win_rate": float(win_rate if np.isfinite(win_rate) else 0.0),
        "final_equity": float(equity.iloc[-1] if np.isfinite(equity.iloc[-1]) else 1.0),
    }
