from __future__ import annotations

import logging

import numpy as np
import pandas as pd

_LOG = logging.getLogger("backtester.engine")


def run_backtest(
    df: pd.DataFrame,
    signal_df: pd.DataFrame,
    commission_bps: float,
    slippage_bps: float,
    initial_cash: float,
    position_size_pct: float,
) -> tuple[pd.DataFrame, dict, list[dict]]:
    if not (0 < float(position_size_pct) <= 1):
        raise ValueError(f"position_size_pct must satisfy 0 < pct <= 1, got {position_size_pct}")

    bt = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    bt["signal"] = signal_df["signal"].reindex(bt.index).fillna(0).astype(int)
    bt["desired_position"] = signal_df["position"].reindex(bt.index).fillna(0).astype(int)
    bt["executed_position"] = bt["desired_position"].shift(1).fillna(0).astype(int)

    cost_rate = max(float(commission_bps) + float(slippage_bps), 0.0) / 10000.0
    cash = float(initial_cash)
    position_qty = 0.0
    position_entry_price = 0.0
    position_entry_alloc = 0.0
    position_entry_cost = 0.0
    entry_ts: pd.Timestamp | None = None

    trades: list[dict] = []
    cash_values: list[float] = []
    qty_values: list[float] = []
    equity_values: list[float] = []

    entry_diag_count = 0
    exit_diag_count = 0

    for i, (ts, row) in enumerate(bt.iterrows()):
        open_price = float(row["Open"])
        close_price = float(row["Close"])
        target_pos = int(row["executed_position"])

        # Entry (no pyramiding)
        if target_pos == 1 and position_qty <= 0.0 and open_price > 0:
            alloc = max(cash * float(position_size_pct), 0.0)
            qty = alloc / open_price if alloc > 0 else 0.0
            gross_entry = qty * open_price
            entry_cost = gross_entry * cost_rate

            if gross_entry + entry_cost > cash and cash > 0:
                gross_entry = cash / (1.0 + cost_rate)
                qty = gross_entry / open_price
                entry_cost = cash - gross_entry
                alloc = gross_entry

            cash -= gross_entry + entry_cost
            position_qty = qty
            position_entry_price = open_price
            position_entry_alloc = gross_entry
            position_entry_cost = entry_cost
            entry_ts = ts

            if entry_diag_count < 2:
                _LOG.info(
                    "trade_diag_entry ts=%s entry_price=%.6f qty=%.6f alloc=%.2f cash_after_entry=%.2f",
                    ts,
                    open_price,
                    qty,
                    gross_entry,
                    cash,
                )
                entry_diag_count += 1

        # Exit
        if target_pos == 0 and position_qty > 0.0 and open_price > 0:
            gross_exit = position_qty * open_price
            exit_cost = gross_exit * cost_rate
            cash += gross_exit - exit_cost

            pnl_abs = gross_exit - exit_cost - position_entry_alloc - position_entry_cost
            pnl_pct = pnl_abs / position_entry_alloc if position_entry_alloc > 0 else 0.0
            trades.append(
                {
                    "entry_date": str(entry_ts),
                    "exit_date": str(ts),
                    "entry_price": float(position_entry_price),
                    "exit_price": float(open_price),
                    "qty": float(position_qty),
                    "pnl_$": float(pnl_abs),
                    "pnl": float(pnl_pct),
                }
            )

            if exit_diag_count < 2:
                _LOG.info(
                    "trade_diag_exit ts=%s exit_price=%.6f cash_after_exit=%.2f",
                    ts,
                    open_price,
                    cash,
                )
                exit_diag_count += 1

            position_qty = 0.0
            position_entry_price = 0.0
            position_entry_alloc = 0.0
            position_entry_cost = 0.0
            entry_ts = None

        equity = cash + position_qty * close_price
        cash_values.append(float(cash))
        qty_values.append(float(position_qty))
        equity_values.append(float(equity))

        # Finalize on last bar
        if i == len(bt) - 1 and position_qty > 0.0 and close_price > 0:
            gross_exit = position_qty * close_price
            exit_cost = gross_exit * cost_rate
            cash += gross_exit - exit_cost

            pnl_abs = gross_exit - exit_cost - position_entry_alloc - position_entry_cost
            pnl_pct = pnl_abs / position_entry_alloc if position_entry_alloc > 0 else 0.0
            trades.append(
                {
                    "entry_date": str(entry_ts),
                    "exit_date": str(ts),
                    "entry_price": float(position_entry_price),
                    "exit_price": float(close_price),
                    "qty": float(position_qty),
                    "pnl_$": float(pnl_abs),
                    "pnl": float(pnl_pct),
                    "forced_exit": True,
                }
            )

            if exit_diag_count < 2:
                _LOG.info(
                    "trade_diag_exit ts=%s exit_price=%.6f cash_after_exit=%.2f",
                    ts,
                    close_price,
                    cash,
                )

            position_qty = 0.0
            position_entry_price = 0.0
            position_entry_alloc = 0.0
            position_entry_cost = 0.0
            entry_ts = None

            cash_values[-1] = float(cash)
            qty_values[-1] = 0.0
            equity_values[-1] = float(cash)

    bt["cash"] = cash_values
    bt["qty"] = qty_values
    bt["equity"] = pd.Series(equity_values, index=bt.index).ffill().bfill().fillna(float(initial_cash))
    bt["strategy_ret"] = bt["equity"].pct_change().fillna(0.0)

    if len(bt) > 1:
        max_equity_jump = float(bt["equity"].diff().abs().max())
    else:
        max_equity_jump = 0.0
    _LOG.info("trade_diag_summary trades=%s max_equity_jump=%.2f", len(trades), max_equity_jump)

    metrics = _compute_metrics(bt, trades, initial_cash=float(initial_cash))
    return bt, metrics, trades


def _compute_metrics(bt: pd.DataFrame, trades: list[dict], initial_cash: float) -> dict:
    equity = bt["equity"].ffill().bfill().fillna(float(initial_cash))
    total_days = max(len(bt), 1)
    years = total_days / 252
    if years > 0 and equity.iloc[-1] > 0 and initial_cash > 0:
        cagr = float((equity.iloc[-1] / initial_cash) ** (1 / years) - 1)
    else:
        cagr = 0.0

    rolling_max = equity.cummax().replace(0, np.nan)
    dd = equity / rolling_max - 1
    max_dd = float(abs(dd.min())) if np.isfinite(dd.min()) else 0.0

    ret_std = bt["strategy_ret"].std(ddof=0)
    sharpe = float((bt["strategy_ret"].mean() / ret_std) * np.sqrt(252)) if ret_std and ret_std > 0 else 0.0

    trades_count = len(trades)
    win_rate = float(sum(1 for t in trades if float(t.get("pnl", 0.0)) > 0) / trades_count) if trades_count else 0.0

    final_equity = float(equity.iloc[-1]) if np.isfinite(equity.iloc[-1]) else float(initial_cash)
    profit_abs = final_equity - float(initial_cash)
    profit_pct = (profit_abs / float(initial_cash)) if initial_cash > 0 else 0.0

    penalty = 2.0 if trades_count < 20 else 0.0
    score = cagr - 0.5 * max_dd - penalty
    if not np.isfinite(score):
        score = -9999.0

    return {
        "score": float(score),
        "cagr": float(cagr if np.isfinite(cagr) else 0.0),
        "max_dd": float(max_dd if np.isfinite(max_dd) else 0.0),
        "max_dd_%": float((max_dd * 100.0) if np.isfinite(max_dd) else 0.0),
        "sharpe": float(sharpe if np.isfinite(sharpe) else 0.0),
        "trades_count": trades_count,
        "win_rate": float(win_rate if np.isfinite(win_rate) else 0.0),
        "final_equity": final_equity,
        "profit_$": float(profit_abs),
        "profit_%": float(profit_pct * 100.0),
    }
