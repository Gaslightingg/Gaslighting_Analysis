from __future__ import annotations

import numpy as np
import pandas as pd


def run_backtest(
    df: pd.DataFrame,
    signal_df: pd.DataFrame,
    commission_bps: float,
    slippage_bps: float,
    initial_cash: float,
    position_size_pct: float,
) -> tuple[pd.DataFrame, dict, list[dict]]:
    bt = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    bt["signal"] = signal_df["signal"].reindex(bt.index).fillna(0).astype(int)
    bt["desired_position"] = signal_df["position"].reindex(bt.index).fillna(0).astype(int)
    bt["executed_position"] = bt["desired_position"].shift(1).fillna(0).astype(int)

    cost_rate = (commission_bps + slippage_bps) / 10000.0
    cash = float(initial_cash)
    qty = 0.0
    in_pos = False
    entry_date: pd.Timestamp | None = None
    entry_price = 0.0
    entry_qty = 0.0
    entry_cash_used = 0.0

    trades: list[dict] = []
    equity_values: list[float] = []
    cash_values: list[float] = []
    qty_values: list[float] = []

    for i, (ts, row) in enumerate(bt.iterrows()):
        open_price = float(row["Open"])
        close_price = float(row["Close"])
        target_pos = int(row["executed_position"])

        if target_pos == 1 and qty <= 0.0:
            alloc_cash = max(cash * float(position_size_pct), 0.0)
            if alloc_cash > 0.0 and open_price > 0.0:
                buy_qty = alloc_cash / open_price
                entry_cost = alloc_cash * cost_rate
                gross_cash_spent = alloc_cash + entry_cost
                if gross_cash_spent > cash:
                    gross_cash_spent = cash
                    alloc_cash = gross_cash_spent / (1.0 + cost_rate)
                    buy_qty = alloc_cash / open_price if open_price > 0 else 0.0
                    entry_cost = gross_cash_spent - alloc_cash

                cash -= gross_cash_spent
                qty = buy_qty
                in_pos = qty > 0.0
                if in_pos:
                    entry_date = ts
                    entry_price = open_price
                    entry_qty = qty
                    entry_cash_used = alloc_cash

        if target_pos == 0 and qty > 0.0:
            proceeds = qty * open_price
            exit_cost = proceeds * cost_rate
            cash += proceeds - exit_cost

            pnl_abs = (open_price - entry_price) * qty - (entry_cash_used * cost_rate) - exit_cost
            pnl_pct = (pnl_abs / entry_cash_used) if entry_cash_used > 0 else 0.0
            trades.append(
                {
                    "entry_date": str(entry_date),
                    "exit_date": str(ts),
                    "entry_price": float(entry_price),
                    "exit_price": float(open_price),
                    "qty": float(entry_qty),
                    "pnl_$": float(pnl_abs),
                    "pnl": float(pnl_pct),
                }
            )
            qty = 0.0
            in_pos = False
            entry_date = None
            entry_price = 0.0
            entry_qty = 0.0
            entry_cash_used = 0.0

        equity = cash + (qty * close_price)
        cash_values.append(float(cash))
        qty_values.append(float(qty))
        equity_values.append(float(equity))

        if i == len(bt) - 1 and in_pos and qty > 0.0:
            proceeds = qty * close_price
            exit_cost = proceeds * cost_rate
            cash += proceeds - exit_cost
            pnl_abs = (close_price - entry_price) * qty - (entry_cash_used * cost_rate) - exit_cost
            pnl_pct = (pnl_abs / entry_cash_used) if entry_cash_used > 0 else 0.0
            trades.append(
                {
                    "entry_date": str(entry_date),
                    "exit_date": str(ts),
                    "entry_price": float(entry_price),
                    "exit_price": float(close_price),
                    "qty": float(entry_qty),
                    "pnl_$": float(pnl_abs),
                    "pnl": float(pnl_pct),
                    "forced_exit": True,
                }
            )
            qty = 0.0
            in_pos = False
            entry_date = None
            entry_price = 0.0
            entry_qty = 0.0
            entry_cash_used = 0.0
            equity_values[-1] = float(cash)
            cash_values[-1] = float(cash)
            qty_values[-1] = 0.0

    bt["cash"] = cash_values
    bt["qty"] = qty_values
    bt["equity"] = pd.Series(equity_values, index=bt.index).ffill().bfill().fillna(float(initial_cash))
    bt["strategy_ret"] = bt["equity"].pct_change().fillna(0.0)

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
