from __future__ import annotations

import logging

import numpy as np
import pandas as pd

_LOG = logging.getLogger("backtester.engine")


def _next_exec_price(bt: pd.DataFrame, i: int, execution_mode: str) -> float | None:
    if i + 1 >= len(bt):
        return None
    mode = (execution_mode or "next_open").lower()
    if mode == "next_close":
        px = float(bt["Close"].iloc[i + 1])
    else:
        px = float(bt["Open"].iloc[i + 1])
        if not np.isfinite(px) or px <= 0:
            px = float(bt["Close"].iloc[i + 1])
    if not np.isfinite(px) or px <= 0:
        return None
    return px


def run_backtest(
    df: pd.DataFrame,
    signal_df: pd.DataFrame,
    commission_bps: float,
    slippage_bps: float,
    initial_cash: float,
    position_size_pct: float,
    sl_pct: float = 0.01,
    tp_pct: float | None = None,
    execution_mode: str = "next_open",
    debug_diagnostics: bool = False,
) -> tuple[pd.DataFrame, dict, list[dict]]:
    if not (0 < float(position_size_pct) <= 1):
        raise ValueError(f"position_size_pct must satisfy 0 < pct <= 1, got {position_size_pct}")
    if float(sl_pct) <= 0:
        raise ValueError(f"sl_pct must be >0, got {sl_pct}")

    min_tp_pct = float(sl_pct) * 3.0
    tp_pct = float(tp_pct) if tp_pct is not None else min_tp_pct
    tp_pct = max(tp_pct, min_tp_pct)

    bt = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    bt["signal"] = signal_df.get("signal", 0).reindex(bt.index).fillna(0).astype(int)
    bt["desired_position"] = signal_df.get("position", 0).reindex(bt.index).fillna(0).astype(int)
    bt["executed_position"] = bt["desired_position"].shift(1).fillna(0).astype(int)
    bt["entry_votes"] = signal_df.get("entry_votes", 0).reindex(bt.index).fillna(0).astype(int)
    bt["exit_votes"] = signal_df.get("exit_votes", 0).reindex(bt.index).fillna(0).astype(int)
    bt["entry_vote_components"] = signal_df.get("entry_vote_components", "").reindex(bt.index).fillna("")
    bt["exit_vote_components"] = signal_df.get("exit_vote_components", "").reindex(bt.index).fillna("")

    cost_rate = max(float(commission_bps) + float(slippage_bps), 0.0) / 10000.0
    cash = float(initial_cash)
    position_qty = 0.0
    entry_price = 0.0
    entry_alloc = 0.0
    entry_cost = 0.0
    entry_ts: pd.Timestamp | None = None
    entry_votes = 0
    entry_vote_components = ""

    pending_exit_reason = ""
    pending_exit_at = -1

    trades: list[dict] = []
    cash_values: list[float] = []
    qty_values: list[float] = []
    equity_values: list[float] = []

    entry_diag_count = 0
    exit_diag_count = 0

    for i, ts in enumerate(bt.index):
        open_price = float(bt["Open"].iloc[i])
        close_price = float(bt["Close"].iloc[i])

        target_pos = int(bt["executed_position"].iloc[i])
        exit_reason = "signal" if target_pos == 0 else ""
        if pending_exit_reason and i >= pending_exit_at:
            target_pos = 0
            exit_reason = pending_exit_reason

        # Exit first (if requested to flat)
        if target_pos == 0 and position_qty > 0.0 and open_price > 0:
            gross_exit = position_qty * open_price
            exit_cost = gross_exit * cost_rate
            cash += gross_exit - exit_cost
            pnl_abs = gross_exit - exit_cost - entry_alloc - entry_cost
            pnl_pct = pnl_abs / entry_alloc if entry_alloc > 0 else 0.0
            risk_abs = entry_alloc * float(sl_pct)
            r_mult = pnl_abs / risk_abs if risk_abs > 0 else 0.0
            holding_days = max((pd.Timestamp(ts) - pd.Timestamp(entry_ts)).days, 1) if entry_ts is not None else 0
            equity_after = cash

            trade = {
                "entry_date": str(entry_ts),
                "exit_date": str(ts),
                "entry_price": float(entry_price),
                "exit_price": float(open_price),
                "qty": float(position_qty),
                "pnl_$": float(pnl_abs),
                "pnl": float(pnl_pct),
                "holding_days": int(holding_days),
                "exit_reason": exit_reason,
                "R": float(r_mult),
                "entry_votes": int(entry_votes),
                "exit_votes": int(bt["exit_votes"].iloc[i]),
                "entry_vote_components": str(entry_vote_components),
                "exit_vote_components": str(bt["exit_vote_components"].iloc[i]),
                "equity_after": float(equity_after),
                "sl_pct": float(sl_pct),
                "tp_pct": float(tp_pct),
            }
            trades.append(trade)

            if debug_diagnostics and exit_diag_count < 2:
                _LOG.info(
                    "trade_diag_exit ts=%s reason=%s exit_price=%.6f cash_after_exit=%.2f votes=%s",
                    ts,
                    exit_reason,
                    open_price,
                    cash,
                    trade["exit_vote_components"],
                )
                exit_diag_count += 1

            position_qty = 0.0
            entry_price = 0.0
            entry_alloc = 0.0
            entry_cost = 0.0
            entry_ts = None
            entry_votes = 0
            entry_vote_components = ""
            pending_exit_reason = ""
            pending_exit_at = -1

        # Entry
        if target_pos == 1 and position_qty <= 0.0 and open_price > 0:
            alloc = max(cash * float(position_size_pct), 0.0)
            qty = alloc / open_price if alloc > 0 else 0.0
            gross_entry = qty * open_price
            this_entry_cost = gross_entry * cost_rate
            if gross_entry + this_entry_cost > cash and cash > 0:
                gross_entry = cash / (1.0 + cost_rate)
                qty = gross_entry / open_price
                this_entry_cost = cash - gross_entry

            cash -= gross_entry + this_entry_cost
            position_qty = qty
            entry_price = open_price
            entry_alloc = gross_entry
            entry_cost = this_entry_cost
            entry_ts = ts
            entry_votes = int(bt["entry_votes"].iloc[i])
            entry_vote_components = str(bt["entry_vote_components"].iloc[i])

            if debug_diagnostics and entry_diag_count < 2:
                _LOG.info(
                    "trade_diag_entry ts=%s entry_price=%.6f qty=%.6f alloc=%.2f cash_after_entry=%.2f votes=%s",
                    ts,
                    open_price,
                    qty,
                    gross_entry,
                    cash,
                    entry_vote_components,
                )
                entry_diag_count += 1

        # Schedule risk exits (known after bar closes, execute next bar)
        if position_qty > 0.0 and i < len(bt) - 1 and not pending_exit_reason:
            sl_level = entry_price * (1.0 - float(sl_pct))
            tp_level = entry_price * (1.0 + float(tp_pct))
            lo = float(bt["Low"].iloc[i])
            hi = float(bt["High"].iloc[i])
            if np.isfinite(lo) and lo <= sl_level:
                pending_exit_reason = "sl"
                pending_exit_at = i + 1
            elif np.isfinite(hi) and hi >= tp_level:
                pending_exit_reason = "tp"
                pending_exit_at = i + 1

        equity = cash + position_qty * close_price
        cash_values.append(float(cash))
        qty_values.append(float(position_qty))
        equity_values.append(float(equity))

        # finalize last bar
        if i == len(bt) - 1 and position_qty > 0.0 and close_price > 0:
            gross_exit = position_qty * close_price
            exit_cost = gross_exit * cost_rate
            cash += gross_exit - exit_cost
            pnl_abs = gross_exit - exit_cost - entry_alloc - entry_cost
            pnl_pct = pnl_abs / entry_alloc if entry_alloc > 0 else 0.0
            risk_abs = entry_alloc * float(sl_pct)
            r_mult = pnl_abs / risk_abs if risk_abs > 0 else 0.0
            holding_days = max((pd.Timestamp(ts) - pd.Timestamp(entry_ts)).days, 1) if entry_ts is not None else 0
            reason = pending_exit_reason or "forced_eod"
            trades.append(
                {
                    "entry_date": str(entry_ts),
                    "exit_date": str(ts),
                    "entry_price": float(entry_price),
                    "exit_price": float(close_price),
                    "qty": float(position_qty),
                    "pnl_$": float(pnl_abs),
                    "pnl": float(pnl_pct),
                    "holding_days": int(holding_days),
                    "exit_reason": reason,
                    "R": float(r_mult),
                    "entry_votes": int(entry_votes),
                    "exit_votes": int(bt["exit_votes"].iloc[i]),
                    "entry_vote_components": str(entry_vote_components),
                    "exit_vote_components": str(bt["exit_vote_components"].iloc[i]),
                    "equity_after": float(cash),
                    "sl_pct": float(sl_pct),
                    "tp_pct": float(tp_pct),
                    "forced_exit": True,
                }
            )
            cash_values[-1] = float(cash)
            qty_values[-1] = 0.0
            equity_values[-1] = float(cash)
            position_qty = 0.0

    bt["cash"] = cash_values
    bt["qty"] = qty_values
    bt["equity"] = pd.Series(equity_values, index=bt.index).ffill().bfill().fillna(float(initial_cash))
    bt["strategy_ret"] = bt["equity"].pct_change().fillna(0.0)

    max_equity_jump = float(bt["equity"].diff().abs().max()) if len(bt) > 1 else 0.0
    if debug_diagnostics:
        _LOG.info("trade_diag_summary trades=%s max_equity_jump=%.2f", len(trades), max_equity_jump)

    metrics = _compute_metrics(bt, trades, initial_cash=float(initial_cash))
    return bt, metrics, trades


def _compute_metrics(bt: pd.DataFrame, trades: list[dict], initial_cash: float) -> dict:
    equity = bt["equity"].ffill().bfill().fillna(float(initial_cash))
    total_days = max(len(bt), 1)
    years = total_days / 252
    cagr = float((equity.iloc[-1] / initial_cash) ** (1 / years) - 1) if years > 0 and initial_cash > 0 else 0.0

    rolling_max = equity.cummax().replace(0, np.nan)
    dd = equity / rolling_max - 1
    max_dd = float(abs(dd.min())) if np.isfinite(dd.min()) else 0.0

    ret_std = bt["strategy_ret"].std(ddof=0)
    sharpe = float((bt["strategy_ret"].mean() / ret_std) * np.sqrt(252)) if ret_std and ret_std > 0 else 0.0

    trades_count = len(trades)
    win_rate = float(sum(1 for t in trades if float(t.get("pnl", 0.0)) > 0) / trades_count) if trades_count else 0.0

    pnl_values = [float(t.get("pnl_$", 0.0)) for t in trades]
    r_values = [float(t.get("R", 0.0)) for t in trades]
    hold_values = [int(t.get("holding_days", 0)) for t in trades]
    gross_profit = sum(p for p in pnl_values if p > 0)
    gross_loss_abs = abs(sum(p for p in pnl_values if p < 0))
    profit_factor = float(gross_profit / gross_loss_abs) if gross_loss_abs > 0 else (999.0 if gross_profit > 0 else 0.0)
    expectancy = float(np.mean(pnl_values)) if pnl_values else 0.0

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
        "avg_R": float(np.mean(r_values)) if r_values else 0.0,
        "median_R": float(np.median(r_values)) if r_values else 0.0,
        "best_R": float(np.max(r_values)) if r_values else 0.0,
        "worst_R": float(np.min(r_values)) if r_values else 0.0,
        "profit_factor": float(profit_factor),
        "expectancy": float(expectancy),
        "holding_days_avg": float(np.mean(hold_values)) if hold_values else 0.0,
        "holding_days_median": float(np.median(hold_values)) if hold_values else 0.0,
    }
