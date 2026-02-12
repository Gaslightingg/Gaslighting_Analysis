from __future__ import annotations

import logging

import numpy as np
import pandas as pd

_LOG = logging.getLogger("backtester.engine")

STATE_FLAT = 0
STATE_LONG = 1
STATE_SHORT = -1


def _series_or_default(signal_df: pd.DataFrame, key: str, index: pd.Index, default: int | str = 0) -> pd.Series:
    if key in signal_df.columns:
        return signal_df[key].reindex(index)
    return pd.Series(default, index=index)


def _build_desired_position(index: pd.Index, signal_df: pd.DataFrame, allow_short: bool = True) -> pd.Series:
    if "position" in signal_df.columns:
        out = signal_df["position"].reindex(index).fillna(0).clip(-1, 1).astype(int)
        if not allow_short:
            out = out.clip(lower=0)
        return out

    signal = _series_or_default(signal_df, "signal", index, 0).fillna(0).astype(int)
    enter_long = _series_or_default(signal_df, "enter_long", index, 0).fillna(0).astype(int)
    exit_long = _series_or_default(signal_df, "exit_long", index, 0).fillna(0).astype(int)
    enter_short = _series_or_default(signal_df, "enter_short", index, 0).fillna(0).astype(int)
    exit_short = _series_or_default(signal_df, "exit_short", index, 0).fillna(0).astype(int)

    out = pd.Series(0, index=index, dtype=int)
    state = STATE_FLAT
    for i in range(len(index)):
        if i == 0:
            out.iloc[i] = state
            continue

        sig = int(signal.iloc[i])
        want_enter_long = bool(enter_long.iloc[i] == 1 or sig == 1)
        want_enter_short = bool(enter_short.iloc[i] == 1 or sig == -1)
        want_exit_long = bool(exit_long.iloc[i] == 1 or sig == 2)
        want_exit_short = bool(exit_short.iloc[i] == 1 or sig == -2)

        if state == STATE_FLAT:
            if want_enter_long and not want_enter_short:
                state = STATE_LONG
            elif allow_short and want_enter_short and not want_enter_long:
                state = STATE_SHORT
        elif state == STATE_LONG:
            if allow_short and want_enter_short:
                state = STATE_SHORT
            elif want_exit_long:
                state = STATE_FLAT
        elif state == STATE_SHORT:
            if want_enter_long:
                state = STATE_LONG
            elif want_exit_short:
                state = STATE_FLAT

        out.iloc[i] = state
    return out.astype(int)


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
    allow_short: bool = True,
) -> tuple[pd.DataFrame, dict, list[dict]]:
    if not (0 < float(position_size_pct) <= 1):
        raise ValueError(f"position_size_pct must satisfy 0 < pct <= 1, got {position_size_pct}")
    if float(sl_pct) <= 0:
        raise ValueError(f"sl_pct must be >0, got {sl_pct}")

    min_tp_pct = float(sl_pct) * 3.0
    tp_pct = float(tp_pct) if tp_pct is not None else min_tp_pct
    tp_pct = max(tp_pct, min_tp_pct)

    bt = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    bt["signal"] = _series_or_default(signal_df, "signal", bt.index, 0).fillna(0).astype(int)
    bt["desired_position"] = _build_desired_position(bt.index, signal_df, allow_short=bool(allow_short))
    bt["executed_position"] = bt["desired_position"].shift(1).fillna(0).astype(int)
    bt["entry_votes"] = _series_or_default(signal_df, "entry_votes", bt.index, 0).fillna(0).astype(int)
    bt["exit_votes"] = _series_or_default(signal_df, "exit_votes", bt.index, 0).fillna(0).astype(int)
    bt["entry_ok"] = _series_or_default(signal_df, "entry_ok", bt.index, 0).fillna(0).astype(int)
    bt["exit_ok"] = _series_or_default(signal_df, "exit_ok", bt.index, 0).fillna(0).astype(int)
    bt["enter_long"] = _series_or_default(signal_df, "enter_long", bt.index, 0).fillna(0).astype(int)
    bt["exit_long"] = _series_or_default(signal_df, "exit_long", bt.index, 0).fillna(0).astype(int)
    bt["enter_short"] = _series_or_default(signal_df, "enter_short", bt.index, 0).fillna(0).astype(int)
    bt["exit_short"] = _series_or_default(signal_df, "exit_short", bt.index, 0).fillna(0).astype(int)
    bt["entry_vote_components"] = _series_or_default(signal_df, "entry_vote_components", bt.index, "").fillna("").astype(str)
    bt["exit_vote_components"] = _series_or_default(signal_df, "exit_vote_components", bt.index, "").fillna("").astype(str)

    cost_rate = max(float(commission_bps) + float(slippage_bps), 0.0) / 10000.0
    cash = float(initial_cash)

    pos_state = STATE_FLAT
    position_qty = 0.0
    entry_price = 0.0
    entry_notional = 0.0
    entry_cost = 0.0
    entry_ts: pd.Timestamp | None = None
    entry_side = STATE_FLAT
    entry_votes = 0
    entry_vote_components = ""
    entry_index = -1

    pending_exit_reason = ""
    pending_exit_at = -1

    trades: list[dict] = []
    fills: list[dict] = []
    cash_values: list[float] = []
    qty_values: list[float] = []
    equity_values: list[float] = []

    enter_count = 0
    exit_count = 0
    forced_exit_count = 0
    hold_bars_total = 0
    hold_bars_values: list[int] = []
    exits_by_rule = 0
    exits_by_sl_tp = 0
    exits_forced_end = 0
    exits_flip = 0
    trade_seq = 0
    session_seq = 0
    current_session_id = 0
    sessions_opened = 0
    sessions_closed = 0

    def _close_position(ts: pd.Timestamp, px: float, reason: str, forced: bool = False) -> None:
        nonlocal cash, pos_state, position_qty, entry_price, entry_notional, entry_cost
        nonlocal entry_ts, entry_side, entry_votes, entry_vote_components, entry_index
        nonlocal exit_count, forced_exit_count, hold_bars_total, exits_by_rule, exits_by_sl_tp, exits_forced_end, exits_flip

        if position_qty <= 0.0 or entry_side == STATE_FLAT or px <= 0:
            return

        exit_notional = position_qty * px
        exit_cost = exit_notional * cost_rate
        if entry_side == STATE_LONG:
            entry_cash_delta = -(entry_notional + entry_cost)
            exit_cash_delta = +(exit_notional - exit_cost)
        else:  # short
            entry_cash_delta = +(entry_notional - entry_cost)
            exit_cash_delta = -(exit_notional + exit_cost)

        cash += exit_cash_delta
        pnl_abs = entry_cash_delta + exit_cash_delta
        pnl_pct = pnl_abs / entry_notional if entry_notional > 0 else 0.0
        risk_abs = entry_notional * float(sl_pct)
        r_mult = pnl_abs / risk_abs if risk_abs > 0 else 0.0
        holding_days = max((pd.Timestamp(ts) - pd.Timestamp(entry_ts)).days, 1) if entry_ts is not None else 0
        hold_bars = max((bt.index.get_loc(ts) - entry_index), 1) if entry_index >= 0 else 1

        nonlocal trade_seq, sessions_closed, current_session_id

        fills.append(
            {
                "fill_id": len(fills) + 1,
                "fill_type": "close",
                "side": "SELL" if entry_side == STATE_LONG else "BUY",
                "qty": float(position_qty),
                "price": float(px),
                "dt": str(ts),
                "session_id": int(current_session_id),
            }
        )

        trade_seq += 1
        trade = {
            "trade_id": int(trade_seq),
            "session_id": int(current_session_id),
            "side": "long" if entry_side == STATE_LONG else "short",
            "entry_date": str(entry_ts),
            "exit_date": str(ts),
            "entry_price": float(entry_price),
            "exit_price": float(px),
            "qty": float(position_qty),
            "pnl_$": float(pnl_abs),
            "pnl": float(pnl_pct),
            "holding_days": int(holding_days),
            "holding_bars": int(hold_bars),
            "exit_reason": reason,
            "R": float(r_mult),
            "entry_votes": int(entry_votes),
            "exit_votes": int(bt.loc[ts, "exit_votes"]),
            "entry_vote_components": str(entry_vote_components),
            "exit_vote_components": str(bt.loc[ts, "exit_vote_components"]),
            "equity_after": float(cash),
            "sl_pct": float(sl_pct),
            "tp_pct": float(tp_pct),
            "forced_exit": bool(forced),
            "entry_fills": [
                {
                    "side": "BUY" if entry_side == STATE_LONG else "SELL",
                    "qty": float(position_qty),
                    "price": float(entry_price),
                    "dt": str(entry_ts),
                }
            ],
            "exit_fills": [
                {
                    "side": "SELL" if entry_side == STATE_LONG else "BUY",
                    "qty": float(position_qty),
                    "price": float(px),
                    "dt": str(ts),
                }
            ],
        }
        trades.append(trade)
        sessions_closed += 1
        exit_count += 1
        if forced:
            forced_exit_count += 1
            exits_forced_end += 1
        elif reason in {"sl", "tp"}:
            exits_by_sl_tp += 1
        elif reason == "flip":
            exits_flip += 1
        else:
            exits_by_rule += 1
        hold_bars_total += int(hold_bars)
        hold_bars_values.append(int(hold_bars))

        pos_state = STATE_FLAT
        position_qty = 0.0
        entry_price = 0.0
        entry_notional = 0.0
        entry_cost = 0.0
        entry_ts = None
        entry_side = STATE_FLAT
        entry_votes = 0
        entry_vote_components = ""
        entry_index = -1
        pending_exit_reason = ""

    def _open_position(ts: pd.Timestamp, side: int, px: float) -> None:
        nonlocal cash, pos_state, position_qty, entry_price, entry_notional, entry_cost
        nonlocal entry_ts, entry_side, entry_votes, entry_vote_components, entry_index, enter_count
        nonlocal session_seq, current_session_id, sessions_opened
        if side not in {STATE_LONG, STATE_SHORT} or px <= 0 or pos_state != STATE_FLAT:
            return

        alloc_base = cash if side == STATE_LONG else max(cash, float(initial_cash))
        alloc = max(float(alloc_base) * float(position_size_pct), 0.0)
        qty = alloc / px if alloc > 0 else 0.0
        if qty <= 0.0:
            return

        notional = qty * px
        this_entry_cost = notional * cost_rate

        if side == STATE_LONG and notional + this_entry_cost > cash and cash > 0:
            notional = cash / (1.0 + cost_rate)
            qty = notional / px
            this_entry_cost = cash - notional

        if side == STATE_LONG:
            cash -= (notional + this_entry_cost)
        else:
            cash += (notional - this_entry_cost)

        session_seq += 1
        current_session_id = session_seq
        sessions_opened += 1

        pos_state = side
        position_qty = qty
        entry_price = px
        entry_notional = notional
        entry_cost = this_entry_cost
        entry_ts = ts
        entry_side = side
        entry_votes = int(bt.loc[ts, "entry_votes"]) if side == STATE_LONG else int(bt.loc[ts, "exit_votes"])
        entry_vote_components = str(bt.loc[ts, "entry_vote_components"] if side == STATE_LONG else bt.loc[ts, "exit_vote_components"])
        entry_index = bt.index.get_loc(ts)
        enter_count += 1
        fills.append(
            {
                "fill_id": len(fills) + 1,
                "fill_type": "open",
                "side": "BUY" if side == STATE_LONG else "SELL",
                "qty": float(qty),
                "price": float(px),
                "dt": str(ts),
                "session_id": int(current_session_id),
            }
        )

    for i, ts in enumerate(bt.index):
        open_price = float(bt["Open"].iloc[i])
        close_price = float(bt["Close"].iloc[i])

        target_pos = int(bt["executed_position"].iloc[i])

        if pending_exit_reason and i >= pending_exit_at and pos_state != STATE_FLAT:
            target_pos = STATE_FLAT

        # State machine: exit first if target changed, then optional entry (flip allowed).
        if pos_state != STATE_FLAT and target_pos != pos_state and open_price > 0:
            close_reason = pending_exit_reason or ("flip" if target_pos in {STATE_LONG, STATE_SHORT} else "signal")
            _close_position(ts, open_price, close_reason, forced=False)
            pending_exit_reason = ""
            pending_exit_at = -1

        if pos_state == STATE_FLAT and target_pos in {STATE_LONG, STATE_SHORT} and open_price > 0:
            _open_position(ts, target_pos, open_price)

        # Schedule risk exits (known after bar closes, execute next bar)
        if pos_state != STATE_FLAT and i < len(bt) - 1 and not pending_exit_reason:
            lo = float(bt["Low"].iloc[i])
            hi = float(bt["High"].iloc[i])
            if entry_side == STATE_LONG:
                sl_level = entry_price * (1.0 - float(sl_pct))
                tp_level = entry_price * (1.0 + float(tp_pct))
                if np.isfinite(lo) and lo <= sl_level:
                    pending_exit_reason = "sl"
                    pending_exit_at = i + 1
                elif np.isfinite(hi) and hi >= tp_level:
                    pending_exit_reason = "tp"
                    pending_exit_at = i + 1
            else:  # short
                sl_level = entry_price * (1.0 + float(sl_pct))
                tp_level = entry_price * (1.0 - float(tp_pct))
                if np.isfinite(hi) and hi >= sl_level:
                    pending_exit_reason = "sl"
                    pending_exit_at = i + 1
                elif np.isfinite(lo) and lo <= tp_level:
                    pending_exit_reason = "tp"
                    pending_exit_at = i + 1

        equity = cash + (float(pos_state) * position_qty * close_price)
        cash_values.append(float(cash))
        qty_values.append(float(pos_state * position_qty))
        equity_values.append(float(equity))

        # forced close at end-of-test should be counted as normal exit event
        if i == len(bt) - 1 and pos_state != STATE_FLAT and close_price > 0:
            _close_position(ts, close_price, pending_exit_reason or "forced_eod", forced=True)
            cash_values[-1] = float(cash)
            qty_values[-1] = 0.0
            equity_values[-1] = float(cash)

    bt["cash"] = cash_values
    bt["qty"] = qty_values
    bt["equity"] = pd.Series(equity_values, index=bt.index).ffill().bfill().fillna(float(initial_cash))
    bt["strategy_ret"] = bt["equity"].pct_change().fillna(0.0)

    exposure = float((bt["qty"] != 0).sum() / len(bt)) if len(bt) else 0.0

    metrics = _compute_metrics(
        bt,
        trades,
        initial_cash=float(initial_cash),
        enter_count=int(enter_count),
        exit_count=int(exit_count),
        forced_exit_count=int(forced_exit_count),
        open_position=int(pos_state),
        open_position_qty=float(pos_state * position_qty),
        exposure=float(exposure),
        avg_hold_bars=float(np.mean(hold_bars_values)) if hold_bars_values else 0.0,
        exits_by_rule=int(exits_by_rule),
        exits_by_sl_tp=int(exits_by_sl_tp),
        exits_forced_end=int(exits_forced_end),
        exits_flip=int(exits_flip),
        fills=fills,
        sessions_opened=int(sessions_opened),
        sessions_closed=int(sessions_closed),
        allow_short=bool(allow_short),
    )

    if debug_diagnostics:
        _LOG.info(
            "trade_diag_summary enter_count=%s exit_count=%s trades=%s forced_exit=%s exits_by_rule=%s exits_by_sl_tp=%s exits_forced_end=%s exits_flip=%s open_position=%s",
            enter_count,
            exit_count,
            len(trades),
            forced_exit_count,
            exits_by_rule,
            exits_by_sl_tp,
            exits_forced_end,
            exits_flip,
            pos_state,
        )

    return bt, metrics, trades


def _compute_metrics(
    bt: pd.DataFrame,
    trades: list[dict],
    initial_cash: float,
    enter_count: int = 0,
    exit_count: int = 0,
    forced_exit_count: int = 0,
    open_position: int = 0,
    open_position_qty: float = 0.0,
    exposure: float = 0.0,
    avg_hold_bars: float = 0.0,
    exits_by_rule: int = 0,
    exits_by_sl_tp: int = 0,
    exits_forced_end: int = 0,
    exits_flip: int = 0,
    fills: list[dict] | None = None,
    sessions_opened: int = 0,
    sessions_closed: int = 0,
    allow_short: bool = True,
) -> dict:
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

    fills = fills or []
    fills_open = int(sum(1 for f in fills if str(f.get("fill_type")) == "open"))
    fills_close = int(sum(1 for f in fills if str(f.get("fill_type")) == "close"))
    short_trades = int(sum(1 for t in trades if str(t.get("side", "")).lower() == "short"))

    if exit_count > enter_count:
        _LOG.warning("invariant_violation exit_count_gt_enter_count enter=%s exit=%s", enter_count, exit_count)
    if forced_exit_count > exit_count:
        _LOG.warning("invariant_violation forced_exit_gt_exit forced=%s exit=%s", forced_exit_count, exit_count)

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
        "enter_count": int(enter_count),
        "exit_count": int(exit_count),
        "closed_trades_count": int(trades_count),
        "open_position": int(open_position),
        "open_position_qty": float(open_position_qty),
        "forced_exit_count": int(forced_exit_count),
        "exposure": float(exposure),
        "avg_hold_bars": float(avg_hold_bars),
        "entry_events_count": int(enter_count),
        "exit_events_count": int(exit_count),
        "open_positions_count": int(1 if open_position != 0 else 0),
        "fills_open": int(fills_open),
        "fills_close": int(fills_close),
        "trades_closed": int(trades_count),
        "position_sessions_opened": int(sessions_opened),
        "position_sessions_closed": int(sessions_closed),
        "short_trades_count": int(short_trades),
        "allow_short": bool(allow_short),
        "exits_by_rule": int(exits_by_rule),
        "exits_by_sl_tp": int(exits_by_sl_tp),
        "exits_forced_end": int(exits_forced_end),
        "exits_flip": int(exits_flip),
    }
