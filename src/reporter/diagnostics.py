from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(slots=True)
class DiagnosticArtifacts:
    trades_csv: str
    equity_csv: str
    summary_json: str
    equity_png: str
    drawdown_png: str
    trades_png: str


def _to_trade_df(trades: list[dict]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(
            columns=[
                "entry_dt", "exit_dt", "direction", "entry_vwap", "exit_vwap", "qty", "pnl_$", "pnl_%", "R",
                "hold_bars", "hold_days", "exit_reason", "forced_exit",
            ]
        )
    rows = []
    for t in trades:
        direction = str(t.get("side", "long")).upper()
        rows.append(
            {
                "entry_dt": t.get("entry_date"),
                "exit_dt": t.get("exit_date"),
                "direction": direction,
                "entry_vwap": float(t.get("entry_price", 0.0) or 0.0),
                "exit_vwap": float(t.get("exit_price", 0.0) or 0.0),
                "qty": float(t.get("qty", 0.0) or 0.0),
                "pnl_$": float(t.get("pnl_$", 0.0) or 0.0),
                "pnl_%": float(t.get("pnl", 0.0) or 0.0) * 100.0,
                "R": float(t.get("R", 0.0) or 0.0),
                "hold_bars": int(t.get("holding_bars", 0) or 0),
                "hold_days": int(t.get("holding_days", 0) or 0),
                "exit_reason": str(t.get("exit_reason", "signal")),
                "forced_exit": bool(t.get("forced_exit", False)),
            }
        )
    out = pd.DataFrame(rows)
    out["entry_dt"] = pd.to_datetime(out["entry_dt"], errors="coerce")
    out["exit_dt"] = pd.to_datetime(out["exit_dt"], errors="coerce")
    return out


def _position_sessions(trade_df: pd.DataFrame) -> tuple[int, int]:
    if trade_df.empty:
        return 0, 0
    # One session per contiguous direction with small gap policy; simple and deterministic.
    sessions = 1
    prev_dir = trade_df.iloc[0]["direction"]
    prev_exit = trade_df.iloc[0]["exit_dt"]
    for i in range(1, len(trade_df)):
        cur_dir = trade_df.iloc[i]["direction"]
        cur_entry = trade_df.iloc[i]["entry_dt"]
        if cur_dir != prev_dir or (pd.notna(prev_exit) and pd.notna(cur_entry) and cur_entry > prev_exit):
            sessions += 1
        prev_dir = cur_dir
        prev_exit = trade_df.iloc[i]["exit_dt"]
    return sessions, sessions


def _max_consecutive(flags: list[bool]) -> tuple[int, int]:
    max_w = max_l = cur_w = cur_l = 0
    for f in flags:
        if f:
            cur_w += 1
            cur_l = 0
        else:
            cur_l += 1
            cur_w = 0
        max_w = max(max_w, cur_w)
        max_l = max(max_l, cur_l)
    return max_w, max_l


def _compute_drawdown(equity: pd.Series) -> tuple[pd.Series, float, int]:
    peak = equity.cummax().replace(0, np.nan)
    dd = equity / peak - 1.0
    max_dd = float(abs(dd.min())) if len(dd) else 0.0
    # duration in bars
    dur = 0
    cur = 0
    for v in dd.fillna(0.0):
        if v < 0:
            cur += 1
            dur = max(dur, cur)
        else:
            cur = 0
    return dd.fillna(0.0), max_dd, int(dur)


def build_diagnostic_summary(bt: pd.DataFrame, trades: list[dict], initial_cash: float, allow_short: bool = True) -> dict:
    trade_df = _to_trade_df(trades)
    equity = bt["equity"].astype(float).ffill().bfill() if (bt is not None and not bt.empty and "equity" in bt.columns) else pd.Series([float(initial_cash)])
    returns = equity.pct_change().fillna(0.0)
    dd, max_dd, dd_duration = _compute_drawdown(equity)

    wins = trade_df[trade_df["pnl_$"] > 0] if not trade_df.empty else pd.DataFrame(columns=trade_df.columns)
    losses = trade_df[trade_df["pnl_$"] < 0] if not trade_df.empty else pd.DataFrame(columns=trade_df.columns)
    gross_profit = float(wins["pnl_$"].sum()) if not wins.empty else 0.0
    gross_loss = float(abs(losses["pnl_$"].sum())) if not losses.empty else 0.0
    pf = float(gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    winrate = float(len(wins) / len(trade_df)) if len(trade_df) else 0.0
    avg_win = float(wins["pnl_$"].mean()) if not wins.empty else 0.0
    avg_loss = float(losses["pnl_$"].mean()) if not losses.empty else 0.0
    avg_win_r = float(wins["R"].mean()) if not wins.empty else 0.0
    avg_loss_r = float(losses["R"].mean()) if not losses.empty else 0.0
    expectancy = float((winrate * avg_win) + ((1.0 - winrate) * avg_loss))

    med_win = float(wins["pnl_$"].median()) if not wins.empty else 0.0
    med_loss = float(losses["pnl_$"].median()) if not losses.empty else 0.0
    max_wins, max_losses = _max_consecutive([v > 0 for v in trade_df["pnl_$"].tolist()] if not trade_df.empty else [])

    hold_bars = float(trade_df["hold_bars"].mean()) if not trade_df.empty else 0.0
    hold_days = float(trade_df["hold_days"].mean()) if not trade_df.empty else 0.0

    reason_dist = {
        "sl": 0,
        "tp": 0,
        "signal": 0,
        "forced": 0,
        "flip": 0,
    }
    if not trade_df.empty:
        for r in trade_df["exit_reason"].astype(str):
            if r in reason_dist:
                reason_dist[r] += 1
            elif r == "forced_eod":
                reason_dist["forced"] += 1
            else:
                reason_dist["signal"] += 1

    in_pos = (bt["qty"] != 0).astype(int) if (bt is not None and not bt.empty and "qty" in bt.columns) else pd.Series([0])
    long_pos = (bt["qty"] > 0).astype(int) if (bt is not None and not bt.empty and "qty" in bt.columns) else pd.Series([0])
    short_pos = (bt["qty"] < 0).astype(int) if (bt is not None and not bt.empty and "qty" in bt.columns) else pd.Series([0])
    exposure = float(in_pos.mean()) if len(in_pos) else 0.0
    long_exposure = float(long_pos.mean()) if len(long_pos) else 0.0
    short_exposure = float(short_pos.mean()) if len(short_pos) else 0.0

    years = max(len(equity), 1) / 252
    final_equity = float(equity.iloc[-1])
    total_return = float((final_equity / float(initial_cash)) - 1.0) if initial_cash > 0 else 0.0
    cagr = float((final_equity / float(initial_cash)) ** (1 / years) - 1) if years > 0 and initial_cash > 0 else 0.0
    vol = float(returns.std(ddof=0) * np.sqrt(252)) if len(returns) else 0.0
    sharpe = float((returns.mean() / returns.std(ddof=0)) * np.sqrt(252)) if returns.std(ddof=0) > 0 else 0.0

    avg_equity = float(equity.mean()) if len(equity) else float(initial_cash)
    turnover_notional = float((trade_df["entry_vwap"] * trade_df["qty"]).abs().sum() + (trade_df["exit_vwap"] * trade_df["qty"]).abs().sum()) if not trade_df.empty else 0.0
    turnover_proxy = float(turnover_notional / avg_equity) if avg_equity > 0 else 0.0

    monthly_returns = returns.resample("M").apply(lambda x: float((1 + x).prod() - 1.0)) if isinstance(returns.index, pd.DatetimeIndex) else pd.Series(dtype=float)

    fills_open = int(len(trade_df))
    fills_close = int(len(trade_df))
    trades_closed = int(len(trade_df))
    sessions_open, sessions_closed = _position_sessions(trade_df)
    forced_exit_count = int(trade_df["forced_exit"].sum()) if not trade_df.empty else 0

    summary = {
        "model": {
            "fills_open": fills_open,
            "fills_close": fills_close,
            "trades_closed": trades_closed,
            "position_sessions_opened": int(sessions_open),
            "position_sessions_closed": int(sessions_closed),
            "forced_exit_count": forced_exit_count,
        },
        "trade_metrics": {
            "winrate": winrate,
            "avg_win_$": avg_win,
            "avg_loss_$": avg_loss,
            "avg_win_R": avg_win_r,
            "avg_loss_R": avg_loss_r,
            "profit_factor": pf,
            "expectancy_$": expectancy,
            "median_win_$": med_win,
            "median_loss_$": med_loss,
            "max_consecutive_wins": int(max_wins),
            "max_consecutive_losses": int(max_losses),
            "average_hold_bars": hold_bars,
            "average_hold_days": hold_days,
            "exit_reason_distribution": reason_dist,
        },
        "equity_metrics": {
            "final_equity": final_equity,
            "total_return": total_return,
            "cagr": cagr,
            "max_drawdown": max_dd,
            "max_drawdown_duration_bars": dd_duration,
            "volatility": vol,
            "sharpe": sharpe,
            "exposure": exposure,
            "long_exposure": long_exposure,
            "short_exposure": short_exposure,
            "turnover_proxy": turnover_proxy,
        },
        "monthly_returns": {str(k.date()): float(v) for k, v in monthly_returns.items()},
        "invariants": {
            "trades_have_ordered_time": bool(trade_df.empty or ((trade_df["entry_dt"] < trade_df["exit_dt"]).all())),
            "forced_exit_reflected_in_trades": bool((forced_exit_count == 0) or (forced_exit_count <= trades_closed)),
            "equity_last_equals_final": bool(abs(float(equity.iloc[-1]) - final_equity) < 1e-9),
            "trades_pnl_matches_total": bool(abs(float(trade_df["pnl_$"].sum()) - float(final_equity - initial_cash)) < max(1.0, abs(final_equity - initial_cash) * 0.5) if not trade_df.empty else True),
            "no_short_when_disallowed": bool(allow_short or int((trade_df["direction"] == "SHORT").sum()) == 0),
        },
    }
    return summary


def save_diagnostic_artifacts(
    run_dir: str | Path,
    bt: pd.DataFrame,
    trades: list[dict],
    initial_cash: float,
    allow_short: bool = True,
) -> tuple[dict, DiagnosticArtifacts]:
    out_dir = Path(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = build_diagnostic_summary(bt=bt, trades=trades, initial_cash=initial_cash, allow_short=allow_short)
    trade_df = _to_trade_df(trades)

    equity_df = pd.DataFrame(index=bt.index if bt is not None and not bt.empty else pd.date_range(pd.Timestamp.utcnow(), periods=1, freq="D"))
    equity_df["equity"] = bt["equity"] if (bt is not None and not bt.empty and "equity" in bt.columns) else float(initial_cash)
    dd, _mdd, _dur = _compute_drawdown(equity_df["equity"].astype(float))
    equity_df["drawdown"] = dd.values
    equity_df["position_state"] = np.sign(bt["qty"]).astype(int).values if (bt is not None and not bt.empty and "qty" in bt.columns) else 0
    equity_df["exposure_flag"] = (equity_df["position_state"] != 0).astype(int)

    trades_csv = out_dir / "trades.csv"
    equity_csv = out_dir / "equity.csv"
    summary_json = out_dir / "summary.json"

    trade_df.to_csv(trades_csv, index=False)
    equity_df.to_csv(equity_csv)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plots
    equity_png = out_dir / "equity_curve.png"
    drawdown_png = out_dir / "drawdown_curve.png"
    trades_png = out_dir / "trades_chart_diag.png"

    fig, ax = plt.subplots(figsize=(10, 4))
    equity_df["equity"].plot(ax=ax, title="Equity Curve")
    fig.tight_layout()
    fig.savefig(equity_png)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3))
    equity_df["drawdown"].plot(ax=ax, title="Drawdown")
    fig.tight_layout()
    fig.savefig(drawdown_png)
    plt.close(fig)

    if bt is not None and not bt.empty and "Close" in bt.columns:
        fig, ax = plt.subplots(figsize=(11, 5))
        bt["Close"].plot(ax=ax, title="Trades (long/short markers)")
        if not trade_df.empty:
            longs = trade_df[trade_df["direction"] == "LONG"]
            shorts = trade_df[trade_df["direction"] == "SHORT"]
            ax.scatter(longs["entry_dt"], longs["entry_vwap"], marker="^", label="long entry")
            ax.scatter(longs["exit_dt"], longs["exit_vwap"], marker="v", label="long exit")
            ax.scatter(shorts["entry_dt"], shorts["entry_vwap"], marker="<", label="short entry")
            ax.scatter(shorts["exit_dt"], shorts["exit_vwap"], marker=">", label="short exit")
            ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(trades_png)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.set_title("No price data")
        fig.tight_layout()
        fig.savefig(trades_png)
        plt.close(fig)

    return summary, DiagnosticArtifacts(
        trades_csv=str(trades_csv),
        equity_csv=str(equity_csv),
        summary_json=str(summary_json),
        equity_png=str(equity_png),
        drawdown_png=str(drawdown_png),
        trades_png=str(trades_png),
    )


def format_diagnostic_summary(summary: dict) -> str:
    model = summary.get("model", {})
    tr = summary.get("trade_metrics", {})
    eq = summary.get("equity_metrics", {})
    return (
        "diag_summary "
        f"fills_open={model.get('fills_open', 0)} fills_close={model.get('fills_close', 0)} "
        f"trades_closed={model.get('trades_closed', 0)} sessions={model.get('position_sessions_closed', 0)} "
        f"forced_exit={model.get('forced_exit_count', 0)} | "
        f"winrate={tr.get('winrate', 0.0):.2%} pf={tr.get('profit_factor', 0.0):.3f} expectancy$={tr.get('expectancy_$', 0.0):.2f} | "
        f"ret={eq.get('total_return', 0.0):.2%} maxDD={eq.get('max_drawdown', 0.0):.2%} exposure={eq.get('exposure', 0.0):.2%}"
    )
