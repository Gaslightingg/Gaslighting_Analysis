from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class QualityThresholds:
    n_min_trades: int = 30
    pf_min: float = 1.1
    max_dd_min: float = -0.10
    top3_max: float = 0.60


def _max_streak(flags: list[bool]) -> tuple[int, int]:
    max_win = max_loss = run_win = run_loss = 0
    for ok in flags:
        if ok:
            run_win += 1
            run_loss = 0
        else:
            run_loss += 1
            run_win = 0
        max_win = max(max_win, run_win)
        max_loss = max(max_loss, run_loss)
    return int(max_win), int(max_loss)


def _drawdown_stats(equity: pd.Series) -> tuple[float, int, pd.Series]:
    if equity.empty:
        return 0.0, 0, pd.Series(dtype=float)
    peak = equity.cummax().replace(0, np.nan)
    dd = (equity / peak) - 1.0
    max_dd = float(dd.min()) if len(dd) else 0.0
    longest = cur = 0
    for v in dd.fillna(0.0).tolist():
        if v < 0:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return float(max_dd if np.isfinite(max_dd) else 0.0), int(longest), dd.fillna(0.0)


def compute_diagnostic_metrics(
    trades: list[dict],
    equity: pd.Series,
    qty: pd.Series | None = None,
    allow_short: bool = True,
) -> dict:
    trade_df = pd.DataFrame(trades or [])
    if trade_df.empty:
        trade_df = pd.DataFrame(columns=["pnl_$", "R", "exit_reason", "side"])

    pnl = pd.to_numeric(trade_df.get("pnl_$", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    rvals = pd.to_numeric(trade_df.get("R", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    n_trades = int(len(trade_df))

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(losses.sum())
    profit_factor = float(gross_profit / abs(gross_loss)) if gross_loss < 0 else (float("inf") if gross_profit > 0 else 0.0)

    top3_sum = float(pnl.nlargest(3).sum()) if n_trades else 0.0
    total_pnl = float(pnl.sum())
    top3_contribution = float(top3_sum / total_pnl) if total_pnl > 0 else 0.0

    max_wins, max_losses = _max_streak((pnl > 0).tolist())

    reason_counts = trade_df.get("exit_reason", pd.Series(dtype=str)).fillna("signal").astype(str).value_counts().to_dict()
    reason_breakdown = {
        reason: {"count": int(cnt), "pct": float(cnt / n_trades) if n_trades else 0.0}
        for reason, cnt in reason_counts.items()
    }

    eq = pd.to_numeric(equity, errors="coerce").ffill().bfill().fillna(0.0)
    final_equity = float(eq.iloc[-1]) if len(eq) else 0.0
    total_return = float((eq.iloc[-1] / eq.iloc[0]) - 1.0) if len(eq) and eq.iloc[0] != 0 else 0.0
    max_dd, dd_duration, _dd_series = _drawdown_stats(eq)

    rets = eq.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vol = float(rets.std(ddof=0) * np.sqrt(252)) if len(rets) else 0.0
    sharpe = float((rets.mean() / rets.std(ddof=0)) * np.sqrt(252)) if rets.std(ddof=0) > 0 else 0.0

    if qty is None:
        qty = pd.Series(0.0, index=eq.index)
    q = pd.to_numeric(qty, errors="coerce").reindex(eq.index).fillna(0.0)
    exposure = float((q != 0).mean()) if len(q) else 0.0
    long_exposure = float((q > 0).mean()) if len(q) else 0.0
    short_exposure = float((q < 0).mean()) if len(q) else 0.0

    short_trades = int((trade_df.get("side", pd.Series(dtype=str)).fillna("").astype(str).str.lower() == "short").sum())

    return {
        "n_trades": n_trades,
        "winrate": float((pnl > 0).mean()) if n_trades else 0.0,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "avg_win_$": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_$": float(losses.mean()) if len(losses) else 0.0,
        "median_win_$": float(wins.median()) if len(wins) else 0.0,
        "median_loss_$": float(losses.median()) if len(losses) else 0.0,
        "avg_R": float(rvals.mean()) if n_trades else 0.0,
        "avg_win_R": float(rvals[pnl > 0].mean()) if len(wins) else 0.0,
        "avg_loss_R": float(rvals[pnl < 0].mean()) if len(losses) else 0.0,
        "expectancy$": float(pnl.mean()) if n_trades else 0.0,
        "expectancyR": float(rvals.mean()) if n_trades else 0.0,
        "max_consecutive_losses": max_losses,
        "max_consecutive_wins": max_wins,
        "exit_reason_breakdown": reason_breakdown,
        "top3_contribution": top3_contribution,
        "total_pnl$": total_pnl,
        "final_equity": final_equity,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "dd_duration": dd_duration,
        "volatility": vol,
        "sharpe": sharpe,
        "exposure": exposure,
        "long_exposure": long_exposure,
        "short_exposure": short_exposure,
        "time_in_market_days": int((q != 0).sum()),
        "short_trades": short_trades,
        "allow_short": bool(allow_short),
        "equity_points": int(len(eq)),
    }


def quality_flags(metrics: dict, thresholds: QualityThresholds) -> dict:
    n_trades = int(metrics.get("n_trades", 0))
    pf = float(metrics.get("profit_factor", 0.0))
    exp_d = float(metrics.get("expectancy$", 0.0))
    max_dd = float(metrics.get("max_drawdown", 0.0))
    top3 = float(metrics.get("top3_contribution", 0.0))
    total_pnl = float(metrics.get("total_pnl$", 0.0))
    short_trades = int(metrics.get("short_trades", 0))
    allow_short = bool(metrics.get("allow_short", True))

    return {
        "too_few_trades": bool(n_trades < int(thresholds.n_min_trades)),
        "pf_bad": bool(np.isfinite(pf) and pf < float(thresholds.pf_min)),
        "expectancy_bad": bool(exp_d <= 0.0),
        "dd_too_high": bool(max_dd < float(thresholds.max_dd_min)),
        "top3_dominates": bool(total_pnl > 0 and top3 > float(thresholds.top3_max)),
        "no_short_when_allowed": bool(allow_short and short_trades == 0),
    }


def quality_penalty(flags: dict) -> float:
    weights = {
        "too_few_trades": 1.2,
        "pf_bad": 1.5,
        "expectancy_bad": 1.5,
        "dd_too_high": 1.2,
        "top3_dominates": 1.0,
        "no_short_when_allowed": 0.3,
    }
    return float(sum(weights.get(k, 0.0) for k, v in flags.items() if bool(v)))
