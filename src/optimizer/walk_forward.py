from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtester.engine import run_backtest
from src.indicators.calculator import add_indicators
from src.strategy.hybrid_vote import generate_positions
from src.config import SETTINGS


@dataclass(slots=True)
class WalkForwardConfig:
    train_months: int = 12
    test_months: int = 3
    step_months: int = 3
    folds: int = 0


def build_fold_windows(index: pd.DatetimeIndex, folds: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if len(index) < 100 or folds <= 1:
        return []
    n = len(index)
    fold_size = n // folds
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for k in range(1, folds):
        start_i = k * fold_size
        end_i = n if k == folds - 1 else (k + 1) * fold_size
        if start_i >= n:
            break
        windows.append((index[start_i], index[end_i - 1] + pd.Timedelta(days=1)))
    return windows


def build_windows(index: pd.DatetimeIndex, cfg: WalkForwardConfig) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if len(index) < 100:
        return []
    start = index.min().normalize()
    end = index.max().normalize()
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    anchor = start + pd.DateOffset(months=cfg.train_months)
    while True:
        test_start = anchor
        test_end = test_start + pd.DateOffset(months=cfg.test_months)
        if test_end > end:
            break
        windows.append((test_start, test_end))
        anchor = anchor + pd.DateOffset(months=cfg.step_months)
    return windows


def evaluate_config_walk_forward(
    df: pd.DataFrame,
    strategy_config: dict,
    commission_bps: float,
    slippage_bps: float,
    wf_cfg: WalkForwardConfig,
) -> tuple[float, dict, pd.DataFrame, list[dict]]:
    windows = build_fold_windows(df.index, wf_cfg.folds) if wf_cfg.folds > 0 else build_windows(df.index, wf_cfg)
    if not windows:
        start_cash = float(SETTINGS.initial_cash)
        empty = pd.DataFrame({"equity": [start_cash]})
        metrics = {
            "score": -999.0,
            "cagr": 0.0,
            "max_dd": 0.0,
            "max_dd_%": 0.0,
            "sharpe": 0.0,
            "trades_count": 0,
            "win_rate": 0.0,
            "final_equity": start_cash,
            "profit_$": 0.0,
            "profit_%": 0.0,
            "signals_count_enter": 0,
            "signals_count_exit": 0,
        }
        return float(metrics["score"]), metrics, empty, []

    all_test_bt: list[pd.DataFrame] = []
    all_trades: list[dict] = []

    warmup_days = 220
    for test_start, test_end in windows:
        warmup_start = test_start - pd.Timedelta(days=warmup_days)
        local = df.loc[(df.index >= warmup_start) & (df.index <= test_end)].copy()
        if len(local) < 80:
            continue

        local_ind = add_indicators(local, strategy_config)
        pos = generate_positions(local_ind, strategy_config)
        bt_full, _metrics, trades = run_backtest(
            local_ind,
            pos,
            commission_bps,
            slippage_bps,
            initial_cash=float(SETTINGS.initial_cash),
            position_size_pct=float(strategy_config.get("position_size_pct", SETTINGS.position_size_pct)),
            sl_pct=float(strategy_config.get("sl_pct", 0.01)),
            tp_pct=float(strategy_config.get("tp_pct", max(0.03, 3 * float(strategy_config.get("sl_pct", 0.01))))),
            execution_mode=str(strategy_config.get("execution_mode", "next_open")),
            debug_diagnostics=bool(SETTINGS.debug_diagnostics),
        )

        test_bt = bt_full.loc[(bt_full.index >= test_start) & (bt_full.index < test_end)].copy()
        if test_bt.empty:
            continue

        test_trades = [
            t
            for t in trades
            if pd.Timestamp(t["entry_date"]) >= test_start and pd.Timestamp(t["entry_date"]) < test_end
        ]
        all_test_bt.append(test_bt)
        all_trades.extend(test_trades)

    if not all_test_bt:
        start_cash = float(SETTINGS.initial_cash)
        empty = pd.DataFrame({"equity": [start_cash]})
        metrics = {
            "score": -999.0,
            "cagr": 0.0,
            "max_dd": 0.0,
            "max_dd_%": 0.0,
            "sharpe": 0.0,
            "trades_count": 0,
            "win_rate": 0.0,
            "final_equity": start_cash,
            "profit_$": 0.0,
            "profit_%": 0.0,
            "signals_count_enter": 0,
            "signals_count_exit": 0,
        }
        return float(metrics["score"]), metrics, empty, []

    combined = pd.concat(all_test_bt).sort_index()
    metrics = _aggregate_metrics(combined, all_trades, initial_cash=float(SETTINGS.initial_cash))
    metrics["windows"] = len(all_test_bt)
    entry_signal_series = (combined["signal"] == 1) if "signal" in combined else pd.Series(0, index=combined.index)
    exit_signal_series = (combined["signal"] == -1) if "signal" in combined else pd.Series(0, index=combined.index)
    metrics["signals_count_enter"] = int(entry_signal_series.astype(int).sum())
    metrics["signals_count_exit"] = int(exit_signal_series.astype(int).sum())
    return float(metrics["score"]), metrics, combined, all_trades


def _aggregate_metrics(combined: pd.DataFrame, trades: list[dict], initial_cash: float = 1.0) -> dict:
    equity = combined["equity"].ffill().bfill().fillna(float(initial_cash))
    years = max(len(combined), 1) / 252
    cagr = float((equity.iloc[-1] / initial_cash) ** (1 / years) - 1) if years > 0 and initial_cash > 0 else 0.0
    dd = equity / equity.cummax() - 1
    max_dd = float(abs(dd.min()))

    std = combined["strategy_ret"].std(ddof=0)
    sharpe = (
        float((combined["strategy_ret"].mean() / std) * np.sqrt(252))
        if std is not None and std > 0
        else 0.0
    )

    trades_count = len(trades)
    win_rate = float(sum(1 for t in trades if t["pnl"] > 0) / trades_count) if trades_count else 0.0
    penalty = 2.0 if trades_count < 20 else 0.0
    score = cagr - 0.5 * max_dd - penalty

    final_equity = float(equity.iloc[-1])
    profit_abs = final_equity - float(initial_cash)
    profit_pct = (profit_abs / float(initial_cash)) * 100.0 if initial_cash > 0 else 0.0

    return {
        "score": float(score),
        "cagr": cagr,
        "max_dd": max_dd,
        "max_dd_%": float(max_dd * 100.0),
        "sharpe": sharpe,
        "trades_count": trades_count,
        "win_rate": win_rate,
        "final_equity": final_equity,
        "profit_$": float(profit_abs),
        "profit_%": float(profit_pct),
    }
