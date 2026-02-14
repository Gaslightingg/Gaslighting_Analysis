from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtester.engine import run_backtest
from src.config import SETTINGS
from src.indicators.calculator import add_indicators
from src.strategy.hybrid_vote import generate_positions
from src.strategy.retrieval_strategy import generate_retrieval_positions


@dataclass(slots=True)
class WalkForwardConfig:
    train_months: int = 12
    test_months: int = 3
    step_months: int = 3
    folds: int = 0
    train_size: int = 0
    test_size: int = 0
    step_size: int = 0
    window_type: str = "expanding"


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

    if cfg.train_size > 0 and cfg.test_size > 0:
        step = cfg.step_size if cfg.step_size > 0 else cfg.test_size
        anchor = cfg.train_size
        windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        while anchor + cfg.test_size <= len(index):
            test_start_i = anchor
            test_end_i = anchor + cfg.test_size
            windows.append((index[test_start_i], index[test_end_i - 1] + pd.Timedelta(days=1)))
            anchor += step
        return windows

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


def _build_window_slice(df: pd.DataFrame, test_start: pd.Timestamp, test_end: pd.Timestamp, cfg: WalkForwardConfig) -> pd.DataFrame:
    test_mask = (df.index >= test_start) & (df.index < test_end)
    if not test_mask.any():
        return pd.DataFrame(columns=df.columns)

    if cfg.train_size > 0 and cfg.test_size > 0:
        test_start_i = int(np.argmax(test_mask))
        train_start_i = 0
        if str(cfg.window_type).lower() == "rolling":
            train_start_i = max(0, test_start_i - cfg.train_size)
        local = df.iloc[train_start_i : test_start_i + cfg.test_size].copy()
        return local

    warmup_days = 220
    warmup_start = test_start - pd.Timedelta(days=warmup_days)
    return df.loc[(df.index >= warmup_start) & (df.index <= test_end)].copy()


def _baseline_curves(price_df: pd.DataFrame, initial_cash: float) -> pd.DataFrame:
    close = pd.to_numeric(price_df["Close"], errors="coerce").ffill().bfill()
    ret = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    buyhold = float(initial_cash) * (1.0 + ret).cumprod()

    ma200 = close.rolling(200, min_periods=200).mean()
    ma_pos = (close > ma200).astype(int).shift(1).fillna(0)
    ma_ret = ret * ma_pos
    ma_eq = float(initial_cash) * (1.0 + ma_ret).cumprod()

    out = pd.DataFrame(index=price_df.index)
    out["buyhold_equity"] = buyhold
    out["ma200_equity"] = ma_eq
    out["ma200_position"] = ma_pos
    return out


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
            "score": -1.0,
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
            "enter_count": 0,
            "exit_count": 0,
            "closed_trades_count": 0,
            "open_position": 0,
            "open_position_qty": 0.0,
            "forced_exit_count": 0,
            "exits_by_rule": 0,
            "exits_by_sl_tp": 0,
            "exits_forced_end": 0,
            "exits_flip": 0,
            "exposure": 0.0,
            "avg_hold_bars": 0.0,
            "entry_events_count": 0,
            "exit_events_count": 0,
            "open_positions_count": 0,
        }
        return float(metrics["score"]), metrics, empty, []

    all_test_bt: list[pd.DataFrame] = []
    all_trades: list[dict] = []
    strategy_family = str(strategy_config.get("strategy_family", "hybrid_vote"))

    for test_start, test_end in windows:
        local = _build_window_slice(df, test_start=test_start, test_end=test_end, cfg=wf_cfg)
        if len(local) < 80:
            continue

        local_cfg = dict(strategy_config)
        local_cfg.setdefault("allow_short", strategy_family != "retrieval")
        if strategy_family == "retrieval":
            out = generate_retrieval_positions(local, local_cfg, test_start=test_start)
            local_ind = local.copy()
            local_ind = pd.concat([local_ind, out.features.add_prefix("feat_")], axis=1)
            pos = out.signal_df
        else:
            local_ind = add_indicators(local, local_cfg)
            pos = generate_positions(local_ind, local_cfg)

        bt_full, _metrics, trades = run_backtest(
            local_ind,
            pos,
            commission_bps,
            slippage_bps,
            initial_cash=float(SETTINGS.initial_cash),
            position_size_pct=float(local_cfg.get("position_size_pct", SETTINGS.position_size_pct)),
            sl_pct=float(local_cfg.get("sl_pct", 0.01)),
            tp_pct=float(local_cfg.get("tp_pct", max(0.03, 3 * float(local_cfg.get("sl_pct", 0.01))))),
            execution_mode=str(local_cfg.get("execution_mode", "next_open")),
            debug_diagnostics=bool(SETTINGS.debug_diagnostics),
            allow_short=bool(local_cfg.get("allow_short", True)),
        )

        test_bt = bt_full.loc[(bt_full.index >= test_start) & (bt_full.index < test_end)].copy()
        if test_bt.empty:
            continue

        baseline_df = _baseline_curves(test_bt, initial_cash=float(SETTINGS.initial_cash))
        test_bt = pd.concat([test_bt, baseline_df], axis=1)

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
            "score": -1.0,
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
            "enter_count": 0,
            "exit_count": 0,
            "closed_trades_count": 0,
            "open_position": 0,
            "open_position_qty": 0.0,
            "forced_exit_count": 0,
            "exits_by_rule": 0,
            "exits_by_sl_tp": 0,
            "exits_forced_end": 0,
            "exits_flip": 0,
            "exposure": 0.0,
            "avg_hold_bars": 0.0,
            "entry_events_count": 0,
            "exit_events_count": 0,
            "open_positions_count": 0,
        }
        return float(metrics["score"]), metrics, empty, []

    combined = pd.concat(all_test_bt).sort_index()
    metrics = _aggregate_metrics(combined, all_trades, initial_cash=float(SETTINGS.initial_cash))
    metrics["windows"] = len(all_test_bt)
    enter_long = combined["enter_long"] if "enter_long" in combined.columns else pd.Series(0, index=combined.index)
    enter_short = combined["enter_short"] if "enter_short" in combined.columns else pd.Series(0, index=combined.index)
    exit_long = combined["exit_long"] if "exit_long" in combined.columns else pd.Series(0, index=combined.index)
    exit_short = combined["exit_short"] if "exit_short" in combined.columns else pd.Series(0, index=combined.index)
    metrics["signals_count_enter"] = int(enter_long.fillna(0).astype(int).sum() + enter_short.fillna(0).astype(int).sum())
    metrics["signals_count_exit"] = int(exit_long.fillna(0).astype(int).sum() + exit_short.fillna(0).astype(int).sum())
    forced_exit_count = int(sum(1 for t in all_trades if bool(t.get("forced_exit", False))))
    exits_by_sl_tp = int(sum(1 for t in all_trades if str(t.get("exit_reason", "")) in {"sl", "tp"}))
    exits_flip = int(sum(1 for t in all_trades if str(t.get("exit_reason", "")) == "flip"))
    exits_by_rule = int(len(all_trades) - forced_exit_count - exits_by_sl_tp - exits_flip)
    metrics["forced_exit_count"] = forced_exit_count
    metrics["exits_by_sl_tp"] = max(exits_by_sl_tp, 0)
    metrics["exits_flip"] = max(exits_flip, 0)
    metrics["exits_by_rule"] = max(exits_by_rule, 0)
    metrics["exits_forced_end"] = forced_exit_count
    metrics["closed_trades_count"] = int(len(all_trades))
    metrics["exit_count"] = int(metrics["signals_count_exit"] + forced_exit_count)
    metrics["enter_count"] = int(metrics["signals_count_enter"])
    if metrics["exit_count"] > metrics["enter_count"]:
        metrics["exit_count"] = int(metrics["enter_count"])
    metrics["open_position"] = int(np.sign(float(combined["qty"].iloc[-1]))) if "qty" in combined.columns and not combined.empty else 0
    metrics["open_position_qty"] = float(combined["qty"].iloc[-1]) if "qty" in combined.columns and not combined.empty else 0.0
    metrics["exposure"] = float((combined["qty"] != 0).sum() / len(combined)) if "qty" in combined.columns and len(combined) else 0.0
    metrics["avg_hold_bars"] = float(np.mean([int(t.get("holding_bars", 0)) for t in all_trades])) if all_trades else 0.0
    metrics["entry_events_count"] = int(metrics["enter_count"])
    metrics["exit_events_count"] = int(metrics["exit_count"])
    metrics["open_positions_count"] = int(1 if metrics["open_position"] != 0 else 0)

    if "buyhold_equity" in combined.columns:
        metrics["buyhold_final_equity"] = float(combined["buyhold_equity"].iloc[-1])
        metrics["buyhold_return"] = float((metrics["buyhold_final_equity"] / float(SETTINGS.initial_cash)) - 1.0)
    if "ma200_equity" in combined.columns:
        metrics["ma200_final_equity"] = float(combined["ma200_equity"].iloc[-1])
        metrics["ma200_return"] = float((metrics["ma200_final_equity"] / float(SETTINGS.initial_cash)) - 1.0)

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
    target_trades = 50.0
    trade_scale = np.sqrt(min(max(trades_count, 1) / target_trades, 1.5))
    score = float(sharpe * trade_scale - 0.5 * max_dd)

    final_equity = float(equity.iloc[-1])
    profit_abs = final_equity - float(initial_cash)
    profit_pct = (profit_abs / float(initial_cash)) * 100.0 if initial_cash > 0 else 0.0

    stability_score = float((sharpe / (1.0 + 5.0 * max_dd)) * np.sqrt((trades_count + 1) / (trades_count + 20)))

    return {
        "score": score,
        "cagr": cagr,
        "max_dd": max_dd,
        "max_dd_%": float(max_dd * 100.0),
        "sharpe": sharpe,
        "trades_count": trades_count,
        "win_rate": win_rate,
        "final_equity": final_equity,
        "profit_$": float(profit_abs),
        "profit_%": float(profit_pct),
        "stability_score": stability_score,
    }
