from __future__ import annotations

import json
import logging
import time
import traceback
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from src.analytics.metrics import QualityThresholds, compute_diagnostic_metrics, quality_flags, quality_penalty
from src.config import SETTINGS
from src.indicators.calculator import add_indicators
from src.optimizer.walk_forward import WalkForwardConfig, evaluate_config_walk_forward
from src.reporter.diagnostics import save_diagnostic_artifacts
from src.reporter.report import save_best_artifacts
from src.storage.repository import Repository

MAX_IDENTICAL_EXCEPTIONS = 5
TRACEBACK_LIMIT = 2000
_LOG = logging.getLogger("optimizer.optuna")
MIN_TRADES_REQUIRED = 20
SOFT_FAIL_SCORE = -50.0


def _safe_float(value: object, default: float) -> float:
    try:
        out = float(value)
    except Exception:  # noqa: BLE001
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return int(default)


def _ensure_trial_metrics(metrics: dict | None, initial_cash: float, trades: list[dict]) -> dict:
    out = dict(metrics or {})
    start_cash = float(initial_cash)
    trades_count = len(trades)
    out["trades_count"] = int(trades_count)
    out["start_cash"] = start_cash

    final_equity = _safe_float(out.get("final_equity"), start_cash)
    if trades_count == 0:
        final_equity = start_cash
    out["final_equity"] = float(final_equity)
    out["profit_$"] = float(final_equity - start_cash)
    out["profit_%"] = float(((final_equity / start_cash) - 1.0) * 100.0) if start_cash > 0 else 0.0
    out["signals_count_enter"] = _safe_int(out.get("signals_count_enter", 0), 0)
    out["signals_count_exit"] = _safe_int(out.get("signals_count_exit", 0), 0)
    out["score"] = _safe_float(out.get("score", SOFT_FAIL_SCORE), SOFT_FAIL_SCORE)
    out["closed_trades_count"] = _safe_int(out.get("closed_trades_count", trades_count), trades_count)
    out["entry_events_count"] = _safe_int(out.get("entry_events_count", out.get("enter_count", out.get("signals_count_enter", 0))), 0)
    out["exit_events_count"] = _safe_int(out.get("exit_events_count", out.get("exit_count", out.get("signals_count_exit", 0))), 0)
    out["open_positions_count"] = _safe_int(out.get("open_positions_count", 0 if _safe_int(out.get("open_position", 0), 0) == 0 else 1), 0)
    out["forced_exit_count"] = _safe_int(out.get("forced_exit_count", 0), 0)
    out["exits_by_rule"] = _safe_int(out.get("exits_by_rule", 0), 0)
    out["exits_by_sl_tp"] = _safe_int(out.get("exits_by_sl_tp", 0), 0)
    out["exits_forced_end"] = _safe_int(out.get("exits_forced_end", out.get("forced_exit_count", 0)), 0)
    out["exits_flip"] = _safe_int(out.get("exits_flip", 0), 0)
    return out


def _trial_snapshot(number: int, score: float, note: str, reason: str, metrics: dict, cfg: dict, diag_metrics: dict | None = None, flags: dict | None = None) -> dict:
    return {
        "number": int(number),
        "score": float(score),
        "base_score": float(metrics.get("base_score", score)),
        "diag_score": float(metrics.get("diag_score", score)),
        "note": str(note),
        "reason": str(reason),
        "trades_count": int(metrics.get("trades_count", 0)),
        "closed_trades_count": int(metrics.get("closed_trades_count", metrics.get("trades_count", 0))),
        "signals_count_enter": int(metrics.get("signals_count_enter", 0)),
        "signals_count_exit": int(metrics.get("signals_count_exit", 0)),
        "entry_events_count": int(metrics.get("entry_events_count", metrics.get("signals_count_enter", 0))),
        "exit_events_count": int(metrics.get("exit_events_count", metrics.get("signals_count_exit", 0))),
        "open_positions_count": int(metrics.get("open_positions_count", 0)),
        "forced_exit_count": int(metrics.get("forced_exit_count", 0)),
        "exits_by_rule": int(metrics.get("exits_by_rule", 0)),
        "exits_by_sl_tp": int(metrics.get("exits_by_sl_tp", 0)),
        "exits_forced_end": int(metrics.get("exits_forced_end", metrics.get("forced_exit_count", 0))),
        "exits_flip": int(metrics.get("exits_flip", 0)),
        "final_equity": float(metrics.get("final_equity", 0.0)),
        "profit_$": float(metrics.get("profit_$", 0.0)),
        "profit_%": float(metrics.get("profit_%", 0.0)),
        "start_cash": float(metrics.get("start_cash", SETTINGS.initial_cash)),
        "diag_flags": dict(flags or {}),
        "diag_metrics": dict(diag_metrics or {}),
        "params": {
            "ema_fast": cfg.get("ema_fast"),
            "ema_slow": cfg.get("ema_slow"),
            "rsi_period": cfg.get("rsi_period"),
            "buy_below": cfg.get("buy_below"),
            "sell_above": cfg.get("sell_above"),
            "bb_period": cfg.get("bb_period"),
            "bb_std": cfg.get("bb_std"),
            "adx_period": cfg.get("adx_period"),
            "adx_min": cfg.get("adx_min"),
            "regime_mode": cfg.get("regime_mode"),
            "enter_long": cfg.get("enter_long"),
            "exit_long": cfg.get("exit_long"),
            "sl_pct": cfg.get("sl_pct"),
            "tp_pct": cfg.get("tp_pct"),
            "execution_mode": cfg.get("execution_mode"),
            "position_size_pct": cfg.get("position_size_pct"),
            "strategy_family": cfg.get("strategy_family"),
            "retrieval_k": cfg.get("retrieval_k"),
            "retrieval_horizon": cfg.get("retrieval_horizon"),
            "retrieval_min_neighbors": cfg.get("retrieval_min_neighbors"),
            "entry_mean_threshold": cfg.get("entry_mean_threshold"),
            "entry_score_threshold": cfg.get("entry_score_threshold"),
            "exit_score_threshold": cfg.get("exit_score_threshold"),
            "risk_limit": cfg.get("risk_limit"),
            "max_dist_percentile": cfg.get("max_dist_percentile"),
            "hold_bars": cfg.get("hold_bars"),
            "embargo_bars": cfg.get("embargo_bars"),
            "wf_train_size": cfg.get("wf_train_size"),
            "wf_test_size": cfg.get("wf_test_size"),
            "wf_window_type": cfg.get("wf_window_type"),
        },
    }


def _save_trial_summary(run_dir: Path, trial_number: int, payload: dict) -> Path:
    trial_dir = run_dir / "trials" / f"trial_{int(trial_number):05d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    out_path = trial_dir / "summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _save_trial_heavy_artifacts(run_dir: Path, trial_number: int, eq_df: pd.DataFrame | None, trades: list[dict]) -> tuple[str | None, str | None]:
    trial_dir = run_dir / "trials" / f"trial_{int(trial_number):05d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    eq_path = None
    tr_path = None
    if eq_df is not None and not eq_df.empty:
        out = eq_df.copy()
        keep_cols = [c for c in ["equity", "qty", "Close", "strategy_ret"] if c in out.columns]
        if keep_cols:
            out[keep_cols].to_csv(trial_dir / "equity.csv")
            eq_path = str(trial_dir / "equity.csv")

    tr_df = pd.DataFrame(trades or [])
    tr_df.to_csv(trial_dir / "trades.csv", index=False)
    (trial_dir / "trades.json").write_text(json.dumps(trades or [], indent=2), encoding="utf-8")
    tr_path = str(trial_dir / "trades.csv")
    return eq_path, tr_path


def _flags_to_short(flags: dict) -> str:
    bad = [k for k, v in flags.items() if bool(v)]
    return ",".join(sorted(bad)) if bad else "ok"


def _is_maximize(study: optuna.Study) -> bool:
    return str(study.direction).lower().endswith("maximize")


def _is_better_score(candidate: float, current: float | None, maximize: bool) -> bool:
    if current is None:
        return True
    if maximize:
        return float(candidate) > float(current)
    return float(candidate) < float(current)


def _sort_rows_by_key(rows: list[dict], key: str, maximize: bool) -> list[dict]:
    return sorted(rows, key=lambda r: float(r.get(key, SOFT_FAIL_SCORE if maximize else abs(SOFT_FAIL_SCORE))), reverse=maximize)


def _select_leaders(leaderboard_rows: list[dict], maximize: bool) -> dict:
    if not leaderboard_rows:
        return {"best_by_equity": None, "best_by_base_score": None, "best_by_diag_score": None}

    best_by_equity = max(leaderboard_rows, key=lambda r: float(r.get("final_equity", 0.0)))
    best_by_base_score = _sort_rows_by_key(leaderboard_rows, "base_score", maximize=maximize)[0]
    best_by_diag_score = _sort_rows_by_key(leaderboard_rows, "diag_score", maximize=maximize)[0]
    return {
        "best_by_equity": best_by_equity,
        "best_by_base_score": best_by_base_score,
        "best_by_diag_score": best_by_diag_score,
    }


def run_optimization_job(job_id: str, df) -> dict:
    repo = Repository()
    job_data = repo.get_job_data(job_id)
    if not job_data:
        return {"status": "failed", "error": "job not found"}

    params = job_data["params"]
    ticker = str(params.get("ticker", ""))
    trials_total = int(params["trials_total"])
    checkpoint_n = int(params["checkpoint_n"])
    min_required = int(SETTINGS.min_bars)
    wf_folds = int(params.get("wf_folds", 0) or 0)

    if len(df) <= 0:
        repo.set_job_status(job_id, "finished_no_results")
        reason = "no_data: bars=0"
        repo.update_progress(
            job_id=job_id,
            trials_done=0,
            trials_total=trials_total,
            last_score=0.0,
            best_score=None,
            state="finished_no_results",
            reason=reason,
        )
        return {"status": "finished_no_results", "reason": reason}

    if len(df) < min_required:
        repo.set_job_status(job_id, "finished_no_results")
        reason = f"not_enough_bars: bars={len(df)} min_required={min_required}"
        repo.update_progress(
            job_id=job_id,
            trials_done=0,
            trials_total=trials_total,
            last_score=0.0,
            best_score=None,
            state="finished_no_results",
            reason=reason,
        )
        return {"status": "finished_no_results", "reason": reason}

    repo.set_job_status(job_id, "running")

    best_score: float | None = None
    best_metrics: dict = {}
    best_config: dict = {}
    best_overall: dict | None = None
    best_valid: dict | None = None
    best_diag: dict | None = None
    last_score = 0.0
    trials_done = 0

    repeated_exception_count = 0
    last_exception_key = ""

    run_dir = Path(SETTINGS.runs_dir) / job_id
    run_dir.mkdir(parents=True, exist_ok=True)
    study_storage = f"sqlite:///{(run_dir / 'study.db').as_posix()}"
    study = optuna.create_study(
        study_name=f"job_{job_id}",
        direction="maximize",
        sampler=TPESampler(seed=SETTINGS.seed),
        pruner=MedianPruner(n_startup_trials=30, n_warmup_steps=1, interval_steps=1),
        storage=study_storage,
        load_if_exists=True,
    )
    score_maximize = True
    if not _is_maximize(study):
        _LOG.warning("study_direction_mismatch expected=maximize actual=%s; forcing maximize ranking", str(study.direction))
    wf_cfg = WalkForwardConfig(train_months=12, test_months=3, step_months=3, folds=wf_folds)
    thresholds = QualityThresholds(
        n_min_trades=int(SETTINGS.diag_n_min_trades),
        pf_min=float(SETTINGS.diag_pf_min),
        max_dd_min=float(SETTINGS.diag_max_dd_min),
        top3_max=float(SETTINGS.diag_top3_max),
    )
    diag_top_k = max(int(SETTINGS.diag_top_k), 1)
    diag_top_plot_k = max(int(SETTINGS.diag_top_plot_k), 1)
    leaderboard_rows: list[dict] = []

    completed_trials = len([tr for tr in study.trials if tr.state.is_finished()])
    start_i = completed_trials + 1
    if start_i > trials_total:
        start_i = trials_total + 1

    for i in range(start_i, trials_total + 1):
        trials_done = i
        if repo.stop_requested(job_id):
            repo.update_progress(
                job_id=job_id,
                trials_done=trials_done,
                trials_total=trials_total,
                last_score=last_score,
                best_score=best_score,
                state="stopped",
            )
            return {"status": "stopped", "best_score": best_score}

        trial = study.ask()
        cfg = _sample(trial)
        wf_cfg_trial = WalkForwardConfig(
            train_months=12,
            test_months=3,
            step_months=3,
            folds=wf_folds,
            train_size=int(cfg.get("wf_train_size", 0) or 0),
            test_size=int(cfg.get("wf_test_size", 0) or 0),
            step_size=int(cfg.get("wf_test_size", 0) or 0),
            window_type=str(cfg.get("wf_window_type", "expanding")),
        )
        trial_started_at = time.monotonic()
        _LOG.info("trial_start number=%s bars=%s min_required=%s proceeding=true", i, len(df), min_required)
        note = "ok"
        reason = ""
        error_text = ""
        traceback_text = ""
        metrics: dict = {}
        eq_df = None
        trades: list[dict] = []

        try:
            if SETTINGS.debug_diagnostics and i <= 5:
                local_ind = add_indicators(df.copy(), cfg)
                feature_cols = ["ema_fast", "ema_slow", "rsi", "adx", "bb_lower", "bb_upper"]
                exists = [c for c in feature_cols if c in local_ind.columns]
                nan_share = {
                    c: round(float(local_ind[c].isna().mean()), 4)
                    for c in ["rsi", "adx", "bb_lower", "bb_upper"]
                    if c in local_ind.columns
                }
                rows_after_dropna = int(
                    local_ind.dropna(
                        subset=[c for c in ["rsi", "adx", "bb_lower", "bb_upper"] if c in local_ind.columns]
                    ).shape[0]
                )
                _LOG.info(
                    "trial_diag number=%s feature_cols=%s nan_share=%s rows_after_dropna=%s",
                    i,
                    exists,
                    nan_share,
                    rows_after_dropna,
                )

            score, metrics, eq_df, trades = evaluate_config_walk_forward(
                df,
                cfg,
                SETTINGS.commission_bps,
                SETTINGS.slippage_bps,
                wf_cfg_trial,
            )
            trial.report(float(score), step=1)
            if trial.should_prune():
                raise optuna.TrialPruned()
            repeated_exception_count = 0
            last_exception_key = ""
        except optuna.TrialPruned:
            score = SOFT_FAIL_SCORE
            note = "pruned"
            reason = "trial pruned by MedianPruner"
        except Exception as exc:  # noqa: BLE001
            score = SOFT_FAIL_SCORE
            note = "exception"
            reason = "Исключение в расчёте trial"
            error_text = f"{type(exc).__name__}: {exc}"
            traceback_text = traceback.format_exc()[:TRACEBACK_LIMIT]
            _LOG.exception(
                "trial failed",
                extra={"job_id": job_id, "trial": i, "ticker": ticker},
            )
            if error_text == last_exception_key:
                repeated_exception_count += 1
            else:
                repeated_exception_count = 1
                last_exception_key = error_text

        metrics = _ensure_trial_metrics(metrics, float(SETTINGS.initial_cash), trades)

        if note != "exception":
            if not np.isfinite(score):
                note = "nan_score"
                reason = "score не является конечным числом"
                score = SOFT_FAIL_SCORE
            else:
                enter_signals = int(metrics.get("entry_events_count", metrics.get("signals_count_enter", 0)))
                exit_signals = int(metrics.get("exit_events_count", metrics.get("signals_count_exit", 0)))
                trades_count = len(trades)
                metrics["trades_count"] = trades_count
                metrics["closed_trades_count"] = trades_count
                if exit_signals > enter_signals:
                    exit_signals = enter_signals
                    metrics["exit_events_count"] = exit_signals
                    metrics["signals_count_exit"] = exit_signals
                if enter_signals == 0:
                    note = "no_entries"
                    reason = "no_entries: 0 entry signals"
                    score = min(float(score), SOFT_FAIL_SCORE - 5.0)
                elif trades_count == 0:
                    note = "no_trades"
                    reason = "no_trades: entry signals were present, but no trades executed"
                    score = min(float(score), SOFT_FAIL_SCORE - 2.5)

        metrics["reason"] = note if note else "ok"
        metrics["base_score"] = float(score)

        eq_series = eq_df["equity"] if (eq_df is not None and not eq_df.empty and "equity" in eq_df.columns) else pd.Series([float(SETTINGS.initial_cash)])
        qty_series = eq_df["qty"] if (eq_df is not None and not eq_df.empty and "qty" in eq_df.columns) else pd.Series([0.0], index=eq_series.index)
        diag_metrics = compute_diagnostic_metrics(
            trades=trades,
            equity=eq_series,
            qty=qty_series,
            allow_short=bool(cfg.get("allow_short", True)),
        )
        flags = quality_flags(diag_metrics, thresholds)
        penalty = quality_penalty(flags)
        diag_score = float(score - penalty) if np.isfinite(score) else SOFT_FAIL_SCORE

        metrics["diag_score"] = float(diag_score)
        metrics["diag_penalty"] = float(penalty)
        metrics["diag_flags"] = dict(flags)
        metrics["diag_metrics"] = dict(diag_metrics)
        metrics["score"] = float(diag_score)

        enter_events = int(metrics.get("entry_events_count", metrics.get("signals_count_enter", 0)))
        exit_events = int(metrics.get("exit_events_count", metrics.get("signals_count_exit", 0)))
        forced_exits = int(metrics.get("forced_exit_count", 0))
        if exit_events > enter_events:
            _LOG.warning("trial_invariant exit_gt_enter number=%s enter=%s exit=%s", i, enter_events, exit_events)
            exit_events = enter_events
            metrics["exit_events_count"] = exit_events
            metrics["signals_count_exit"] = exit_events
        if forced_exits > exit_events:
            _LOG.warning("trial_invariant forced_gt_exit number=%s forced=%s exit=%s", i, forced_exits, exit_events)
            metrics["forced_exit_count"] = exit_events
            forced_exits = exit_events

        position_opened = bool((eq_df is not None) and (not eq_df.empty) and ("qty" in eq_df.columns) and (eq_df["qty"] != 0).any())
        _LOG.info(
            "trial_diag_end number=%s entry_signals=%s exit_signals=%s entry_events=%s exit_events=%s trades_count=%s position_opened=%s base_score=%.6f diag_score=%.6f penalty=%.3f flags=%s",
            i,
            int(metrics.get("signals_count_enter", 0)),
            int(metrics.get("signals_count_exit", 0)),
            int(metrics.get("entry_events_count", metrics.get("signals_count_enter", 0))),
            int(metrics.get("exit_events_count", metrics.get("signals_count_exit", 0))),
            int(metrics.get("trades_count", 0)),
            str(position_opened).lower(),
            float(metrics.get("base_score", SOFT_FAIL_SCORE)),
            float(metrics.get("diag_score", SOFT_FAIL_SCORE)),
            float(penalty),
            _flags_to_short(flags),
        )

        trial_duration = round(max(time.monotonic() - trial_started_at, 0.01), 3)

        last_trades_path = run_dir / "trades_last.json"
        last_equity_path = run_dir / "equity_last.csv"
        if trades:
            last_trades_path.write_text(json.dumps(trades, indent=2), encoding="utf-8")
        else:
            last_trades_path.write_text("[]", encoding="utf-8")
        if eq_df is not None and not eq_df.empty:
            eq_df.to_csv(last_equity_path)
        else:
            eq_df = None

        trial_snap = _trial_snapshot(i, float(diag_score), note, reason, metrics, cfg, diag_metrics=diag_metrics, flags=flags)
        summary_payload = {
            "trial_number": int(i),
            "params": trial_snap.get("params", {}),
            "note": note,
            "reason": reason,
            "base_score": float(metrics.get("base_score", SOFT_FAIL_SCORE)),
            "diag_score": float(metrics.get("diag_score", SOFT_FAIL_SCORE)),
            "diag_penalty": float(metrics.get("diag_penalty", 0.0)),
            "score_direction": "maximize" if score_maximize else "minimize",
            "diag_metrics": diag_metrics,
            "flags": flags,
        }
        summary_path = _save_trial_summary(run_dir, i, summary_payload)

        should_save_heavy = bool(SETTINGS.diag_save_all or len(leaderboard_rows) < diag_top_k)
        if not should_save_heavy and leaderboard_rows:
            worst_diag = min(float(r.get("diag_score", SOFT_FAIL_SCORE)) for r in leaderboard_rows) if score_maximize else max(float(r.get("diag_score", abs(SOFT_FAIL_SCORE))) for r in leaderboard_rows)
            should_save_heavy = float(diag_score) >= worst_diag if score_maximize else float(diag_score) <= worst_diag
        trial_eq_path = None
        trial_tr_path = None
        if should_save_heavy:
            trial_eq_path, trial_tr_path = _save_trial_heavy_artifacts(run_dir, i, eq_df, trades)

        leaderboard_row = {
            "trial": int(i),
            "base_score": float(metrics.get("base_score", SOFT_FAIL_SCORE)),
            "diag_score": float(metrics.get("diag_score", SOFT_FAIL_SCORE)),
            "final_equity": float(diag_metrics.get("final_equity", metrics.get("final_equity", 0.0))),
            "total_return": float(diag_metrics.get("total_return", 0.0)),
            "n_trades": int(diag_metrics.get("n_trades", metrics.get("trades_count", 0))),
            "PF": float(diag_metrics.get("profit_factor", 0.0)),
            "maxDD": float(diag_metrics.get("max_drawdown", 0.0)),
            "winrate": float(diag_metrics.get("winrate", 0.0)),
            "expectancy$": float(diag_metrics.get("expectancy$", 0.0)),
            "exposure": float(diag_metrics.get("exposure", metrics.get("exposure", 0.0))),
            "sharpe": float(diag_metrics.get("sharpe", metrics.get("sharpe", 0.0))),
            "stability_score": float(metrics.get("stability_score", 0.0)),
            "baseline_gap": float(metrics.get("final_equity", 0.0) - metrics.get("buyhold_final_equity", SETTINGS.initial_cash)),
            "top3_contribution": float(diag_metrics.get("top3_contribution", 0.0)),
            "flags": _flags_to_short(flags),
            "score_direction": "maximize" if score_maximize else "minimize",
            "flags_map": flags,
            "summary_path": str(summary_path),
            "equity_path": trial_eq_path,
            "trades_path": trial_tr_path,
        }
        leaderboard_rows.append(leaderboard_row)

        last_trial_payload = {
            "number": i,
            "score": float(diag_score),
            "base_score": float(metrics.get("base_score", SOFT_FAIL_SCORE)),
            "diag_score": float(metrics.get("diag_score", SOFT_FAIL_SCORE)),
            **trial_snap,
            "duration_sec": trial_duration,
            "error": error_text,
            "traceback": traceback_text,
            "trades_path": str(last_trades_path),
            "equity_path": str(last_equity_path) if (eq_df is not None and not eq_df.empty) else None,
        }
        repo.update_progress_last_trial(job_id, last_trial_payload)

        if note == "exception" and repeated_exception_count >= MAX_IDENTICAL_EXCEPTIONS:
            fail_reason = (
                f"Повторяющаяся ошибка trial x{repeated_exception_count}: {error_text}. "
                "Оптимизация остановлена автоматически."
            )
            repo.update_progress(
                job_id=job_id,
                trials_done=i,
                trials_total=trials_total,
                last_score=diag_score,
                best_score=best_score,
                state="failed",
                reason=fail_reason,
            )
            repo.set_job_status(job_id, "failed")
            return {"status": "failed", "reason": fail_reason}

        last_score = float(diag_score)
        study.tell(trial, float(diag_score))

        if _is_better_score(float(diag_score), None if best_overall is None else float(best_overall.get("diag_score", 0.0)), score_maximize):
            best_overall = trial_snap

        if _is_better_score(float(diag_score), None if best_diag is None else float(best_diag.get("diag_score", 0.0)), score_maximize):
            best_diag = trial_snap

        is_valid_trial = bool(
            note == "ok"
            and np.isfinite(diag_score)
            and int(metrics.get("trades_count", 0)) >= MIN_TRADES_REQUIRED
        )
        if is_valid_trial:
            if _is_better_score(float(diag_score), best_score, score_maximize):
                best_score = float(diag_score)
                best_metrics = metrics
                best_config = cfg
                best_valid = trial_snap
                equity_path, trades_path, config_path, _summary_path = save_best_artifacts(
                    job_id=job_id,
                    equity_df=eq_df,
                    trades=trades,
                    best_config=best_config,
                    runs_dir=SETTINGS.runs_dir,
                    best_metrics=best_metrics,
                )
                repo.save_best(job_id, best_config, best_metrics, equity_path, trades_path, config_path)

        top5_diag = _sort_rows_by_key(leaderboard_rows, "diag_score", maximize=score_maximize)[:5]
        repo.update_progress(
            job_id=job_id,
            trials_done=i,
            trials_total=trials_total,
            last_score=float(diag_score),
            best_score=best_score,
            state="running",
            extra={"best_valid": best_valid, "best_overall": best_overall, "best_diag": best_diag, "top_diag": top5_diag},
        )

        if i % 100 == 0:
            _LOG.info("progress job=%s trials=%s/%s best_diag=%s", job_id, i, trials_total, best_score)

        if i % checkpoint_n == 0 and best_score is not None:
            repo.add_checkpoint(job_id, checkpoint_no=i // checkpoint_n, trials_done=i)

    if trials_done > 0 and best_overall is None:
        best_overall = {
            "number": trials_done,
            "score": float(last_score),
            "note": "unknown",
            "reason": "unknown",
            "trades_count": 0,
            "closed_trades_count": 0,
            "signals_count_enter": 0,
            "signals_count_exit": 0,
            "entry_events_count": 0,
            "exit_events_count": 0,
            "forced_exit_count": 0,
            "final_equity": float(SETTINGS.initial_cash),
            "profit_$": 0.0,
            "profit_%": 0.0,
            "start_cash": float(SETTINGS.initial_cash),
            "params": {},
        }

    leaderboard_sorted = _sort_rows_by_key(leaderboard_rows, "diag_score", maximize=score_maximize)
    leaderboard_csv_path = run_dir / "leaderboard.csv"
    leaderboard_json_path = run_dir / "leaderboard.json"
    if leaderboard_sorted:
        pd.DataFrame(leaderboard_sorted).to_csv(leaderboard_csv_path, index=False)
        leaderboard_json_path.write_text(json.dumps(leaderboard_sorted, indent=2), encoding="utf-8")

    leaders = _select_leaders(leaderboard_sorted, maximize=score_maximize)

    # Build diagnostic plots only for top N by diag_score.
    for row in leaderboard_sorted[:diag_top_plot_k]:
        trial_n = int(row.get("trial", 0))
        trial_dir = run_dir / "trials" / f"trial_{trial_n:05d}"
        eq_path = trial_dir / "equity.csv"
        tr_path = trial_dir / "trades.json"
        if not eq_path.exists() or not tr_path.exists():
            continue
        try:
            eq_df = pd.read_csv(eq_path, index_col=0, parse_dates=True)
            trades = json.loads(tr_path.read_text(encoding="utf-8"))
            if "Close" not in eq_df.columns and "equity" in eq_df.columns:
                eq_df = pd.concat([eq_df, eq_df[["equity"]].rename(columns={"equity": "Close"})], axis=1)
            save_diagnostic_artifacts(
                run_dir=trial_dir,
                bt=eq_df,
                trades=trades,
                initial_cash=float(SETTINGS.initial_cash),
                allow_short=True,
                metrics={
                    "fills_open": int(row.get("n_trades", 0)),
                    "fills_close": int(row.get("n_trades", 0)),
                    "trades_closed": int(row.get("n_trades", 0)),
                },
            )
        except Exception:
            _LOG.exception("failed to build top diagnostic artifacts for trial=%s", trial_n)

    top5_diag = leaderboard_sorted[:5]
    if best_score is None:
        repo.update_progress(
            job_id=job_id,
            trials_done=trials_done,
            trials_total=trials_total,
            last_score=last_score,
            best_score=None,
            state="finished_no_results",
            reason="Ни один trial не дал даже конечный score (exception/nan).",
            extra={
                "best_valid": best_valid,
                "best_overall": best_overall,
                "best_diag": best_diag,
                "top_diag": top5_diag,
                "leaderboard_path": str(leaderboard_csv_path),
                "leaders": leaders,
            },
        )
        repo.set_job_status(job_id, "finished_no_results")

        with open(run_dir / "summary.txt", "w", encoding="utf-8") as f:
            f.write("Optimization finished with no valid results\n")
            f.write("Reason: no finite/valid score\n")
            if top5_diag:
                f.write("Top by diag_score:\n")
                for row in top5_diag:
                    f.write(
                        f"trial={row['trial']} diag={row['diag_score']:.6f} base={row['base_score']:.6f} "
                        f"ret={row['total_return']:.2%} pf={row['PF']:.3f} trades={row['n_trades']} flags={row['flags']}\n"
                    )
            f.write("Leaders:\n")
            for k, row in leaders.items():
                if not row:
                    continue
                f.write(
                    f"{k}: trial={row['trial']} equity={row['final_equity']:.2f} ret={row['total_return']:.2%} "
                    f"base={row['base_score']:.6f} diag={row['diag_score']:.6f} pf={row['PF']:.3f} "
                    f"maxDD={row['maxDD']:.2%} trades={row['n_trades']} flags={row['flags']}\n"
                )
            f.write(f"leaderboard_csv: {leaderboard_csv_path}\n")
        return {"status": "finished_no_results", "reason": "no finite scores", "leaderboard": str(leaderboard_csv_path), "leaders": leaders}

    repo.update_progress(
        job_id=job_id,
        trials_done=trials_done,
        trials_total=trials_total,
        last_score=last_score,
        best_score=best_score,
        state="finished",
        extra={
            "best_valid": best_valid,
            "best_overall": best_overall,
            "best_diag": best_diag,
            "top_diag": top5_diag,
            "leaderboard_path": str(leaderboard_csv_path),
            "leaders": leaders,
        },
    )
    repo.set_job_status(job_id, "finished")

    run_dir = Path(SETTINGS.runs_dir) / job_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "summary.txt", "a", encoding="utf-8") as f:
        f.write("Optimization finished\n")
        f.write(json.dumps(best_metrics, indent=2) + "\n")
        f.write("Top 5 by diag_score:\n")
        for row in top5_diag:
            f.write(
                f"trial={row['trial']} diag={row['diag_score']:.6f} base={row['base_score']:.6f} "
                f"eq={row['final_equity']:.2f} ret={row['total_return']:.2%} pf={row['PF']:.3f} "
                f"maxDD={row['maxDD']:.2%} trades={row['n_trades']} winrate={row['winrate']:.2%} "
                f"exp$={row['expectancy$']:.2f} top3={row['top3_contribution']:.2%} flags={row['flags']}\n"
            )
        f.write("Leaders:\n")
        for k, row in leaders.items():
            if not row:
                continue
            f.write(
                f"{k}: trial={row['trial']} equity={row['final_equity']:.2f} ret={row['total_return']:.2%} "
                f"base={row['base_score']:.6f} diag={row['diag_score']:.6f} pf={row['PF']:.3f} "
                f"maxDD={row['maxDD']:.2%} trades={row['n_trades']} flags={row['flags']}\n"
            )
        f.write(f"leaderboard_csv: {leaderboard_csv_path}\n")
        f.write(f"leaderboard_json: {leaderboard_json_path}\n")

    return {
        "status": "finished",
        "best_score": best_score,
        "metrics": best_metrics,
        "leaderboard_csv": str(leaderboard_csv_path),
        "leaderboard_json": str(leaderboard_json_path),
        "top_diag": top5_diag,
        "leaders": leaders,
        "score_direction": "maximize" if score_maximize else "minimize",
    }


def _sample(trial: optuna.trial.Trial) -> dict:
    strategy_family = trial.suggest_categorical("strategy_family", ["hybrid_vote", "retrieval"])
    ema_fast = trial.suggest_int("ema_fast", 5, 100)
    ema_slow = trial.suggest_int("ema_slow", 20, 200)
    if ema_slow <= ema_fast:
        ema_slow = ema_fast + 1
    sl_pct = trial.suggest_float("sl_pct", 0.002, 0.03, log=True)
    tp_pct = max(3.0 * sl_pct, 0.01)
    return {
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "rsi_period": trial.suggest_int("rsi_period", 5, 30),
        "buy_below": trial.suggest_int("buy_below", 10, 45),
        "sell_above": trial.suggest_int("sell_above", 55, 90),
        "bb_period": trial.suggest_int("bb_period", 10, 30),
        "bb_std": trial.suggest_float("bb_std", 1.5, 3.0),
        "adx_min": trial.suggest_int("adx_min", 5, 25),
        "adx_period": trial.suggest_int("adx_period", 7, 30),
        "regime_mode": trial.suggest_categorical("regime_mode", ["on", "off"]),
        "enter_long": trial.suggest_int("enter_long", 1, 3),
        "exit_long": trial.suggest_int("exit_long", 1, 3),
        "sl_pct": sl_pct,
        "tp_pct": tp_pct,
        "execution_mode": "next_open",
        "initial_cash": float(SETTINGS.initial_cash),
        "position_size_pct": float(SETTINGS.position_size_pct),
        "allow_short": strategy_family != "retrieval",
        "strategy_family": strategy_family,
        "retrieval_k": trial.suggest_int("retrieval_k", 20, 120),
        "retrieval_horizon": trial.suggest_int("retrieval_horizon", 3, 15),
        "retrieval_min_neighbors": trial.suggest_int("retrieval_min_neighbors", 8, 40),
        "entry_mean_threshold": trial.suggest_float("entry_mean_threshold", 0.0001, 0.004),
        "entry_score_threshold": trial.suggest_float("entry_score_threshold", 0.05, 1.2),
        "exit_score_threshold": trial.suggest_float("exit_score_threshold", -1.0, 0.2),
        "risk_limit": trial.suggest_float("risk_limit", 0.005, 0.04),
        "max_dist_percentile": trial.suggest_float("max_dist_percentile", 0.5, 0.95),
        "hold_bars": trial.suggest_int("hold_bars", 2, 20),
        "embargo_bars": trial.suggest_int("embargo_bars", 1, 8),
        "wf_train_size": trial.suggest_int("wf_train_size", 252, 756),
        "wf_test_size": trial.suggest_int("wf_test_size", 63, 252),
        "wf_window_type": trial.suggest_categorical("wf_window_type", ["expanding", "rolling"]),
    }
