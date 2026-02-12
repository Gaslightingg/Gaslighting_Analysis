from __future__ import annotations

import json
import logging
import time
import traceback
from pathlib import Path

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from src.config import SETTINGS
from src.indicators.calculator import add_indicators
from src.optimizer.walk_forward import WalkForwardConfig, evaluate_config_walk_forward
from src.reporter.report import save_best_artifacts
from src.storage.repository import Repository

MAX_IDENTICAL_EXCEPTIONS = 5
TRACEBACK_LIMIT = 2000
_LOG = logging.getLogger("optimizer.optuna")
MIN_TRADES_REQUIRED = 20


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
    out["score"] = _safe_float(out.get("score", -9999.0), -9999.0)
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


def _trial_snapshot(number: int, score: float, note: str, reason: str, metrics: dict, cfg: dict) -> dict:
    return {
        "number": int(number),
        "score": float(score),
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
        },
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
    wf_cfg = WalkForwardConfig(train_months=12, test_months=3, step_months=3, folds=wf_folds)

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
                wf_cfg,
            )
            trial.report(float(score), step=1)
            if trial.should_prune():
                raise optuna.TrialPruned()
            repeated_exception_count = 0
            last_exception_key = ""
        except optuna.TrialPruned:
            score = -9999.0
            note = "pruned"
            reason = "trial pruned by MedianPruner"
        except Exception as exc:  # noqa: BLE001
            score = -9999.0
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
                score = -9999.0
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
                    score = min(float(score), -1000.0)
                elif trades_count == 0:
                    note = "no_trades"
                    reason = "no_trades: entry signals were present, but no trades executed"
                    score = min(float(score), -500.0)

        metrics["score"] = float(score)
        metrics["reason"] = note if note else "ok"

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
            "trial_diag_end number=%s entry_signals=%s exit_signals=%s entry_events=%s exit_events=%s trades_count=%s position_opened=%s exits_by_rule=%s exits_by_sl_tp=%s exits_forced_end=%s exits_flip=%s reason=%s",
            i,
            int(metrics.get("signals_count_enter", 0)),
            int(metrics.get("signals_count_exit", 0)),
            int(metrics.get("entry_events_count", metrics.get("signals_count_enter", 0))),
            int(metrics.get("exit_events_count", metrics.get("signals_count_exit", 0))),
            int(metrics.get("trades_count", 0)),
            str(position_opened).lower(),
            int(metrics.get("exits_by_rule", 0)),
            int(metrics.get("exits_by_sl_tp", 0)),
            int(metrics.get("exits_forced_end", metrics.get("forced_exit_count", 0))),
            int(metrics.get("exits_flip", 0)),
            note,
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

        last_trial_payload = {
            "number": i,
            "score": float(score),
            **_trial_snapshot(i, float(score), note, reason, metrics, cfg),
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
                last_score=score,
                best_score=best_score,
                state="failed",
                reason=fail_reason,
            )
            repo.set_job_status(job_id, "failed")
            return {"status": "failed", "reason": fail_reason}

        last_score = score
        study.tell(trial, score)

        trial_snap = _trial_snapshot(i, float(score), note, reason, metrics, cfg)
        if best_overall is None or float(score) > float(best_overall.get("score", -np.inf)):
            best_overall = trial_snap

        is_valid_trial = bool(
            note == "ok"
            and np.isfinite(score)
            and int(metrics.get("trades_count", 0)) >= MIN_TRADES_REQUIRED
        )
        if is_valid_trial:
            if best_score is None or score > best_score:
                best_score = score
                best_metrics = metrics
                best_config = cfg
                best_valid = _trial_snapshot(i, float(score), note, reason, metrics, cfg)
                equity_path, trades_path, config_path, _summary_path = save_best_artifacts(
                    job_id=job_id,
                    equity_df=eq_df,
                    trades=trades,
                    best_config=best_config,
                    runs_dir=SETTINGS.runs_dir,
                    best_metrics=best_metrics,
                )
                repo.save_best(job_id, best_config, best_metrics, equity_path, trades_path, config_path)

        repo.update_progress(
            job_id=job_id,
            trials_done=i,
            trials_total=trials_total,
            last_score=score,
            best_score=best_score,
            state="running",
            extra={"best_valid": best_valid, "best_overall": best_overall},
        )

        if i % 100 == 0:
            _LOG.info("progress job=%s trials=%s/%s best=%s", job_id, i, trials_total, best_score)

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

    if best_score is None:
        repo.update_progress(
            job_id=job_id,
            trials_done=trials_done,
            trials_total=trials_total,
            last_score=last_score,
            best_score=None,
            state="finished_no_results",
            reason="Ни один trial не дал даже конечный score (exception/nan).",
            extra={"best_valid": best_valid, "best_overall": best_overall},
        )
        repo.set_job_status(job_id, "finished_no_results")

        with open(run_dir / "summary.txt", "w", encoding="utf-8") as f:
            f.write("Optimization finished with no valid results\n")
            f.write("Reason: no finite/valid score\n")
        return {"status": "finished_no_results", "reason": "no finite scores"}

    repo.update_progress(
        job_id=job_id,
        trials_done=trials_done,
        trials_total=trials_total,
        last_score=last_score,
        best_score=best_score,
        state="finished",
        extra={"best_valid": best_valid, "best_overall": best_overall},
    )
    repo.set_job_status(job_id, "finished")

    run_dir = Path(SETTINGS.runs_dir) / job_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "summary.txt", "a", encoding="utf-8") as f:
        f.write("Optimization finished\n")
        f.write(json.dumps(best_metrics, indent=2))

    return {"status": "finished", "best_score": best_score, "metrics": best_metrics}


def _sample(trial: optuna.trial.Trial) -> dict:
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
        "allow_short": True,
    }
