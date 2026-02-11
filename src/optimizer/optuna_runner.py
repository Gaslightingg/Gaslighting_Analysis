from __future__ import annotations

import json
import logging
import time
import traceback
from pathlib import Path

import numpy as np
import optuna
from optuna.samplers import TPESampler

from src.config import SETTINGS
from src.indicators.calculator import add_indicators
from src.optimizer.walk_forward import WalkForwardConfig, evaluate_config_walk_forward
from src.reporter.report import save_best_artifacts
from src.storage.repository import Repository

MAX_IDENTICAL_EXCEPTIONS = 5
TRACEBACK_LIMIT = 2000
_LOG = logging.getLogger("optimizer.optuna")


def _is_valid_trial(score: float, note: str) -> bool:
    # trades==0 is still a valid trial; only non-finite scores and hard exceptions are invalid.
    return bool(np.isfinite(score) and note != "exception")


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
    last_score = 0.0
    trials_done = 0

    repeated_exception_count = 0
    last_exception_key = ""

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SETTINGS.seed))
    wf_cfg = WalkForwardConfig(train_months=12, test_months=3, step_months=3)

    for i in range(1, trials_total + 1):
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
            if i <= 5:
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
            repeated_exception_count = 0
            last_exception_key = ""
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

        if note != "exception":
            if not np.isfinite(score):
                note = "nan_score"
                reason = "score не является конечным числом"
                score = -9999.0
            else:
                enter_signals = int(metrics.get("signals_count_enter", 0)) if metrics else 0
                trades_count = int(metrics.get("trades_count", 0)) if metrics else 0
                if enter_signals == 0:
                    note = "no_entries"
                    reason = "no_entries: 0 entry signals"
                elif trades_count == 0:
                    note = "no_trades"
                    reason = "no_trades: entry signals were present, but no trades executed"

        trial_duration = round(max(time.monotonic() - trial_started_at, 0.01), 3)

        run_dir = Path(SETTINGS.runs_dir) / job_id
        run_dir.mkdir(parents=True, exist_ok=True)
        last_trades_path = run_dir / "trades_last.json"
        last_equity_path = run_dir / "equity_last.csv"
        if trades:
            last_trades_path.write_text(json.dumps(trades, indent=2), encoding="utf-8")
        else:
            last_trades_path.write_text("[]", encoding="utf-8")
        if eq_df is not None and not eq_df.empty:
            eq_df.to_csv(last_equity_path)

        last_trial_payload = {
            "number": i,
            "score": float(score),
            "trades_count": int(metrics.get("trades_count", 0)) if metrics else 0,
            "signals_count_enter": int(metrics.get("signals_count_enter", 0)) if metrics else 0,
            "signals_count_exit": int(metrics.get("signals_count_exit", 0)) if metrics else 0,
            "params": {
                "ema_fast": cfg.get("ema_fast"),
                "ema_slow": cfg.get("ema_slow"),
                "rsi_period": cfg.get("rsi_period"),
                "buy_below": cfg.get("buy_below"),
                "sell_above": cfg.get("sell_above"),
                "bb_period": cfg.get("bb_period"),
                "bb_std": cfg.get("bb_std"),
                "adx_min": cfg.get("adx_min"),
                "regime_mode": cfg.get("regime_mode"),
                "enter_long": cfg.get("enter_long"),
                "exit_long": cfg.get("exit_long"),
            },
            "duration_sec": trial_duration,
            "note": note,
            "reason": reason,
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

        if _is_valid_trial(score, note):
            if best_score is None or score > best_score:
                best_score = score
                best_metrics = metrics
                best_config = cfg
                equity_path, trades_path, config_path, _summary_path = save_best_artifacts(
                    job_id=job_id,
                    equity_df=eq_df,
                    trades=trades,
                    best_config=best_config,
                    runs_dir=SETTINGS.runs_dir,
                )
                repo.save_best(job_id, best_config, best_metrics, equity_path, trades_path, config_path)

        repo.update_progress(
            job_id=job_id,
            trials_done=i,
            trials_total=trials_total,
            last_score=score,
            best_score=best_score,
            state="running",
        )

        if i % checkpoint_n == 0 and best_score is not None:
            repo.add_checkpoint(job_id, checkpoint_no=i // checkpoint_n, trials_done=i)

    if best_score is None:
        repo.update_progress(
            job_id=job_id,
            trials_done=trials_done,
            trials_total=trials_total,
            last_score=last_score,
            best_score=None,
            state="finished_no_results",
            reason="Ни один trial не дал даже конечный score (exception/nan).",
        )
        repo.set_job_status(job_id, "finished_no_results")

        run_dir = Path(SETTINGS.runs_dir) / job_id
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "summary.txt", "w", encoding="utf-8") as f:
            f.write("Optimization finished with no valid results\n")
            f.write("Reason: no finite score\n")
        return {"status": "finished_no_results", "reason": "no finite scores"}

    repo.update_progress(
        job_id=job_id,
        trials_done=trials_done,
        trials_total=trials_total,
        last_score=last_score,
        best_score=best_score,
        state="finished",
    )
    repo.set_job_status(job_id, "finished")

    run_dir = Path(SETTINGS.runs_dir) / job_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "summary.txt", "a", encoding="utf-8") as f:
        f.write("Optimization finished\n")
        f.write(json.dumps(best_metrics, indent=2))

    return {"status": "finished", "best_score": best_score, "metrics": best_metrics}


def _sample(trial: optuna.trial.Trial) -> dict:
    ema_fast = trial.suggest_int("ema_fast", 5, 50)
    ema_slow = trial.suggest_int("ema_slow", 20, 200)
    if ema_slow <= ema_fast:
        ema_slow = ema_fast + 1
    return {
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "rsi_period": trial.suggest_int("rsi_period", 5, 30),
        "buy_below": trial.suggest_int("buy_below", 20, 45),
        "sell_above": trial.suggest_int("sell_above", 55, 90),
        "bb_period": trial.suggest_int("bb_period", 10, 30),
        "bb_std": trial.suggest_float("bb_std", 1.5, 2.5),
        "adx_min": trial.suggest_int("adx_min", 5, 25),
        "regime_mode": trial.suggest_categorical("regime_mode", ["on", "off"]),
        "enter_long": trial.suggest_int("enter_long", 1, 2),
        "exit_long": trial.suggest_int("exit_long", -1, 1),
    }
