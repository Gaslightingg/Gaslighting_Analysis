from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from src.data_provider.provider import DataProviderError, get_ohlcv_with_meta
from src.optimizer.optuna_runner import run_optimization_job
from src.storage.repository import Repository
from src.worker.celery_app import app

MIN_BARS = 252


def _today_utc() -> date:
    return datetime.now(timezone.utc).date()


def _horizon_to_start(horizon: str) -> str:
    today = _today_utc()
    days = {"1y": 365, "2y": 730, "5y": 365 * 5}.get(horizon, 365 * 5)
    return (today - timedelta(days=days)).isoformat()


@app.task(name="preload.run")
def preload_data_run(job_id: str) -> dict:
    repo = Repository()
    data = repo.get_job_data(job_id)
    if not data:
        return {"status": "failed", "error": "job not found"}

    params = data["params"]
    tickers = [str(t).strip().upper() for t in params.get("tickers", []) if str(t).strip()]
    horizon = str(params.get("horizon", "5y"))
    start = _horizon_to_start(horizon)
    end = _today_utc().isoformat()

    total = len(tickers)
    done = 0
    repo.set_job_status(job_id, "running")

    for ticker in tickers:
        status = "ok"
        reason = ""
        bars_count = 0
        min_date = ""
        max_date = ""
        source = ""
        end_trimmed = False
        try:
            df, meta = get_ohlcv_with_meta(ticker, start, end)
            bars_count = int(meta.get("bars_count", len(df)))
            min_date = str(meta.get("min_date", ""))
            max_date = str(meta.get("max_date", ""))
            source = str(meta.get("source", ""))
            end_trimmed = bool(meta.get("end_trimmed", False))
            if df.empty:
                status = "not_found"
                reason = "provider_empty"
        except DataProviderError as exc:
            status = "not_found" if exc.reason in {"provider_empty", "future_period"} else "error"
            reason = exc.reason
        except Exception as exc:  # noqa: BLE001
            status = "error"
            reason = f"error: {exc}"

        done += 1
        repo.update_preload_progress(
            job_id=job_id,
            done=done,
            total=total,
            item={
                "ticker": ticker,
                "status": status,
                "reason": reason,
                "bars_count": bars_count,
                "min_date": min_date,
                "max_date": max_date,
                "source": source,
                "end_trimmed": end_trimmed,
            },
            state="running",
        )

    repo.set_job_status(job_id, "finished")
    return {"status": "finished", "done": done, "total": total}


@app.task(name="optimization.run")
def optimization_run(job_id: str) -> dict:
    repo = Repository()
    data = repo.get_job_data(job_id)
    if not data:
        return {"status": "failed", "error": "job not found"}

    params = data["params"]
    trials_total = int(params.get("trials_total", 0))

    def _set_last_trial(note: str, reason: str, score: float = -9999.0) -> None:
        repo.update_progress_last_trial(
            job_id,
            {
                "number": 0,
                "score": score,
                "trades_count": 0,
                "params": {},
                "duration_sec": 0.0,
                "note": note,
                "reason": reason,
            },
        )

    try:
        start_date = date.fromisoformat(params["start"])
        end_date = date.fromisoformat(params["end"])
    except Exception:  # noqa: BLE001
        repo.set_job_status(job_id, "failed")
        _set_last_trial("exception", "Некорректные параметры даты")
        return {"status": "failed", "error": "invalid date params"}

    today = _today_utc()
    effective_end = min(end_date, today)
    clipped_warning: str | None = None
    if end_date > today:
        clipped_warning = f"⚠️ Конец периода обрезан до {today.isoformat()}, потому что будущих данных нет."

    if start_date >= effective_end:
        repo.set_job_status(job_id, "finished_no_results")
        _set_last_trial("no_data", "future_period")
        repo.update_progress(
            job_id=job_id,
            trials_done=0,
            trials_total=trials_total,
            last_score=0.0,
            best_score=None,
            state="finished_no_results",
            reason="future_period: дата начала должна быть раньше даты окончания.",
        )
        return {"status": "finished_no_results", "reason": "future_period"}

    start_s = start_date.isoformat()
    end_s = effective_end.isoformat()

    try:
        df, meta = get_ohlcv_with_meta(params["ticker"], start_s, end_s)
        if clipped_warning:
            meta["warning"] = clipped_warning
    except DataProviderError as exc:
        repo.set_job_status(job_id, "finished_no_results")
        _set_last_trial("no_data", exc.reason)
        repo.update_progress(
            job_id=job_id,
            trials_done=0,
            trials_total=trials_total,
            last_score=0.0,
            best_score=None,
            state="finished_no_results",
            reason=f"{exc.reason}: {exc}",
        )
        return {"status": "finished_no_results", "reason": exc.reason}

    if df.empty or len(df) < MIN_BARS:
        reason = "provider_empty" if df.empty else "not_enough_bars"
        repo.set_job_status(job_id, "finished_no_results")
        _set_last_trial("no_data", reason)
        repo.update_progress(
            job_id=job_id,
            trials_done=0,
            trials_total=trials_total,
            last_score=0.0,
            best_score=None,
            state="finished_no_results",
            reason=f"{reason}: bars={len(df)} min_required={MIN_BARS}",
            data_info=meta,
        )
        return {"status": "finished_no_results", "reason": reason}

    repo.update_progress(
        job_id=job_id,
        trials_done=0,
        trials_total=trials_total,
        last_score=0.0,
        best_score=None,
        state="running",
        reason=(clipped_warning or None),
        data_info=meta,
    )

    return run_optimization_job(job_id, df)
