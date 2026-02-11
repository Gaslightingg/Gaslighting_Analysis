from __future__ import annotations

from datetime import date, datetime, timezone

from src.data_provider.provider import DataProviderError, get_ohlcv
from src.optimizer.optuna_runner import run_optimization_job
from src.storage.repository import Repository
from src.worker.celery_app import app

MIN_BARS = 60


def _today_utc() -> date:
    return datetime.now(timezone.utc).date()


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
    effective_end = end_date
    clipped_warning: str | None = None
    if end_date > today:
        effective_end = today
        clipped_warning = f"⚠️ Конец периода обрезан до {today.isoformat()}, потому что будущих данных нет."

    if start_date >= effective_end:
        repo.set_job_status(job_id, "failed")
        _set_last_trial("no_data", "future_end_date: после обрезки периода start_date >= end_date")
        repo.update_progress(
            job_id=job_id,
            trials_done=0,
            trials_total=trials_total,
            last_score=0.0,
            best_score=None,
            state="failed",
            reason="Период некорректен: дата начала должна быть раньше даты окончания.",
        )
        return {"status": "failed", "error": "invalid period"}

    if clipped_warning:
        repo.update_progress(
            job_id=job_id,
            trials_done=0,
            trials_total=trials_total,
            last_score=0.0,
            best_score=None,
            state="queued",
            reason=clipped_warning,
        )

    start_s = start_date.isoformat()
    end_s = effective_end.isoformat()

    try:
        df = get_ohlcv(params["ticker"], start_s, end_s)
    except DataProviderError as exc:
        repo.set_job_status(job_id, "failed")
        _set_last_trial("no_data", f"{exc.reason}: {exc}")
        repo.update_progress(
            job_id=job_id,
            trials_done=0,
            trials_total=trials_total,
            last_score=0.0,
            best_score=None,
            state="failed",
            reason=f"{exc.reason}: {exc}",
        )
        return {"status": "failed", "error": str(exc), "reason": exc.reason}
    except Exception as exc:  # noqa: BLE001
        repo.set_job_status(job_id, "failed")
        _set_last_trial("exception", f"network_error: {exc}")
        repo.update_progress(
            job_id=job_id,
            trials_done=0,
            trials_total=trials_total,
            last_score=0.0,
            best_score=None,
            state="failed",
            reason=f"network_error: {exc}",
        )
        return {"status": "failed", "error": str(exc), "reason": "network_error"}

    if df.empty:
        repo.set_job_status(job_id, "finished_no_results")
        _set_last_trial("no_data", "provider_empty: провайдер вернул пустой датасет")
        repo.update_progress(
            job_id=job_id,
            trials_done=0,
            trials_total=trials_total,
            last_score=0.0,
            best_score=None,
            state="finished_no_results",
            reason="provider_empty: нет данных за выбранный период.",
        )
        return {"status": "finished_no_results", "reason": "provider_empty"}

    if len(df) < MIN_BARS:
        repo.set_job_status(job_id, "finished_no_results")
        _set_last_trial("no_data", f"provider_empty: недостаточно баров ({len(df)} < {MIN_BARS})")
        repo.update_progress(
            job_id=job_id,
            trials_done=0,
            trials_total=trials_total,
            last_score=0.0,
            best_score=None,
            state="finished_no_results",
            reason=f"Недостаточно данных для оптимизации: {len(df)} баров (< {MIN_BARS}).",
        )
        return {"status": "finished_no_results", "reason": "not_enough_bars"}

    return run_optimization_job(job_id, df)
