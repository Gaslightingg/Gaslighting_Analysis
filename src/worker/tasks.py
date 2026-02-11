from __future__ import annotations

from datetime import date

from src.data_provider.provider import get_ohlcv
from src.optimizer.optuna_runner import run_optimization_job
from src.storage.repository import Repository
from src.worker.celery_app import app


@app.task(name="optimization.run")
def optimization_run(job_id: str) -> dict:
    repo = Repository()
    data = repo.get_job_data(job_id)
    if not data:
        return {"status": "failed", "error": "job not found"}

    params = data["params"]
    trials_total = int(params.get("trials_total", 0))

    def _set_last_trial_no_data(note_text: str, score: float = -9999.0) -> None:
        repo.update_progress_last_trial(
            job_id,
            {
                "number": 0,
                "score": score,
                "trades_count": 0,
                "params": {},
                "duration_sec": 0.0,
                "note": "no_data",
                "reason": note_text,
            },
        )

    try:
        end_date = date.fromisoformat(params["end"])
        if end_date > date.today():
            repo.set_job_status(job_id, "failed")
            _set_last_trial_no_data("Дата окончания в будущем. Выберите период до сегодняшнего дня.")
            repo.update_progress(
                job_id=job_id,
                trials_done=0,
                trials_total=trials_total,
                last_score=0.0,
                best_score=None,
                state="failed",
                reason="Дата окончания в будущем. Выберите период до сегодняшнего дня.",
            )
            return {"status": "failed", "error": "end date is in the future"}
    except Exception:  # noqa: BLE001
        repo.set_job_status(job_id, "failed")
        repo.update_progress_last_trial(
            job_id,
            {
                "number": 0,
                "score": -9999.0,
                "trades_count": 0,
                "params": {},
                "duration_sec": 0.0,
                "note": "exception",
                "reason": "Некорректные параметры даты",
            },
        )
        return {"status": "failed", "error": "invalid date params"}

    try:
        df = get_ohlcv(params["ticker"], params["start"], params["end"])
        if df.empty:
            repo.set_job_status(job_id, "failed")
            _set_last_trial_no_data("Нет данных за выбранный период.")
            repo.update_progress(
                job_id=job_id,
                trials_done=0,
                trials_total=trials_total,
                last_score=0.0,
                best_score=None,
                state="failed",
                reason="Нет данных за выбранный период.",
            )
            return {"status": "failed", "error": "empty data"}
    except Exception as exc:  # noqa: BLE001
        repo.set_job_status(job_id, "failed")
        repo.update_progress_last_trial(
            job_id,
            {
                "number": 0,
                "score": -9999.0,
                "trades_count": 0,
                "params": {},
                "duration_sec": 0.0,
                "note": "exception",
                "reason": f"Ошибка получения данных: {exc}",
            },
        )
        repo.update_progress(
            job_id=job_id,
            trials_done=0,
            trials_total=trials_total,
            last_score=0.0,
            best_score=None,
            state="failed",
            reason=f"Ошибка получения данных: {exc}",
        )
        return {"status": "failed", "error": str(exc)}

    return run_optimization_job(job_id, df)
